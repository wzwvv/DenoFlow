import csv
import json
import os
import re
import time
from typing import Dict, Iterable

import numpy as np
import torch
import yaml

from etc.global_config import config
from Data.core import attach_generated_signals, build_experiment_signals, load_split, resolve_view_pair
from Data.signal_ops import set_random_seed
from Support.config import (
    get_augmentation_methods,
    get_downstream_methods,
    get_method_view_override,
    get_evaluation_pairs,
    get_experiment_config,
    get_output_paths,
    get_noise_config,
    resolve_dataset_name,
    resolve_device,
    resolve_fold_ids,
    resolve_subject_ids,
)
from Trainers.Augmentation.core import train_augmentation_method
from Trainers.Classification.core import evaluate_downstream, get_downstream_method_family
from Support.denoise_metrics import (
    append_denoise_result,
    compute_denoise_metrics,
    initialize_denoise_results_file,
)


RESULT_COLUMNS = [
    'timestamp',
    'split_mode',
    'task_mode',
    'augmentation_task',
    'subject',
    'fold',
    'target_subject',
    'augmenter',
    'source_name',
    'train_view',
    'test_view',
    'downstream_method',
    'train_samples',
    'test_samples',
    'acc',
    'itr',
    'extra_metrics',
]


def _slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r'[^A-Za-z0-9._+-]+', '-', str(value)).strip('-._')
    if not slug:
        slug = 'default'
    return slug[:max_length]


def _summarize_methods(methods, prefix: str) -> str:
    methods = [str(method) for method in methods if str(method).strip()]
    if not methods:
        return f'{prefix}-none'
    if len(methods) == 1:
        return f'{prefix}-{_slugify(methods[0], 36)}'
    return f'{prefix}-{len(methods)}items'


def _build_run_folder_name(config: Dict) -> str:
    experiment_cfg = get_experiment_config(config)
    task_mode = experiment_cfg.get('task_mode', 'downstream_only')
    downstream_methods = get_downstream_methods(config)
    augmentation_methods = get_augmentation_methods(config) if task_mode == 'augmentation' else []
    timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    seed_part = f"seed-{int(experiment_cfg.get('random_seed', 0))}"
    parts = [
        timestamp,
        seed_part,
        _summarize_methods(downstream_methods, 'down'),
    ]
    # Without this, two concurrent runs that share seed / downstream set / augmenter
    # land in the same folder whenever they reach this line in the same second, and
    # silently interleave their rows into one CSV. Staggering the launches does not
    # help: the child spends several seconds importing torch before it gets here.
    run_name = experiment_cfg.get('run_name')
    if run_name:
        parts.append(f'run-{_slugify(str(run_name), 40)}')
    if augmentation_methods:
        parts.append(_summarize_methods(augmentation_methods, 'aug'))
    return '__'.join(parts)


def _prepare_output_paths(config: Dict) -> Dict:
    outputs_cfg = get_output_paths(config)
    output_root = outputs_cfg.get('root')
    if not output_root:
        raise KeyError('Missing paths.outputs.root in config.yaml')

    experiment_cfg = get_experiment_config(config)
    dataset_name = resolve_dataset_name(config)
    split_mode = experiment_cfg.get('split_mode', 'intra')
    task_mode = experiment_cfg.get('task_mode', 'downstream_only')
    augmentation_task = experiment_cfg.get('augmentation_task', 'standard') if task_mode == 'augmentation' else 'standard'
    run_root = os.path.join(
        output_root,
        _slugify(dataset_name, 32),
        _slugify(split_mode, 32),
        _slugify(task_mode, 32),
        _slugify(augmentation_task, 32),
        _build_run_folder_name(config),
    )
    runtime_paths = {
        'root': run_root,
        'results': os.path.join(run_root, 'results'),
        'config': os.path.join(run_root, 'config'),
        'generated': os.path.join(run_root, 'generated'),
        'checkpoints': os.path.join(run_root, 'checkpoints'),
    }
    for key in ('root', 'results', 'config'):
        os.makedirs(runtime_paths[key], exist_ok=True)
    config.setdefault('_runtime', {})['output_paths'] = runtime_paths
    return runtime_paths


def _get_output_root(config: Dict) -> str:
    output_root = get_output_paths(config).get('root')
    if not output_root:
        raise KeyError('Missing resolved output root in config.')
    os.makedirs(output_root, exist_ok=True)
    return output_root


def get_results_path(config: Dict) -> str:
    experiment_cfg = get_experiment_config(config)
    outputs_cfg = get_output_paths(config)
    filename = experiment_cfg.get('results_filename', 'experiment_results.csv')
    result_dir = outputs_cfg.get('results', _get_output_root(config))
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, filename)


def initialize_results_file(config: Dict):
    result_path = get_results_path(config)
    if os.path.exists(result_path):
        return result_path
    with open(result_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
    return result_path


def append_result(config: Dict, row: Dict):
    result_path = initialize_results_file(config)
    payload = {column: row.get(column, '') for column in RESULT_COLUMNS}
    if isinstance(payload['extra_metrics'], dict):
        payload['extra_metrics'] = json.dumps(payload['extra_metrics'], ensure_ascii=False, sort_keys=True)
    with open(result_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writerow(payload)


def save_config_snapshot(config: Dict):
    outputs_cfg = get_output_paths(config)
    config_dir = outputs_cfg.get('config', _get_output_root(config))
    os.makedirs(config_dir, exist_ok=True)
    snapshot_path = os.path.join(config_dir, 'experiment_config_snapshot.yaml')
    snapshot_cfg = {key: value for key, value in config.items() if key != '_runtime'}
    with open(snapshot_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(snapshot_cfg, handle, allow_unicode=True, sort_keys=False)


def _build_split_iterator(config: Dict) -> Iterable[Dict]:
    experiment_cfg = get_experiment_config(config)
    split_mode = experiment_cfg.get('split_mode', 'intra')
    subject_ids = resolve_subject_ids(config)
    fold_ids = resolve_fold_ids(config)
    if split_mode == 'intra':
        for fold_id in fold_ids:
            for subject_id in subject_ids:
                yield {'split_mode': split_mode, 'subject_id': subject_id, 'fold_id': fold_id, 'target_subject_id': None}
    elif split_mode == 'loso':
        for fold_id in fold_ids:
            for target_subject_id in subject_ids:
                yield {'split_mode': split_mode, 'subject_id': None, 'fold_id': fold_id, 'target_subject_id': target_subject_id}
    else:
        raise ValueError(f'Unsupported split mode: {split_mode}')


def _build_split_tag(split_kwargs: Dict) -> str:
    if split_kwargs['split_mode'] == 'intra':
        return f"intra-sub{split_kwargs['subject_id']}-fold{split_kwargs['fold_id']}"
    return f"loso-target{split_kwargs['target_subject_id']}-fold{split_kwargs['fold_id']}"


def _save_generated_arrays(config: Dict, split_tag: str, method: str, generated):
    experiment_cfg = get_experiment_config(config)
    if not experiment_cfg.get('save_generated_signals', False):
        return
    generated_root = get_output_paths(config).get('generated')
    if not generated_root:
        return
    split_dir = os.path.join(generated_root, _slugify(split_tag, 64))
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, f'{_slugify(method, 64)}-train.npy'), generated.train)
    np.save(os.path.join(split_dir, f'{_slugify(method, 64)}-test.npy'), generated.test)


def run_experiment(config: Dict):
    experiment_cfg = get_experiment_config(config)
    torch.set_num_threads(int(experiment_cfg.get('num_threads', 5)))
    runtime_paths = _prepare_output_paths(config)
    device = resolve_device(config)
    initialize_results_file(config)
    save_config_snapshot(config)

    task_mode = experiment_cfg.get('task_mode', 'downstream_only')
    augmentation_task = experiment_cfg.get('augmentation_task', 'denoise')
    if task_mode == 'augmentation' and augmentation_task == 'denoise':
        initialize_denoise_results_file(config)
    downstream_methods = get_downstream_methods(config)
    if not downstream_methods:
        raise ValueError('No downstream methods were configured in experiment.methods.')
    evaluation_pairs = get_evaluation_pairs(config)
    augmentation_methods = get_augmentation_methods(config) if task_mode == 'augmentation' else []
    random_seed = int(experiment_cfg.get('random_seed', 0))

    print(f'Experiment device={device}')
    print(f'Experiment task_mode={task_mode}, augmentation_task={augmentation_task}')
    print(f'Results dir={runtime_paths["root"]}')
    print(f'Downstream methods: {downstream_methods}')
    if augmentation_methods:
        print(f'Augmentation methods: {augmentation_methods}')

    for split_index, split_kwargs in enumerate(_build_split_iterator(config)):
        split_seed = random_seed + split_index
        set_random_seed(split_seed)
        split_tag = _build_split_tag(split_kwargs)
        # if split_kwargs['fold_id']*100+split_kwargs['target_subject_id'] < 360:
        #     #print(split_kwargs['fold_id']*100+split_kwargs['target_subject_id'])
        #     continue
        print(f'Running split {split_tag}')

        split = load_split(
            config,
            split_kwargs['split_mode'],
            subject_id=split_kwargs['subject_id'],
            fold_id=split_kwargs['fold_id'],
            target_subject_id=split_kwargs['target_subject_id'],
        )
        base_signals = build_experiment_signals(config, split)

        signal_variants = [(None, base_signals, {})]
        if task_mode == 'augmentation':
            signal_variants = []
            for augmenter in augmentation_methods:
                set_random_seed(split_seed)
                print(f'Training augmenter {augmenter}')
                augmentation_output = train_augmentation_method(augmenter, base_signals, config, device, split_tag)
                generated_signals = attach_generated_signals(
                    base_signals,
                    augmentation_output.generated.train,
                    augmentation_output.generated.test,
                    augmenter,
                )
                _save_generated_arrays(config, split_tag, augmenter, augmentation_output.generated)

                if augmentation_task == 'denoise':
                    try:
                        denoise_metrics = compute_denoise_metrics(
                            clean=base_signals.clean.test,
                            denoised=augmentation_output.generated.test,
                            noisy=base_signals.source.test,
                        )
                        # artifact_type lives under augmentation.noise, not experiment,
                        # so reading it off experiment_cfg left this column blank in every
                        # denoise_results.csv written so far.
                        artifact_type = get_noise_config(config).get('artifact_type', '')
                        append_denoise_result(config, {
                            'timestamp':       time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                            'split_mode':      split_kwargs['split_mode'],
                            'subject':         split_kwargs['subject_id'] or '',
                            'fold':            split_kwargs['fold_id'],
                            'target_subject':  split_kwargs['target_subject_id'] or '',
                            'augmenter':       augmenter,
                            'artifact_type':   artifact_type,
                            'num_samples':     denoise_metrics['num_samples'],
                            'cc':              f"{denoise_metrics['cc']:.6f}",
                            'rrmse':           f"{denoise_metrics['rrmse']:.6f}",
                            'snr_in':          f"{denoise_metrics['snr_in']:.4f}",
                            'snr_out':         f"{denoise_metrics['snr_out']:.4f}",
                            'prd':             f"{denoise_metrics['prd']:.4f}",
                            'snri':            f"{denoise_metrics['snri']:.4f}",
                            'nrr':             f"{denoise_metrics['nrr']:.6f}",
                            'snr_in_global':   f"{denoise_metrics['snr_in_global']:.4f}",
                            'snr_out_global':  f"{denoise_metrics['snr_out_global']:.4f}",
                            'snri_global':     f"{denoise_metrics['snri_global']:.4f}",
                        })
                        print(
                            f"[denoise:{augmenter}] PRD={denoise_metrics['prd']:.2f}% "
                            f"SNRI={denoise_metrics['snri']:.2f}dB "
                            f"NRR={denoise_metrics['nrr']:.4f}"
                        )
                    except Exception as exc:
                        print(f'[denoise metrics] skipped for {augmenter}: {exc}')

                signal_variants.append((augmenter, generated_signals, augmentation_output.metrics))

        seen_context = set()
        for augmenter, signals, augmentation_metrics in signal_variants:
            for pair in evaluation_pairs:
                # A view pair that never reads the generated signal is the same for every
                # augmenter, so evaluate it once and record it without an augmenter.
                context_only = 'generated' not in (pair['train_view'], pair['test_view'])
                if context_only:
                    context_key = (pair['train_view'], pair['test_view'])
                    if context_key in seen_context:
                        continue
                    seen_context.add(context_key)
                row_augmenter = None if context_only else augmenter
                print(f"Base views train={pair['train_view']}, test={pair['test_view']}")
                for method in downstream_methods:
                    set_random_seed(split_seed)
                    method_family = get_downstream_method_family(method)
                    method_train_view, method_test_view = get_method_view_override(
                        config,
                        method_family,
                        pair['train_view'],
                        pair['test_view'],
                    )
                    train_x, train_y, test_x, test_y, train_view, test_view = resolve_view_pair(
                        signals,
                        method_train_view,
                        method_test_view,
                    )

                    eval_kwargs = {}
                    method_metrics = {}
                    if method_family == 'hybrid':
                        knowledge_train_view_req, knowledge_test_view_req = get_method_view_override(
                            config,
                            'hybrid_knowledge',
                            train_view,
                            test_view,
                        )
                        knowledge_train_x, knowledge_train_y, knowledge_test_x, knowledge_test_y, knowledge_train_view, knowledge_test_view = resolve_view_pair(
                            signals,
                            knowledge_train_view_req,
                            knowledge_test_view_req,
                        )
                        if method.endswith('+EnhSSVEPformer') and (
                            knowledge_train_x.shape[-1] != train_x.shape[-1] or knowledge_test_x.shape[-1] != test_x.shape[-1]
                        ):
                            raise ValueError(
                                f"{method} requires hybrid_knowledge views to share the same time dimension as the main hybrid views. "
                                f"Got train {knowledge_train_x.shape[-1]} vs {train_x.shape[-1]}, "
                                f"test {knowledge_test_x.shape[-1]} vs {test_x.shape[-1]}."
                            )

                        eval_kwargs = {
                            'knowledge_train_data': knowledge_train_x,
                            'knowledge_train_labels': knowledge_train_y,
                            'knowledge_test_data': knowledge_test_x,
                            'knowledge_test_labels': knowledge_test_y,
                        }
                        method_metrics.update({
                            'knowledge_train_view': knowledge_train_view,
                            'knowledge_test_view': knowledge_test_view,
                        })
                        print(
                            f'Evaluating {method} train={train_view} test={test_view} ' 
                            f'knowledge_train={knowledge_train_view} knowledge_test={knowledge_test_view}'
                        )
                    else:
                        print(f'Evaluating {method} train={train_view} test={test_view}')

                    result = evaluate_downstream(
                        method,
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        signals.meta,
                        config,
                        device,
                        **eval_kwargs,
                    )
                    row = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                        'split_mode': split_kwargs['split_mode'],
                        'task_mode': task_mode,
                        'augmentation_task': augmentation_task if task_mode == 'augmentation' else '',
                        'subject': split_kwargs['subject_id'] or '',
                        'fold': split_kwargs['fold_id'],
                        'target_subject': split_kwargs['target_subject_id'] or '',
                        'augmenter': row_augmenter or '',
                        'source_name': signals.source_name,
                        'train_view': train_view,
                        'test_view': test_view,
                        'downstream_method': method,
                        'train_samples': len(train_y),
                        'test_samples': len(test_y),
                        'acc': result.acc,
                        'itr': result.itr,
                        'extra_metrics': {**augmentation_metrics, **method_metrics, **result.extra_metrics},
                    }
                    append_result(config, row)

    print('Experiment finished.')


def run():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    run_experiment(config)


if __name__ == '__main__':
    run()
