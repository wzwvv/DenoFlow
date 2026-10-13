import itertools
from typing import Dict, List, Tuple

import torch


TRADITIONAL_METHODS = ['FBCCA', 'CCA', 'MSI', 'TRCA', 'TDCA']
DEEP_METHODS = ['EEGNet', 'CCNN', 'FBtCNN', 'ConvCA', 'SSVEPNet', 'SSVEPformer',
                'DDGCNN', 'DualBranchNet']
HYBRID_KNOWLEDGE_METHODS = ['TRCA', 'TDCA', 'TD-RCA']
HYBRID_DEEP_METHODS = ['EEGNet', 'CCNN', 'FBtCNN', 'SSVEPNet', 'SSVEPformer', 'DDGCNN', 'EnhSSVEPformer']
DIFFUSION_METHODS = ['DiffusionClassifier']
DIRECT_AUGMENTATION_METHODS = ['Scale', 'Noise', 'Flip', 'FShift', 'DWTaug', 'HHTaug']
STATEFUL_AUGMENTATION_METHODS = [
    'CNN-BASE',
    'CNN-GAN',
    'CNN-VAE',
    'CNN-GAN-VAE',
    'CNN-LSTM_BASE',
    'CNN-LSTM_GAN',
    'CNN-LSTM_VAE',
    'CNN-LSTM_GAN-VAE',
    'CNN-Trans_BASE',
    'CNN-Trans_GAN',
    'CNN-Trans_VAE',
    'CNN-Trans_GAN-VAE',
]
# The seven published denoisers the paper compares against, as peers. Listing them in one
# place keeps the comparison set explicit: a method belongs here because it was published as
# an EEG denoiser, not because of how it happens to be implemented internally.
PUBLISHED_BASELINES = [
    'TOEDGAN',
    'EEGDNet',
    'GCTNet',
    'CT_DCENet',
    'FCNN',
    'SimpleCNN',
    'ResCNN',
]

# Conditional generative denoisers: rectified flow and DDPM, each optionally with class
# supervision and adversarial supervision on the sampled output.
GENERATIVE_AUGMENTATION_METHODS = [
    'FLOW',
    'FLOW_CLS',
    'FLOW_GAN',
    'DIFF',
    'DIFF_CLS',
    'DIFF_GAN',
]

AUGMENTATION_METHODS = (
    DIRECT_AUGMENTATION_METHODS
    + STATEFUL_AUGMENTATION_METHODS
    + PUBLISHED_BASELINES
    + GENERATIVE_AUGMENTATION_METHODS
    + [
        'Diffusion',
        'DiffusionGAN',
        'GeneratorCNN',
        'GeneratorTransformer',
        'NovelCNN',
        'DuoCL',
    ]
)
DATASET_NAME_TO_ID = {
    'benchmark12': 1,
    'benchmark40': 2,
    'beta': 3,
}
CLASSIFICATION_MODEL_KEYS = {
    'EEGNet': 'eegnet',
    'CCNN': 'ccnn',
    'FBtCNN': 'fbtcnn',
    'ConvCA': 'convca',
    'SSVEPNet': 'ssvepnet',
    'DualBranchNet': 'dualbranchnet',
    'SSVEPformer': 'ssvepformer',
    'DDGCNN': 'ddgcnn',
    'EnhSSVEPformer': 'enh_ssvepformer',
}
KNOWLEDGE_METHOD_KEYS = {
    'TRCA': 'trca',
    'TDCA': 'tdca',
    'TD-RCA': 'td_rca',
}
AUGMENTATION_METHOD_KEYS = {
    'Scale': 'scale',
    'Noise': 'noise_aug',
    'Flip': 'flip',
    'FShift': 'fshift',
    'DWTaug': 'dwtaug',
    'HHTaug': 'hhtaug',
    'Diffusion': 'diffusion',
    'DiffusionGAN': 'diffusion_gan',
    'CT_DCENet': 'ct_dcenet',
    'EEGDNet': 'eegdnet',
    'GCTNet': 'gctnet',
    'SimpleCNN': 'simple_cnn',
    'FCNN': 'fcnn',
    'ResCNN': 'res_cnn',
    'GeneratorCNN': 'generator_cnn',
    'GeneratorTransformer': 'generator_transformer',
    'NovelCNN': 'novel_cnn',
    'DuoCL': 'duocl',
}
STATEFUL_BACKBONES = ('CNN', 'CNN-LSTM', 'CNN-Trans')
STATEFUL_BACKBONE_KEYS = {
    'CNN': 'cnn',
    'CNN-LSTM': 'cnn-lstm',
    'CNN-Trans': 'cnn-trans',
}
STATEFUL_MODES = ('BASE', 'GAN', 'VAE', 'GAN-VAE')
STATEFUL_MODE_KEYS = {
    'BASE': 'base',
    'GAN': 'gan',
    'VAE': 'vae',
    'GAN-VAE': 'gan-vae',
}


def _dict_or_empty(value):
    return value if isinstance(value, dict) else {}


def _list_or_empty(value):
    return value if isinstance(value, list) else []


def _merge_dicts(*parts: Dict) -> Dict:
    merged: Dict = {}
    for part in parts:
        merged.update(_dict_or_empty(part))
    return merged


def _unique(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def get_paths_config(config: Dict) -> Dict:
    paths_cfg = _dict_or_empty(config.get('paths'))
    config['paths'] = paths_cfg
    return paths_cfg


def get_experiment_config(config: Dict) -> Dict:
    experiment_cfg = _dict_or_empty(config.get('experiment'))
    config['experiment'] = experiment_cfg
    return experiment_cfg


def get_dataset_settings(config: Dict) -> Dict:
    dataset_cfg = _dict_or_empty(config.get('dataset'))
    config['dataset'] = dataset_cfg
    return dataset_cfg


def get_dataset_catalog(config: Dict) -> Dict:
    datasets_cfg = _dict_or_empty(config.get('datasets'))
    config['datasets'] = datasets_cfg
    return datasets_cfg


def get_signal_config(config: Dict) -> Dict:
    signal_cfg = _dict_or_empty(config.get('signal'))
    config['signal'] = signal_cfg
    return signal_cfg



def get_loso_transfer_config(config: Dict) -> Dict:
    dataset_cfg = get_dataset_settings(config)
    transfer_cfg = _dict_or_empty(dataset_cfg.get('loso_calibration'))

    transform = str(transfer_cfg.get('transform', 'sd-lst')).strip().lower()
    if transform in {'sdlst', 'sd_lst'}:
        transform = 'sd-lst'
    elif transform not in {'none', 'lst', 'sd-lst'}:
        raise ValueError(f'Unsupported loso_calibration.transform={transform!r}.')

    selection = transfer_cfg.get('class_selection', transfer_cfg.get('frequency_selection', 'uniform'))
    selection = str(selection).strip().lower() or 'uniform'
    if selection in {'classes', 'frequencies'}:
        selection = 'uniform'
    if selection not in {'uniform', 'all', 'explicit'}:
        raise ValueError(f'Unsupported loso_calibration.class_selection={selection!r}.')

    nall_raw = transfer_cfg.get('nall', 'auto')
    nall = None if str(nall_raw).strip().lower() == 'auto' else int(nall_raw)
    k_raw = transfer_cfg.get('k_classes', transfer_cfg.get('k_frequencies', 'auto'))
    k_classes = None if str(k_raw).strip().lower() == 'auto' else int(k_raw)
    selected_raw = transfer_cfg.get('selected_classes', transfer_cfg.get('selected_frequencies', []))

    return {
        'enabled': bool(transfer_cfg.get('enabled', False)),
        'transform': transform,
        'ncali': int(transfer_cfg.get('ncali', 1)),
        'nall': nall,
        'k_classes': k_classes,
        'class_selection': selection,
        'selected_classes': [int(value) for value in _list_or_empty(selected_raw)],
        'holdout_all_classes': bool(transfer_cfg.get('holdout_all_classes', True)),
        'trial_selection': str(transfer_cfg.get('trial_selection', 'first')).strip().lower() or 'first',
        'trial_offset': int(transfer_cfg.get('trial_offset', 0)),
        'trial_indices': [int(value) for value in _list_or_empty(transfer_cfg.get('trial_indices'))],
        'use_second_stage': bool(transfer_cfg.get('use_second_stage', True)),
    }


def get_training_config(config: Dict) -> Dict:
    training_cfg = _dict_or_empty(config.get('training'))
    config['training'] = training_cfg
    return training_cfg


def get_augmentation_config(config: Dict) -> Dict:
    augmentation_cfg = _dict_or_empty(config.get('augmentation'))
    config['augmentation'] = augmentation_cfg
    return augmentation_cfg


def get_classification_models_config(config: Dict) -> Dict:
    models_cfg = _dict_or_empty(config.get('classification_models'))
    config['classification_models'] = models_cfg
    return models_cfg


def get_knowledge_methods_config(config: Dict) -> Dict:
    methods_cfg = _dict_or_empty(config.get('knowledge_methods'))
    config['knowledge_methods'] = methods_cfg
    return methods_cfg


def get_diffusion_classifier_config(config: Dict) -> Dict:
    classifier_cfg = _dict_or_empty(config.get('diffusion_classifier'))
    config['diffusion_classifier'] = classifier_cfg
    return classifier_cfg


def resolve_dataset_name(config: Dict) -> str:
    dataset_cfg = get_dataset_settings(config)
    requested = str(dataset_cfg.get('active', 'benchmark12')).strip().lower()
    if requested in {'1', 'dataset1'}:
        requested = 'benchmark12'
    elif requested in {'2', 'dataset2'}:
        requested = 'benchmark40'
    elif requested in {'3', 'dataset3'}:
        requested = 'beta'
    if requested not in DATASET_NAME_TO_ID:
        raise ValueError(f'Unsupported dataset: {requested}')
    return requested


def resolve_dataset_id(config: Dict) -> int:
    return DATASET_NAME_TO_ID[resolve_dataset_name(config)]


def get_active_dataset_config(config: Dict) -> Tuple[str, Dict]:
    dataset_name = resolve_dataset_name(config)
    datasets_cfg = get_dataset_catalog(config)
    dataset_spec = _dict_or_empty(datasets_cfg.get(dataset_name))
    if not dataset_spec:
        raise KeyError(f'Missing datasets.{dataset_name} in config.yaml')
    return dataset_name, dataset_spec


def get_dataset_root(config: Dict, dataset_name: str = None) -> str:
    if dataset_name is None:
        dataset_name = resolve_dataset_name(config)
    datasets_paths = _dict_or_empty(get_paths_config(config).get('datasets'))
    dataset_root = datasets_paths.get(dataset_name)
    if not dataset_root:
        raise KeyError(f'Missing paths.datasets.{dataset_name} in config.yaml')
    return dataset_root


def get_output_paths(config: Dict) -> Dict:
    runtime_cfg = _dict_or_empty(config.get('_runtime'))
    runtime_outputs = _dict_or_empty(runtime_cfg.get('output_paths'))
    if runtime_outputs:
        return runtime_outputs
    outputs_cfg = _dict_or_empty(get_paths_config(config).get('outputs'))
    return outputs_cfg


def get_noise_config(config: Dict) -> Dict:
    noise_cfg = _dict_or_empty(get_augmentation_config(config).get('noise'))
    _, dataset_spec = get_active_dataset_config(config)
    fs_signal = dataset_spec.get('sample_rate',250)
    
    return {
        'fs_signal': fs_signal,
        'noise_family': noise_cfg.get('family', 'gaussian'),   # 'gaussian' or 'real'
        'noise_type': noise_cfg.get('type', 'time'),           # gaussian: time/space/freq/random/all
        'mode': noise_cfg.get('mode', 'time'),                 # real: time/space
        'noise_ratio': noise_cfg.get('ratio', 0.25),
        'snr_db': noise_cfg.get('snr_db', 5),
        'noise_src_num': noise_cfg.get('src_num', 1),          # real only
        'artifact_type': noise_cfg.get('artifact_type', 'both'),  # real: emg/eog/both
        'channel_gain_sigma': noise_cfg.get('channel_gain_sigma', 0.3),  # real: log-normal 通道增益 σ
        'fs_signal': fs_signal,
        'fs_noise_emg': noise_cfg.get('fs_noise_emg', 512),
        'fs_noise_eog': noise_cfg.get('fs_noise_eog', 256), 
    }


def get_default_batch_size(config: Dict) -> int:
    return int(get_training_config(config).get('batch_size', 36))


def get_default_epochs(config: Dict) -> int:
    return int(get_training_config(config).get('epochs', 300))


def use_marginal_smoothing(config: Dict) -> bool:
    return bool(get_signal_config(config).get('use_marginal_smoothing', True))


def _normalize_stateful_method_name(method: str) -> str:
    candidate = str(method).strip()
    candidate = candidate.replace('CNN-Trans', 'CNN-Trans')
    candidate = candidate.replace('CNN-TRANS', 'CNN-Trans')
    candidate = candidate.replace('CNN-TRANS', 'CNN-Trans')
    candidate = candidate.replace('cnn-trans', 'CNN-Trans')
    candidate = candidate.replace('CNN-LSTM', 'CNN-LSTM')
    candidate = candidate.replace('cnn-lstm', 'CNN-LSTM')
    candidate = candidate.replace('GAN_VAE', 'GAN-VAE')
    candidate = candidate.replace('gan_vae', 'GAN-VAE')

    if candidate in {'CNN', 'CNN-LSTM', 'CNN-Trans'}:
        return candidate + '_BASE'

    for backbone in STATEFUL_BACKBONES:
        prefix = f'{backbone}_'
        if candidate.startswith(prefix):
            mode = candidate[len(prefix):].upper().replace('_', '-')
            if mode in STATEFUL_MODES:
                return f'{backbone}_{mode}'
    return candidate


def _parse_stateful_method(method: str):
    normalized = _normalize_stateful_method_name(method)
    for backbone in ('CNN-LSTM', 'CNN-Trans', 'CNN'):
        prefix = f'{backbone}_'
        if normalized.startswith(prefix):
            mode = normalized[len(prefix):]
            if mode in STATEFUL_MODES:
                return normalized, backbone, mode
    return normalized, None, None


def get_augmentation_method_config(config: Dict, method: str) -> Dict:
    augmentation_cfg = get_augmentation_config(config)
    stateful_cfg = _dict_or_empty(augmentation_cfg.get('stateful'))
    methods_cfg = _dict_or_empty(augmentation_cfg.get('methods'))

    normalized, backbone, mode = _parse_stateful_method(method)
    if backbone and mode:
        backbones_cfg = _dict_or_empty(stateful_cfg.get('backbones'))
        modes_cfg = _dict_or_empty(stateful_cfg.get('modes'))
        overrides_cfg = _dict_or_empty(stateful_cfg.get('overrides'))
        override_key = normalized.lower()
        legacy_override_key = (
            override_key
            .replace('cnn-trans', 'cnn-trans')
            .replace('cnn-lstm', 'cnn-lstm')
            .replace('gan-vae', 'gan_vae')
        )
        return _merge_dicts(
            stateful_cfg.get('defaults'),
            backbones_cfg.get(STATEFUL_BACKBONE_KEYS[backbone]),
            modes_cfg.get(STATEFUL_MODE_KEYS[mode]),
            overrides_cfg.get(legacy_override_key),
            overrides_cfg.get(override_key),
        )

    key = AUGMENTATION_METHOD_KEYS.get(method, AUGMENTATION_METHOD_KEYS.get(normalized, method.lower()))
    return _dict_or_empty(methods_cfg.get(key))


# Fallback so that registering a new decoder does not also require editing every config
# file before it can run. Anything present in the YAML still wins.
DEFAULT_CLASSIFIER_CONFIG = {
    'epochs': 200,
    'batch_size': 36,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'use_scheduler': True,
}


def get_classifier_model_config(config: Dict, method: str) -> Dict:
    models_cfg = get_classification_models_config(config)
    key = CLASSIFICATION_MODEL_KEYS.get(method, method.lower())
    merged = dict(DEFAULT_CLASSIFIER_CONFIG)
    merged.update(_dict_or_empty(models_cfg.get(key)))
    return merged


def get_knowledge_method_config(config: Dict, method: str) -> Dict:
    methods_cfg = get_knowledge_methods_config(config)
    key = KNOWLEDGE_METHOD_KEYS.get(method, method.lower())
    return _dict_or_empty(methods_cfg.get(key))


def get_evaluation_pairs(config: Dict) -> List[Dict[str, str]]:
    experiment_cfg = get_experiment_config(config)
    pairs = _list_or_empty(experiment_cfg.get('evaluation_pairs')) or [{'train_view': 'clean', 'test_view': 'clean'}]
    normalized = []
    for pair in pairs:
        pair = _dict_or_empty(pair)
        normalized.append(
            {
                'train_view': pair.get('train_view', 'clean'),
                'test_view': pair.get('test_view', 'clean'),
            }
        )
    return normalized


def get_method_view_override(config: Dict, family: str, default_train_view: str, default_test_view: str) -> Tuple[str, str]:
    experiment_cfg = get_experiment_config(config)
    overrides_cfg = _dict_or_empty(experiment_cfg.get('method_view_overrides'))
    family_cfg = _dict_or_empty(overrides_cfg.get(family))

    train_view = str(family_cfg.get('train_view', 'inherit')).strip() or 'inherit'
    test_view = str(family_cfg.get('test_view', 'inherit')).strip() or 'inherit'

    if train_view == 'inherit':
        train_view = default_train_view
    if test_view == 'inherit':
        test_view = default_test_view
    return train_view, test_view


def get_downstream_methods(config: Dict) -> List[str]:
    experiment_cfg = get_experiment_config(config)
    methods_cfg = _dict_or_empty(experiment_cfg.get('methods'))
    hybrid_cfg = _dict_or_empty(methods_cfg.get('hybrid'))

    methods: List[str] = []
    methods.extend(_list_or_empty(methods_cfg.get('traditional')))
    methods.extend(_list_or_empty(methods_cfg.get('deep')))
    for knowledge_method, deep_method in itertools.product(
        _list_or_empty(hybrid_cfg.get('knowledge')),
        _list_or_empty(hybrid_cfg.get('deep')),
    ):
        methods.append(f'{knowledge_method}+{deep_method}')
    methods.extend(_list_or_empty(methods_cfg.get('diffusion')))
    methods.extend(_list_or_empty(methods_cfg.get('extra')))
    return _unique(methods)


def get_augmentation_methods(config: Dict) -> List[str]:
    experiment_cfg = get_experiment_config(config)
    methods_cfg = _dict_or_empty(experiment_cfg.get('methods'))
    return _unique(_list_or_empty(methods_cfg.get('augmentation')))


def resolve_device(config: Dict) -> str:
    experiment_cfg = get_experiment_config(config)
    requested = experiment_cfg.get('device', 'auto')
    if requested == 'auto':
        cuda_index = get_training_config(config).get('cuda_index', 0)
        if cuda_index != 'cpu' and torch.cuda.is_available():
            return f'cuda:{cuda_index}'
        return 'cpu'
    if requested == 'cpu':
        return 'cpu'
    return requested


def resolve_subject_ids(config: Dict) -> List[int]:
    experiment_cfg = get_experiment_config(config)
    _, dataset_spec = get_active_dataset_config(config)
    total_subjects = int(dataset_spec['num_subjects'])
    split_mode = experiment_cfg.get('split_mode', 'intra')
    key = 'target_subject_ids' if split_mode == 'loso' else 'subject_ids'
    values = _list_or_empty(experiment_cfg.get(key)) or list(range(1, total_subjects + 1))
    return [int(value) for value in values]


def resolve_fold_ids(config: Dict) -> List[int]:
    experiment_cfg = get_experiment_config(config)
    configured = _list_or_empty(experiment_cfg.get('fold_ids'))
    if configured:
        return [int(value) for value in configured]

    dataset_name = resolve_dataset_name(config)
    dataset_cfg = get_dataset_settings(config)
    split_mode = experiment_cfg.get('split_mode', 'intra')

    if split_mode == 'intra':
        if dataset_name == 'benchmark12':
            return list(range(5))
        if dataset_name == 'benchmark40':
            return list(range(6))
        if dataset_name == 'beta':
            return list(range(4))
        return list(range(int(dataset_cfg.get('fallback_num_folds', 4))))

    if split_mode == 'loso':
        transfer_cfg = get_loso_transfer_config(config)
        if transfer_cfg.get('enabled') and int(transfer_cfg.get('ncali', 1)) == 1:
            if dataset_name == 'benchmark12':
                return list(range(5))
            if dataset_name == 'benchmark40':
                return list(range(6))
            if dataset_name == 'beta':
                return list(range(4))
        return [0]

    return [0]




