from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class TransferSplit:
    train_raw: np.ndarray
    train_labels: np.ndarray
    test_raw: np.ndarray
    test_labels: np.ndarray
    calibration_raw: np.ndarray
    calibration_labels: np.ndarray
    adaptation_raw: np.ndarray
    adaptation_labels: np.ndarray
    metadata: Dict[str, object]


def _default_nall(num_classes: int) -> int:
    if num_classes >= 40:
        return 10
    return max(1, min(6, num_classes))


def _select_uniform_classes(num_classes: int, k_classes: int) -> List[int]:
    if k_classes <= 0:
        raise ValueError('k_classes must be positive.')
    if k_classes >= num_classes:
        return list(range(num_classes))

    selected = []
    for rank in range(1, k_classes + 1):
        class_index = int(np.floor(num_classes * (2 * rank - 1) / (2 * k_classes)))
        class_index = min(max(class_index, 0), num_classes - 1)
        if class_index not in selected:
            selected.append(class_index)

    if len(selected) < k_classes:
        for class_index in range(num_classes):
            if class_index not in selected:
                selected.append(class_index)
            if len(selected) == k_classes:
                break
    return sorted(selected)


def _resolve_selected_classes(num_classes: int, transfer_cfg: Dict[str, object]) -> Tuple[List[int], int, int]:
    selection = str(transfer_cfg.get('class_selection', 'uniform')).strip().lower()
    ncali = int(transfer_cfg['ncali'])
    requested_nall = transfer_cfg.get('nall')
    configured_k = transfer_cfg.get('k_classes')

    if selection == 'all':
        selected_classes = list(range(num_classes))
    elif selection == 'explicit':
        selected_classes = sorted({int(class_id) for class_id in transfer_cfg.get('selected_classes', [])})
        if not selected_classes:
            raise ValueError('loso_calibration.selected_classes must be provided when class_selection=explicit.')
        if selected_classes[0] < 0 or selected_classes[-1] >= num_classes:
            raise ValueError(f'loso_calibration.selected_classes must be in [0, {num_classes - 1}].')
    elif selection == 'uniform':
        if configured_k is not None:
            k_classes = int(configured_k)
        elif requested_nall is not None:
            requested_nall = int(requested_nall)
            if requested_nall <= 0:
                raise ValueError('loso_calibration.nall must be positive when provided.')
            if requested_nall % ncali != 0:
                raise ValueError('loso_calibration.nall must satisfy nall = k_classes * ncali.')
            k_classes = requested_nall // ncali
        else:
            k_classes = max(1, _default_nall(num_classes) // ncali)
        k_classes = min(max(int(k_classes), 1), num_classes)
        selected_classes = _select_uniform_classes(num_classes, k_classes)
    else:
        raise ValueError(f'Unsupported loso_calibration.class_selection={selection!r}.')

    resolved_nall = len(selected_classes) * ncali
    return selected_classes, resolved_nall, _default_nall(num_classes)


def _build_class_index_map(labels: np.ndarray, num_classes: int) -> List[np.ndarray]:
    class_indices: List[np.ndarray] = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        if indices.size == 0:
            raise ValueError(f'Class {class_id} has no trials in the provided labels.')
        class_indices.append(indices)
    return class_indices



def _resolve_fold_based_trial_offsets(trials_per_class: int, transfer_cfg: Dict[str, object]) -> np.ndarray | None:
    if int(transfer_cfg.get('ncali', 1)) != 1:
        return None

    dataset_name = str(transfer_cfg.get('dataset_name', '')).strip().lower()
    fold_id = transfer_cfg.get('fold_id')
    if fold_id is None:
        return None
    fold_id = int(fold_id)

    if dataset_name == 'benchmark12':
        if fold_id not in range(5):
            raise ValueError(f'benchmark12 low-calibration LOSO expects fold_id in [0, 4], got {fold_id}.')
        start = fold_id * 3
        candidates = np.arange(start, min(start + 3, trials_per_class), dtype=np.int64)
        if candidates.size == 0:
            raise ValueError(f'benchmark12 fold {fold_id} produced no candidate trials.')
        rng = np.random.default_rng(int(transfer_cfg.get('selection_seed', 0)))
        choice = int(rng.choice(candidates))
        return np.array([choice], dtype=np.int64)

    if dataset_name == 'benchmark40':
        if fold_id not in range(6):
            raise ValueError(f'benchmark40 low-calibration LOSO expects fold_id in [0, 5], got {fold_id}.')
        if fold_id >= trials_per_class:
            raise ValueError(f'benchmark40 fold {fold_id} exceeds trials_per_class={trials_per_class}.')
        return np.array([fold_id], dtype=np.int64)

    if dataset_name == 'beta':
        if fold_id not in range(4):
            raise ValueError(f'beta low-calibration LOSO expects fold_id in [0, 3], got {fold_id}.')
        if fold_id >= trials_per_class:
            raise ValueError(f'beta fold {fold_id} exceeds trials_per_class={trials_per_class}.')
        return np.array([fold_id], dtype=np.int64)

    return None


def _resolve_trial_offsets(trials_per_class: int, transfer_cfg: Dict[str, object]) -> np.ndarray:
    ncali = int(transfer_cfg['ncali'])
    mode = str(transfer_cfg.get('trial_selection', 'first')).strip().lower()

    fold_offsets = _resolve_fold_based_trial_offsets(trials_per_class, transfer_cfg)
    if fold_offsets is not None:
        return fold_offsets

    if ncali <= 0:
        raise ValueError('loso_calibration.ncali must be positive.')
    if ncali >= trials_per_class:
        raise ValueError(
            f'loso_calibration.ncali={ncali} leaves no test trials; available trials_per_class={trials_per_class}.'
        )

    if mode == 'first':
        offsets = np.arange(ncali, dtype=np.int64)
    elif mode == 'last':
        offsets = np.arange(trials_per_class - ncali, trials_per_class, dtype=np.int64)
    elif mode == 'offset':
        start = int(transfer_cfg.get('trial_offset', 0))
        offsets = np.arange(start, start + ncali, dtype=np.int64)
    elif mode == 'explicit':
        offsets = np.asarray(transfer_cfg.get('trial_indices', []), dtype=np.int64)
        if offsets.size != ncali:
            raise ValueError(
                f'loso_calibration.trial_indices must contain exactly ncali={ncali} offsets when trial_selection=explicit.'
            )
    else:
        raise ValueError(f'Unsupported loso_calibration.trial_selection={mode!r}.')

    if np.any(offsets < 0) or np.any(offsets >= trials_per_class):
        raise ValueError(
            f'loso_calibration trial offsets {offsets.tolist()} are outside [0, {trials_per_class - 1}].'
        )
    return np.sort(offsets.astype(np.int64))


def _compute_class_templates(eeg_raw: np.ndarray, labels: np.ndarray, class_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for class_id in class_ids:
        class_trials = eeg_raw[labels == class_id]
        if class_trials.size == 0:
            raise ValueError(f'No trials available to build a template for class {class_id}.')
        templates[int(class_id)] = class_trials.mean(axis=0).astype(np.float64)
    return templates


def _stack_templates(templates: Dict[int, np.ndarray], class_ids: Sequence[int]) -> np.ndarray:
    return np.concatenate([templates[int(class_id)] for class_id in class_ids], axis=-1)


def _solve_linear_transform(source_template: np.ndarray, target_template: np.ndarray) -> np.ndarray:
    source_template = np.asarray(source_template, dtype=np.float64)
    target_template = np.asarray(target_template, dtype=np.float64)
    return (target_template @ np.linalg.pinv(source_template)).astype(np.float32)


def _apply_linear_transform(transform: np.ndarray, eeg_raw: np.ndarray) -> np.ndarray:
    transformed = np.einsum('ij,njt->nit', transform.astype(np.float32), eeg_raw.astype(np.float32), optimize=True)
    return transformed.astype(np.float32)


def _partition_target_subject(
    target_raw: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int,
    selected_classes: Sequence[int],
    transfer_cfg: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if not bool(transfer_cfg.get('holdout_all_classes', True)):
        raise ValueError(
            'Paper-faithful LST/sd-LST requires holding out the same calibration block(s) for every stimulus frequency.'
        )

    class_index_map = _build_class_index_map(target_labels, num_classes)
    trials_per_class = min(len(indices) for indices in class_index_map)
    trial_offsets = _resolve_trial_offsets(trials_per_class, transfer_cfg)

    calibration_mask = np.zeros(target_labels.shape[0], dtype=bool)
    adaptation_mask = np.zeros(target_labels.shape[0], dtype=bool)

    for class_id in range(num_classes):
        chosen = class_index_map[class_id][trial_offsets]
        calibration_mask[chosen] = True
    for class_id in selected_classes:
        chosen = class_index_map[int(class_id)][trial_offsets]
        adaptation_mask[chosen] = True

    calibration_raw = target_raw[calibration_mask].astype(np.float32)
    calibration_labels = target_labels[calibration_mask].astype(np.int64)
    adaptation_raw = target_raw[adaptation_mask].astype(np.float32)
    adaptation_labels = target_labels[adaptation_mask].astype(np.int64)
    test_raw = target_raw[~calibration_mask].astype(np.float32)
    test_labels = target_labels[~calibration_mask].astype(np.int64)

    metadata = {
        'trial_offsets': trial_offsets.astype(int).tolist(),
        'fold_id': int(transfer_cfg.get('fold_id', 0)),
        'dataset_name': str(transfer_cfg.get('dataset_name', '')),
        'holdout_classes': list(range(num_classes)),
        'selected_classes': [int(class_id) for class_id in selected_classes],
        'unused_calibration_trials': int(calibration_raw.shape[0] - adaptation_raw.shape[0]),
    }
    return calibration_raw, calibration_labels, adaptation_raw, adaptation_labels, test_raw, test_labels, metadata


def _apply_original_lst_to_subject(
    source_raw: np.ndarray,
    source_labels: np.ndarray,
    target_templates: Dict[int, np.ndarray],
    all_classes: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    transformed = source_raw.astype(np.float32).copy()
    source_templates = _compute_class_templates(source_raw, source_labels, all_classes)
    transforms: Dict[int, np.ndarray] = {}

    for class_id in all_classes:
        transform = _solve_linear_transform(source_templates[int(class_id)], target_templates[int(class_id)])
        class_mask = source_labels == int(class_id)
        transformed[class_mask] = _apply_linear_transform(transform, source_raw[class_mask])
        transforms[int(class_id)] = transform
    return transformed, transforms


def _apply_sd_lst_stage_one(
    source_raw: np.ndarray,
    source_labels: np.ndarray,
    target_stack: np.ndarray,
    selected_classes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    source_templates = _compute_class_templates(source_raw, source_labels, selected_classes)
    source_stack = _stack_templates(source_templates, selected_classes)
    transform = _solve_linear_transform(source_stack, target_stack)
    return _apply_linear_transform(transform, source_raw), transform


def _build_sd_lst_stage_two_transform(
    stage_one_subjects: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    target_stack: np.ndarray,
    selected_classes: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    combined_raw = np.concatenate([raw for _, raw, _ in stage_one_subjects], axis=0).astype(np.float32)
    combined_labels = np.concatenate([labels for _, _, labels in stage_one_subjects], axis=0).astype(np.int64)
    combined_templates = _compute_class_templates(combined_raw, combined_labels, selected_classes)
    combined_stack = _stack_templates(combined_templates, selected_classes)
    transform = _solve_linear_transform(combined_stack, target_stack)
    return transform, combined_stack


def build_loso_transfer_split(
    source_subjects: Sequence[Tuple[int, np.ndarray, np.ndarray]],
    target_raw: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int,
    transfer_cfg: Dict[str, object],
) -> TransferSplit:
    if not source_subjects:
        raise ValueError('LOSO transfer requires at least one source subject.')

    selected_classes, resolved_nall, default_nall = _resolve_selected_classes(num_classes, transfer_cfg)
    calibration_raw, calibration_labels, adaptation_raw, adaptation_labels, test_raw, test_labels, partition_meta = _partition_target_subject(
        target_raw,
        target_labels,
        num_classes,
        selected_classes,
        transfer_cfg,
    )

    transform_mode = str(transfer_cfg.get('transform', 'sd-lst')).strip().lower()
    if transform_mode == 'lst' and len(selected_classes) != num_classes:
        raise ValueError(
            'Paper-faithful LST requires target calibration data for every stimulus frequency. '
            'Set frequency_selection: all (or select all classes explicitly).'
        )

    target_templates = _compute_class_templates(adaptation_raw, adaptation_labels, selected_classes)
    target_stack = _stack_templates(target_templates, selected_classes)

    transformed_subjects = []
    stage_one_matrices: Dict[int, np.ndarray] = {}
    per_class_matrices: Dict[int, Dict[int, np.ndarray]] = {}
    global_matrix = None
    stage_two_source_stack = None

    if transform_mode == 'none':
        transformed_subjects = [
            (int(subject_id), np.asarray(raw, dtype=np.float32).copy(), np.asarray(labels, dtype=np.int64).copy())
            for subject_id, raw, labels in source_subjects
        ]
        train_raw = np.concatenate([raw for _, raw, _ in transformed_subjects], axis=0).astype(np.float32)
        train_labels = np.concatenate([labels for _, _, labels in transformed_subjects], axis=0).astype(np.int64)
    elif transform_mode == 'lst':
        all_classes = list(range(num_classes))
        for subject_id, raw, labels in source_subjects:
            transformed_raw, transforms = _apply_original_lst_to_subject(
                np.asarray(raw, dtype=np.float32),
                np.asarray(labels, dtype=np.int64),
                target_templates,
                all_classes,
            )
            transformed_subjects.append((int(subject_id), transformed_raw, np.asarray(labels, dtype=np.int64).copy()))
            per_class_matrices[int(subject_id)] = transforms

        source_train_raw = np.concatenate([raw for _, raw, _ in transformed_subjects], axis=0).astype(np.float32)
        source_train_labels = np.concatenate([labels for _, _, labels in transformed_subjects], axis=0).astype(np.int64)
        train_raw = np.concatenate([source_train_raw, calibration_raw.astype(np.float32)], axis=0)
        train_labels = np.concatenate([source_train_labels, calibration_labels.astype(np.int64)], axis=0)
    elif transform_mode == 'sd-lst':
        stage_one_subjects = []
        for subject_id, raw, labels in source_subjects:
            transformed_raw, transform = _apply_sd_lst_stage_one(
                np.asarray(raw, dtype=np.float32),
                np.asarray(labels, dtype=np.int64),
                target_stack,
                selected_classes,
            )
            stage_one_subjects.append((int(subject_id), transformed_raw, np.asarray(labels, dtype=np.int64).copy()))
            stage_one_matrices[int(subject_id)] = transform

        if bool(transfer_cfg.get('use_second_stage', True)):
            global_matrix, stage_two_source_stack = _build_sd_lst_stage_two_transform(
                stage_one_subjects,
                target_stack,
                selected_classes,
            )
            for subject_id, raw, labels in stage_one_subjects:
                transformed_subjects.append((subject_id, _apply_linear_transform(global_matrix, raw), labels))
        else:
            transformed_subjects = stage_one_subjects

        train_raw = np.concatenate([raw for _, raw, _ in transformed_subjects], axis=0).astype(np.float32)
        train_labels = np.concatenate([labels for _, _, labels in transformed_subjects], axis=0).astype(np.int64)
    else:
        raise ValueError(f'Unsupported loso_calibration.transform={transform_mode!r}.')

    metadata: Dict[str, object] = {
        'enabled': True,
        'transform': transform_mode,
        'ncali': int(transfer_cfg['ncali']),
        'resolved_nall': int(resolved_nall),
        'default_nall': int(default_nall),
        'selected_classes': [int(class_id) for class_id in selected_classes],
        'k_classes': len(selected_classes),
        'holdout_all_classes': True,
        'trial_selection': str(transfer_cfg.get('trial_selection', 'first')),
        'use_second_stage': bool(transfer_cfg.get('use_second_stage', True)),
        'partition': partition_meta,
        'fold_id': int(transfer_cfg.get('fold_id', 0)),
        'paper_faithful': True,
        'train_includes_target_calibration': bool(transform_mode == 'lst'),
    }
    if stage_one_matrices:
        metadata['stage_one_matrices'] = stage_one_matrices
    if per_class_matrices:
        metadata['per_class_matrices'] = per_class_matrices
    if global_matrix is not None:
        metadata['global_matrix'] = global_matrix
    if stage_two_source_stack is not None:
        metadata['stage_two_source_stack'] = stage_two_source_stack

    return TransferSplit(
        train_raw=train_raw,
        train_labels=train_labels.astype(np.int64),
        test_raw=test_raw.astype(np.float32),
        test_labels=test_labels.astype(np.int64),
        calibration_raw=calibration_raw,
        calibration_labels=calibration_labels,
        adaptation_raw=adaptation_raw,
        adaptation_labels=adaptation_labels,
        metadata=metadata,
    )
