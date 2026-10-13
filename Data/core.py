import itertools
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import loadmat

from Data.signal_ops import add_noise
from Data.transfer import build_loso_transfer_split
from Support.config import (
    get_active_dataset_config,
    get_dataset_root,
    get_dataset_settings,
    get_experiment_config,
    get_noise_config,
    get_signal_config,
    get_loso_transfer_config,
    resolve_dataset_id,
    resolve_dataset_name,
)


@dataclass
class DatasetMeta:
    dataset_id: int
    dataset_name: str
    sample_rate: int
    num_channels: int
    num_classes: int
    num_subjects: int
    num_trials: int
    num_filter_banks: int
    window_sec: float
    num_timesteps: int
    start_frequency: float
    frequency_step: float
    freqs: np.ndarray
    split_mode: str

    @property
    def Fs(self) -> int:
        return self.sample_rate

    @property
    def Nc(self) -> int:
        return self.num_channels

    @property
    def Nf(self) -> int:
        return self.num_classes

    @property
    def Ns(self) -> int:
        return self.num_subjects

    @property
    def Nh(self) -> int:
        return self.num_trials

    @property
    def Nm(self) -> int:
        return self.num_filter_banks

    @property
    def ws(self) -> float:
        return self.window_sec

    @property
    def Nt(self) -> int:
        return self.num_timesteps

    @property
    def initfreq(self) -> float:
        return self.start_frequency

    @property
    def deltafreq(self) -> float:
        return self.frequency_step


@dataclass
class SignalSet:
    train: np.ndarray
    test: np.ndarray


@dataclass
class SplitArrays:
    train_raw: np.ndarray
    test_raw: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    meta: DatasetMeta
    subject: Optional[int] = None
    fold: Optional[int] = None
    target_subject: Optional[int] = None
    calibration_raw: Optional[np.ndarray] = None
    calibration_labels: Optional[np.ndarray] = None
    adaptation_raw: Optional[np.ndarray] = None
    adaptation_labels: Optional[np.ndarray] = None
    transfer_metadata: Optional[Dict[str, object]] = None


@dataclass
class ExperimentSignals:
    meta: DatasetMeta
    clean: SignalSet
    source: SignalSet
    generated: Optional[SignalSet]
    train_labels: np.ndarray
    test_labels: np.ndarray
    source_name: str
    augmenter_name: Optional[str] = None
    # Ground-truth corruption masks from the noise injector, at raw (N, C, T)
    # resolution. add_noise() has always produced these; they used to be dropped.
    masks: Optional[SignalSet] = None


@dataclass
class AugmentationOutput:
    method: str
    generated: SignalSet
    metrics: Dict[str, float]


DEFAULT_BENCHMARK40_CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]
DEFAULT_BETA_CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]


def build_dataset_meta(config, split_mode: str) -> DatasetMeta:
    dataset_name = resolve_dataset_name(config)
    dataset_id = resolve_dataset_id(config)
    _, dataset_spec = get_active_dataset_config(config)
    dataset_cfg = get_dataset_settings(config)
    signal_cfg = get_signal_config(config)

    window_sec = float(dataset_cfg.get('window_sec', 1.0))
    sample_rate = int(dataset_spec['sample_rate'])
    start_frequency = float(dataset_spec['start_frequency'])
    frequency_step = float(dataset_spec['frequency_step'])
    num_classes = int(dataset_spec['num_classes'])
    freqs = start_frequency + np.arange(num_classes) * frequency_step

    selected_channels = dataset_spec.get('selected_channels')
    num_channels = len(selected_channels) if selected_channels else int(dataset_spec['num_channels'])

    return DatasetMeta(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        sample_rate=sample_rate,
        num_channels=num_channels,
        num_classes=num_classes,
        num_subjects=int(dataset_spec['num_subjects']),
        num_trials=int(dataset_spec['num_trials']),
        num_filter_banks=int(signal_cfg.get('num_filter_banks', 1)),
        window_sec=window_sec,
        num_timesteps=int(sample_rate * window_sec),
        start_frequency=start_frequency,
        frequency_step=frequency_step,
        freqs=freqs.astype(np.float32),
        split_mode=split_mode,
    )


def _load_benchmark12_subject(subject_id: int, dataset_root: str, window_sec: float, decay_sec: float = 0.14) -> Tuple[np.ndarray, np.ndarray]:
    subject_file = loadmat(f'{dataset_root}/S{subject_id}.mat')
    samples = subject_file['eeg']
    eeg_data = samples[0, :, :, :]
    labels = np.zeros((180, 1), dtype=np.int64)
    for class_index in range(1,12):
        eeg_data = np.concatenate([eeg_data, samples[class_index, :, :, :]], axis=2)
        labels[class_index * 15:(class_index + 1) * 15] = class_index
    eeg_data = eeg_data.transpose([2, 0, 1])
    eeg_data = eeg_data[:, :, int(decay_sec * 256):int((decay_sec + window_sec) * 256)]
    return eeg_data.astype(np.float32), labels.reshape(-1)


def _load_benchmark40_subject(
    subject_id: int,
    dataset_root: str,
    window_sec: float,
    num_classes: int,
    num_trials: int,
    channel_indices: List[int],
    decay_sec: float = 0.14,
) -> Tuple[np.ndarray, np.ndarray]:
    data = loadmat(f'{dataset_root}/S{subject_id}.mat')
    eeg = data['data'][
        channel_indices,
        int((0.5 + decay_sec) * 250):int((0.5 + decay_sec + window_sec) * 250),
        :,
        :,
    ]
    eeg = eeg.transpose(2, 0, 1, 3)
    eeg_data = eeg[0]
    labels = np.zeros((240, 1), dtype=np.int64)
    for class_index in range(1,num_classes):
        eeg_data = np.concatenate([eeg_data, eeg[class_index]], axis=2)
        labels[class_index * 6:(class_index + 1) * 6] = class_index
    eeg_data = eeg_data.transpose([2, 0, 1])
    return eeg_data[:num_trials].astype(np.float32), labels[:num_trials].reshape(-1)


def _load_beta_subject(
    subject_id: int,
    dataset_root: str,
    window_sec: float,
    num_trials: int,
    channel_indices: List[int],
    decay_sec: float = 0.14,
) -> Tuple[np.ndarray, np.ndarray]:
    data = loadmat(f'{dataset_root}/S{subject_id}.mat')

    start = int(round((0.5 + decay_sec) * 250))
    end = int(round((0.5 + decay_sec + window_sec) * 250))
    eeg = data['data'][0, 0][0][
        channel_indices,
        start:end,
        :,
        :,
    ]
    eeg = eeg.transpose(3, 2, 0, 1)
    eeg = eeg.reshape(-1, eeg.shape[2], eeg.shape[3])

    labels = np.array(
        np.round((data['data'][0, 0][1][0, 0][4] - 8) / 0.2),
        dtype=np.int64,
    ).reshape(-1)
    labels = np.repeat(labels, 4)

    sorted_indices = np.argsort(labels)
    eeg = eeg[sorted_indices]
    labels = labels[sorted_indices]

    return eeg[:num_trials].astype(np.float32), labels[:num_trials]


def load_subject_raw(config, subject_id: int, meta: DatasetMeta) -> Tuple[np.ndarray, np.ndarray]:
    dataset_root = get_dataset_root(config, meta.dataset_name)
    if meta.dataset_name == 'benchmark12':
        return _load_benchmark12_subject(subject_id, dataset_root, meta.window_sec)
    if meta.dataset_name == 'benchmark40':
        _, dataset_spec = get_active_dataset_config(config)
        channel_indices = dataset_spec.get('selected_channels', DEFAULT_BENCHMARK40_CHANNELS)
        return _load_benchmark40_subject(
            subject_id,
            dataset_root,
            meta.window_sec,
            meta.num_classes,
            meta.num_trials,
            channel_indices,
        )
    if meta.dataset_name == 'beta':
        _, dataset_spec = get_active_dataset_config(config)
        channel_indices = dataset_spec.get('selected_channels', DEFAULT_BETA_CHANNELS) or list(range(64))
        return _load_beta_subject(
            subject_id,
            dataset_root,
            meta.window_sec,
            meta.num_trials,
            channel_indices,
        )
    raise NotImplementedError(f'Dataset {meta.dataset_name} is not implemented in the unified suite.')


def compute_intra_fold_indices(meta: DatasetMeta, train_ratio: float, fold_id: int) -> Tuple[List[int], List[int]]:
    trials_per_class = meta.num_trials // meta.num_classes
    if meta.dataset_name == 'benchmark12':
        step = 3
    elif meta.dataset_name == 'benchmark40':
        step = 1
    elif meta.dataset_name == 'beta':
        step = 1
    else:
        raise NotImplementedError(f'Dataset {meta.dataset_name} is not implemented in the unified suite.')

    train_len = int(trials_per_class * train_ratio)
    if train_len <= 0 or train_len >= trials_per_class:
        raise ValueError(
            f'Unsupported {meta.dataset_name} train_ratio={train_ratio}; expected a value that leaves both train and test trials.'
        )

    start = (fold_id * step) % trials_per_class
    train_trials = {(start + offset) % trials_per_class for offset in range(train_len)}

    train_indices: List[int] = []
    test_indices: List[int] = []
    for class_offset in range(0, meta.num_trials, trials_per_class):
        for trial_offset in range(trials_per_class):
            sample_index = class_offset + trial_offset
            if trial_offset in train_trials:
                train_indices.append(sample_index)
            else:
                test_indices.append(sample_index)
    return train_indices, test_indices


def load_dataset_split(config, split_mode: str, subject_id: int = None, fold_id: int = 0, target_subject_id: int = None) -> SplitArrays:
    meta = build_dataset_meta(config, split_mode)
    dataset_cfg = get_dataset_settings(config)

    if split_mode == 'intra':
        if subject_id is None:
            raise ValueError('subject_id is required for intra-subject splits.')
        eeg_raw, labels = load_subject_raw(config, subject_id, meta)
        train_indices, test_indices = compute_intra_fold_indices(meta, float(dataset_cfg.get('train_ratio', 0.8)), int(fold_id))
        return SplitArrays(
            train_raw=eeg_raw[train_indices],
            test_raw=eeg_raw[test_indices],
            train_labels=labels[train_indices].astype(np.int64),
            test_labels=labels[test_indices].astype(np.int64),
            meta=meta,
            subject=subject_id,
            fold=fold_id,
        )

    if split_mode == 'loso':
        if target_subject_id is None:
            raise ValueError('target_subject_id is required for LOSO splits.')
        transfer_cfg = get_loso_transfer_config(config)
        experiment_seed = int(get_experiment_config(config).get('random_seed', 0))
        transfer_cfg['dataset_name'] = meta.dataset_name
        transfer_cfg['fold_id'] = int(fold_id)
        transfer_cfg['selection_seed'] = experiment_seed + int(target_subject_id) * 1000 + int(fold_id)
        source_subjects = []
        target_raw = None
        target_labels = None
        for subject in range(1, meta.num_subjects + 1):
            eeg_raw, labels = load_subject_raw(config, subject, meta)
            labels = labels.astype(np.int64)
            if subject == target_subject_id:
                target_raw = eeg_raw.astype(np.float32)
                target_labels = labels
            else:
                source_subjects.append((subject, eeg_raw.astype(np.float32), labels))
        if target_raw is None or target_labels is None:
            raise RuntimeError(f'Unable to build LOSO split for subject {target_subject_id}.')

        calibration_raw = None
        calibration_labels = None
        adaptation_raw = None
        adaptation_labels = None
        transfer_metadata = None

        if transfer_cfg['enabled']:
            transfer_split = build_loso_transfer_split(
                source_subjects=source_subjects,
                target_raw=target_raw,
                target_labels=target_labels,
                num_classes=meta.num_classes,
                transfer_cfg=transfer_cfg,
            )
            train_raw = transfer_split.train_raw
            train_labels = transfer_split.train_labels.astype(np.int64)
            test_raw = transfer_split.test_raw
            test_labels = transfer_split.test_labels.astype(np.int64)
            calibration_raw = transfer_split.calibration_raw
            calibration_labels = transfer_split.calibration_labels.astype(np.int64)
            adaptation_raw = transfer_split.adaptation_raw
            adaptation_labels = transfer_split.adaptation_labels.astype(np.int64)
            transfer_metadata = transfer_split.metadata
        else:
            train_raw = np.concatenate([raw for _, raw, _ in source_subjects], axis=0).astype(np.float32)
            train_labels = np.concatenate([labels for _, _, labels in source_subjects], axis=0).astype(np.int64)
            test_raw = target_raw.astype(np.float32)
            test_labels = target_labels.astype(np.int64)

        train_sort_idx = np.argsort(train_labels)
        test_sort_idx = np.argsort(test_labels)

        train_raw = train_raw[train_sort_idx]
        train_labels = train_labels[train_sort_idx]
        test_raw = test_raw[test_sort_idx]
        test_labels = test_labels[test_sort_idx]

        if calibration_labels is not None:
            calibration_sort_idx = np.argsort(calibration_labels)
            calibration_raw = calibration_raw[calibration_sort_idx]
            calibration_labels = calibration_labels[calibration_sort_idx]
        if adaptation_labels is not None:
            adaptation_sort_idx = np.argsort(adaptation_labels)
            adaptation_raw = adaptation_raw[adaptation_sort_idx]
            adaptation_labels = adaptation_labels[adaptation_sort_idx]

        return SplitArrays(
            train_raw=train_raw,
            test_raw=test_raw,
            train_labels=train_labels,
            test_labels=test_labels,
            meta=meta,
            target_subject=target_subject_id,
            fold=fold_id,
            calibration_raw=calibration_raw,
            calibration_labels=calibration_labels,
            adaptation_raw=adaptation_raw,
            adaptation_labels=adaptation_labels,
            transfer_metadata=transfer_metadata,
        )

    raise ValueError(f'Unsupported split mode: {split_mode}')


def apply_filter_bank(eeg: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    num_samples = eeg.shape[0]
    result = np.zeros((num_samples, max(meta.num_filter_banks, 1), meta.num_channels, meta.num_timesteps), dtype=np.float32)
    nyquist = meta.sample_rate / 2.0

    if meta.dataset_name == 'benchmark12':
        passband = [9, 18, 27, 36, 45, 54]
        stopband = [7, 15, 19, 28, 37, 46]
    else:
        passband = [8, 16, 24, 32, 40, 48]
        stopband = [6, 12, 18, 26, 34, 42]

    highcut_pass = 80
    highcut_stop = 90
    gpass = 3
    gstop = 40
    ripple = 0.5

    if meta.num_filter_banks:
        for filter_index in range(meta.num_filter_banks):
            wp = [passband[filter_index] / nyquist, highcut_pass / nyquist]
            ws = [stopband[filter_index] / nyquist, highcut_stop / nyquist]
            order, wn = signal.cheb1ord(wp, ws, gpass, gstop)
            b, a = signal.cheby1(order, ripple, wn, 'bandpass')
            filtered = signal.filtfilt(b, a, eeg, padlen=3 * (max(len(b), len(a)) - 1)).copy()
            result[:, filter_index, :, :] = filtered[:, :, :meta.num_timesteps]
    else:
        result[:, 0, :, :] = eeg[:, :, :meta.num_timesteps]
    return result


def flatten_structured_eeg(eeg: np.ndarray) -> np.ndarray:
    num_samples, num_filters, num_channels, num_timesteps = eeg.shape
    return eeg.reshape(num_samples, num_filters * num_channels, num_timesteps).astype(np.float32)


def _expand_mask_to_filter_bank(mask, meta) -> np.ndarray:
    """Replicate a raw (N, C, T) corruption mask across filter-bank sub-bands.

    flatten_structured_eeg reshapes (N, Nfb, C, T) -> (N, Nfb*C, T), so the sub-band
    index is the slower-varying one; broadcasting then reshaping the same way keeps the
    mask aligned with the flattened channel order.
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).astype(np.float32)
    num_samples, num_channels, num_timesteps = mask.shape
    num_banks = max(meta.num_filter_banks, 1)
    mask = mask[:, None, :, :num_timesteps]
    mask = np.broadcast_to(mask, (num_samples, num_banks, num_channels, num_timesteps))
    return np.ascontiguousarray(mask.reshape(num_samples, num_banks * num_channels, num_timesteps))


def _broadcast_raw_to_filter_bank(raw, meta) -> np.ndarray:
    """Place the unfiltered trial in every sub-band slot of the structured tensor.

    The filter bank sits in front of the denoiser, so part of the injected artifact is
    removed before the model ever sees it -- most starkly for EOG, whose energy lies below
    the passband. Feeding the unfiltered trial instead lets the denoiser act on the whole
    contamination while the target stays the filtered reference, so the generator has to
    absorb the band-splitting as well. Shapes are unchanged, so nothing downstream moves.
    """
    raw = np.asarray(raw, dtype=np.float32)
    num_samples, num_channels, num_timesteps = raw.shape
    num_banks = max(meta.num_filter_banks, 1)
    tiled = np.broadcast_to(raw[:, None, :, :], (num_samples, num_banks, num_channels, num_timesteps))
    return np.ascontiguousarray(tiled[:, :, :, :meta.num_timesteps])


def build_experiment_signals(config, split: SplitArrays) -> ExperimentSignals:
    experiment_cfg = config.get('experiment', {})
    noise_cfg = get_noise_config(config)
    task_mode = experiment_cfg.get('task_mode', 'downstream_only')
    augmentation_task = experiment_cfg.get('augmentation_task', 'denoise')

    clean_train = apply_filter_bank(split.train_raw, split.meta)
    clean_test = apply_filter_bank(split.test_raw, split.meta)

    if task_mode == 'augmentation' and augmentation_task == 'denoise':
        noisy_train_raw, train_masks = add_noise(
            split.train_raw.copy(),
            noise_family=noise_cfg['noise_family'],
            noise_type=noise_cfg['noise_type'],
            mode=noise_cfg['mode'],
            noise_ratio=noise_cfg['noise_ratio'],
            snr_db=noise_cfg['snr_db'],
            noise_src_num=noise_cfg['noise_src_num'],
            artifact_type=noise_cfg['artifact_type'],
            channel_gain_sigma=noise_cfg.get('channel_gain_sigma', 0.3),
            fs_signal=noise_cfg['fs_signal'],
            fs_noise_emg=noise_cfg['fs_noise_emg'],
            fs_noise_eog=noise_cfg['fs_noise_eog'],
        )
        noisy_test_raw, test_masks = add_noise(
            split.test_raw.copy(),
            noise_family=noise_cfg['noise_family'],
            noise_type=noise_cfg['noise_type'],
            mode=noise_cfg['mode'],
            noise_ratio=noise_cfg['noise_ratio'],
            snr_db=noise_cfg['snr_db'],
            noise_src_num=noise_cfg['noise_src_num'],
            artifact_type=noise_cfg['artifact_type'],
            channel_gain_sigma=noise_cfg.get('channel_gain_sigma', 0.3),
            fs_signal=noise_cfg['fs_signal'],
            fs_noise_emg=noise_cfg['fs_noise_emg'],
            fs_noise_eog=noise_cfg['fs_noise_eog'],
        )
        if bool((config.get('augmentation') or {}).get('unfiltered_source', False)):
            source_train = _broadcast_raw_to_filter_bank(noisy_train_raw.numpy(), split.meta)
            source_test = _broadcast_raw_to_filter_bank(noisy_test_raw.numpy(), split.meta)
        else:
            source_train = apply_filter_bank(noisy_train_raw.numpy().astype(np.float32), split.meta)
            source_test = apply_filter_bank(noisy_test_raw.numpy().astype(np.float32), split.meta)
        source_name = 'noisy'
        masks = SignalSet(
            train=_expand_mask_to_filter_bank(train_masks, split.meta),
            test=_expand_mask_to_filter_bank(test_masks, split.meta),
        )
    else:
        source_train = clean_train.copy()
        source_test = clean_test.copy()
        source_name = 'clean'
        masks = None

    return ExperimentSignals(
        meta=split.meta,
        clean=SignalSet(train=clean_train, test=clean_test),
        source=SignalSet(train=source_train, test=source_test),
        generated=None,
        train_labels=split.train_labels.astype(np.int64),
        test_labels=split.test_labels.astype(np.int64),
        source_name=source_name,
        masks=masks,
    )


def attach_generated_signals(signals: ExperimentSignals, generated_train: np.ndarray, generated_test: np.ndarray, method: str) -> ExperimentSignals:
    return replace(
        signals,
        generated=SignalSet(train=generated_train.astype(np.float32), test=generated_test.astype(np.float32)),
        augmenter_name=method,
    )


def canonicalize_view_name(view_name: str, source_name: str) -> str:
    aliases = {
        'original': 'clean',
        'clean_only': 'clean',
        'generated_only': 'generated',
        'source_only': 'source',
        'noisy': 'source' if source_name == 'noisy' else 'noisy',
        'ng': 'concat_source_generated',
        'cn': 'concat_clean_source',
        'cg': 'concat_clean_generated',
    }
    return aliases.get(view_name, view_name)


def _get_view_array(signals: ExperimentSignals, split_name: str, canonical_view: str) -> np.ndarray:
    clean = getattr(signals.clean, split_name)
    source = getattr(signals.source, split_name)
    generated = None if signals.generated is None else getattr(signals.generated, split_name)

    if canonical_view == 'clean':
        return clean
    if canonical_view == 'source':
        return source
    if canonical_view == 'generated':
        if generated is None:
            raise ValueError('The selected view requires generated signals, but no augmenter has been run.')
        return generated
    if canonical_view == 'concat_clean_generated':
        if generated is None:
            raise ValueError('concat_clean_generated requires generated signals.')
        return np.concatenate([clean, generated], axis=-1)
    if canonical_view == 'concat_source_generated':
        if generated is None:
            raise ValueError('concat_source_generated requires generated signals.')
        return np.concatenate([source, generated], axis=-1)
    if canonical_view == 'concat_clean_source':
        return np.concatenate([clean, source], axis=-1)
    raise ValueError(f'Unsupported signal view: {canonical_view}')


def resolve_view_pair(signals: ExperimentSignals, train_view: str, test_view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    canonical_train = canonicalize_view_name(train_view, signals.source_name)
    canonical_test = canonicalize_view_name(test_view, signals.source_name)

    train_labels = signals.train_labels
    test_labels = signals.test_labels

    if canonical_train == 'double_batch_clean_generated':
        if signals.generated is None:
            raise ValueError('double_batch_clean_generated requires generated signals.')
        train_x = np.concatenate([signals.clean.train, signals.generated.train], axis=0)
        train_y = np.concatenate([train_labels, train_labels], axis=0)
    elif canonical_train == 'double_batch_source_generated':
        if signals.generated is None:
            raise ValueError('double_batch_source_generated requires generated signals.')
        train_x = np.concatenate([signals.source.train, signals.generated.train], axis=0)
        train_y = np.concatenate([train_labels, train_labels], axis=0)
    else:
        train_x = _get_view_array(signals, 'train', canonical_train)
        train_y = train_labels

    test_x = _get_view_array(signals, 'test', canonical_test)
    return train_x, train_y, test_x, test_labels, canonical_train, canonical_test


# Backward-compatible aliases for older imports.
build_meta = build_dataset_meta
load_split = load_dataset_split
filter_bank = apply_filter_bank
flatten_structured = flatten_structured_eeg
canonicalize_view = canonicalize_view_name
