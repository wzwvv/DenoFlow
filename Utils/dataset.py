"""Loading, splitting and contaminating the two SSVEP datasets.

Every experiment is within subject: the folds of one participant are split chronologically,
so training and test trials come from the same person but different blocks. Artifacts are
injected into the raw trials, then the reference X and the contaminated X' are passed through
the same filter bank, giving the pair (X~, X~') the model is trained on.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

from Utils.filterbank import apply_filter_bank, flatten
from Utils.inject import inject

DECAY_SEC = 0.14  # visual latency dropped from the head of every trial

# nine posterior channels of the 64-electrode Benchmark montage
BENCHMARK_CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]


@dataclass
class DatasetMeta:
    name: str
    num_subjects: int
    num_channels: int
    num_classes: int
    num_trials: int
    sample_rate: int
    window_sec: float
    num_banks: int

    @property
    def num_timesteps(self) -> int:
        return int(self.window_sec * self.sample_rate)

    @property
    def flat_channels(self) -> int:
        return self.num_channels * max(self.num_banks, 1)


META = {
    'nakanishi2015': DatasetMeta('nakanishi2015', 10, 8, 12, 180, 256, 1.0, 3),
    'benchmark': DatasetMeta('benchmark', 35, 9, 40, 240, 250, 1.0, 3),
}


def load_subject(name: str, root: str, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (trials, labels) for one participant, ordered class by class."""
    meta = META[name]
    if name == 'nakanishi2015':
        samples = loadmat(f'{root}/S{subject_id}.mat')['eeg'][0]
        eeg = samples[0]
        labels = np.zeros(meta.num_trials, dtype=np.int64)
        for cls in range(1, meta.num_classes):
            eeg = np.concatenate([eeg, samples[cls]], axis=2)
            labels[cls * 15:(cls + 1) * 15] = cls
        eeg = eeg.transpose(2, 0, 1)
        start = int(DECAY_SEC * meta.sample_rate)
        eeg = eeg[:, :, start:start + meta.num_timesteps]
    else:
        data = loadmat(f'{root}/S{subject_id}.mat')['data']
        start = int((0.5 + DECAY_SEC) * meta.sample_rate)
        eeg = data[BENCHMARK_CHANNELS, start:start + meta.num_timesteps, :, :]
        eeg = eeg.transpose(2, 0, 1, 3)
        stacked = eeg[0]
        labels = np.zeros(meta.num_trials, dtype=np.int64)
        for cls in range(1, meta.num_classes):
            stacked = np.concatenate([stacked, eeg[cls]], axis=2)
            labels[cls * 6:(cls + 1) * 6] = cls
        eeg = stacked.transpose(2, 0, 1)
    return eeg.astype(np.float32), labels


def fold_indices(meta: DatasetMeta, fold_id: int, train_ratio: float = 0.8
                 ) -> Tuple[List[int], List[int]]:
    """Chronological within-subject split: whole blocks go to train or to test."""
    per_class = meta.num_trials // meta.num_classes
    step = 3 if meta.name == 'nakanishi2015' else 1
    train_len = int(per_class * train_ratio)
    if not 0 < train_len < per_class:
        raise ValueError(f'train_ratio={train_ratio} leaves no test trials')

    start = (fold_id * step) % per_class
    train_blocks = {(start + offset) % per_class for offset in range(train_len)}

    train, test = [], []
    for class_offset in range(0, meta.num_trials, per_class):
        for trial in range(per_class):
            (train if trial in train_blocks else test).append(class_offset + trial)
    return train, test


def build_views(eeg: np.ndarray, meta: DatasetMeta, artifact: str, mode: str,
                ratio: float, snr_db: float):
    """Contaminate, then filter both signals.

    Returns (reference, contaminated, mask), each (N, Nc, T) with Nc = Ne * Nb.
    """
    noisy, mask = inject(eeg, artifact_type=artifact, mode=mode, ratio=ratio,
                         snr_db=snr_db, fs_signal=meta.sample_rate)
    noisy = noisy.numpy()
    clean_fb = apply_filter_bank(eeg, meta.name, meta.sample_rate, meta.num_banks)
    noisy_fb = apply_filter_bank(noisy, meta.name, meta.sample_rate, meta.num_banks)
    mask_fb = np.repeat(mask.numpy()[:, None], max(meta.num_banks, 1), axis=1)
    return flatten(clean_fb), flatten(noisy_fb), flatten(mask_fb.astype(np.float32))


def make_loaders(clean, noisy, labels, train_idx, test_idx, batch_size):
    """Loaders yielding (reference, contaminated, label); the network reads one plane each."""
    def build(idx, shuffle):
        tensors = (torch.tensor(clean[idx]).unsqueeze(1),
                   torch.tensor(noisy[idx]).unsqueeze(1),
                   torch.tensor(labels[idx], dtype=torch.int64))
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                          shuffle=shuffle, drop_last=False)
    return build(train_idx, True), build(test_idx, False)
