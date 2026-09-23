"""Artifact injection: the paired data DenoFlow is trained on.

Public SSVEP datasets record the contaminated signal only, so a clean/contaminated pair has
to be built. Physiological artifacts are non-stationary, spatially structured and occupy
specific bands, so recorded EMG and EOG epochs from EEGdenoiseNet are injected rather than
synthetic perturbations.

For an uncontaminated trial X and the k-th artifact segment s_k, a log-normal per-electrode
gain spreads the segment over the montage, A_k = g_k s_k^T, a coarse stand-in for volume
conduction. A binary mask M selects the contaminated region -- a contiguous time span across
all electrodes, or a subset of electrodes across the whole trial. Each segment is scaled so
that inside the mask it meets a target ratio gamma,

    alpha_k = sqrt( E[(X . M)^2] / ( E[(A_k . M)^2] * 10^(gamma/10) ) ),
    X'      = X + ( sum_k alpha_k A_k ) . M .

Segments are scaled one at a time, so with K of them the ratio realised over the whole mask
is about gamma - 10 log10(K): the EMG and EOG conditions draw one segment and sit at gamma,
while the mixed condition draws one from each library and is some 3 dB harsher.
"""
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

EMG_BANK = os.path.join('Data', 'EMG_all_epochs.npy')
EOG_BANK = os.path.join('Data', 'EOG_all_epochs.npy')

_BANKS = {}


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _bank(path):
    """Load and cache one artifact library; both are small enough to keep resident."""
    if path not in _BANKS:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download the EMG and EOG epochs of EEGdenoiseNet and "
                f"place them under Data/ (see the README).")
        _BANKS[path] = torch.tensor(np.load(path, allow_pickle=True).astype(np.float32))
    return _BANKS[path]


def _resample(x, orig_fs, target_fs):
    if orig_fs == target_fs:
        return x.clone()
    length = max(1, int(round(x.shape[0] * float(target_fs) / float(orig_fs))))
    return F.interpolate(x.view(1, 1, -1), size=length, mode='linear',
                         align_corners=False).view(-1)


def _pick_segment(bank, target_len, fs_signal, fs_noise):
    """Draw one segment, match it to the trial's sampling rate and length, standardise it."""
    seg = bank[torch.randint(0, bank.shape[0], (1,)).item()].clone()
    seg = _resample(seg, fs_noise, fs_signal)
    if seg.shape[0] > target_len:
        start = random.randint(0, seg.shape[0] - target_len)
        seg = seg[start:start + target_len].clone()
    elif seg.shape[0] < target_len:
        seg = seg.repeat(math.ceil(target_len / seg.shape[0]))[:target_len].clone()
    return (seg - seg.mean()) / (seg.std() + 1e-8)


def _sample_sources(artifact_type, target_len, fs_signal, fs_emg, fs_eog):
    """One segment per library: 'emg' and 'eog' give one source, 'both' gives two."""
    out = []
    if artifact_type in ('emg', 'both'):
        out.append(_pick_segment(_bank(EMG_BANK), target_len, fs_signal, fs_emg))
    if artifact_type in ('eog', 'both'):
        out.append(_pick_segment(_bank(EOG_BANK), target_len, fs_signal, fs_eog))
    if not out:
        raise ValueError(f"artifact_type must be emg, eog or both, got {artifact_type}")
    return torch.stack(out, dim=0)


def _time_mask(c, t, ratio, device):
    """A contiguous span covering `ratio` of the trial, across every electrode."""
    mask = torch.zeros((c, t), dtype=torch.bool, device=device)
    span = min(max(1, int(round(t * ratio))), t)
    start = random.randint(0, t - span)
    mask[:, start:start + span] = True
    return mask


def _channel_mask(c, t, ratio, device):
    """A random subset covering `ratio` of the electrodes, across the whole trial."""
    mask = torch.zeros((c, t), dtype=torch.bool, device=device)
    k = min(max(1, int(round(c * ratio))), c)
    mask[torch.randperm(c, device=device)[:k], :] = True
    return mask


def _scale_to_snr(clean, artifact, mask, snr_db):
    """Eq. (1): the scale that puts this segment at `snr_db` inside the mask."""
    mask_f = mask.float()
    p_signal = torch.mean((clean * mask_f) ** 2)
    p_noise = torch.mean((artifact * mask_f) ** 2)
    if p_signal.item() < 1e-12 or p_noise.item() < 1e-12:
        return torch.zeros_like(artifact)
    scale = torch.sqrt(p_signal / (p_noise * 10.0 ** (snr_db / 10.0) + 1e-12))
    return artifact * scale


def inject(eeg, artifact_type='both', mode='time', ratio=0.5, snr_db=0.0,
           channel_gain_sigma=0.3, fs_signal=250, fs_emg=256, fs_eog=256):
    """Contaminate a batch of trials.

    Args:
        eeg: (N, C, T) or (C, T), array or tensor.
        artifact_type: 'emg', 'eog' or 'both'.
        mode: 'time' for a contiguous span, 'space' for a subset of electrodes.
        ratio: mask fraction.
        snr_db: target ratio gamma, per segment, inside the mask.

    Returns:
        (contaminated, mask), both with the shape of `eeg`.
    """
    x = eeg if torch.is_tensor(eeg) else torch.tensor(eeg, dtype=torch.float32)
    x = x.detach().clone().float()
    squeeze_back = x.ndim == 2
    if squeeze_back:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("eeg must have shape (C, T) or (N, C, T)")

    n, c, t = x.shape
    out = x.clone()
    masks = torch.zeros_like(x, dtype=torch.bool)

    for i in range(n):
        sources = _sample_sources(artifact_type, t, fs_signal, fs_emg, fs_eog).to(x.device)
        # one mask per trial, shared by every segment, so the target ratio is comparable
        mask = (_time_mask if mode == 'time' else _channel_mask)(c, t, ratio, x.device)
        artifact = torch.zeros((c, t), device=x.device, dtype=x.dtype)
        for source in sources:
            gain = torch.exp(torch.randn(c, device=x.device) * channel_gain_sigma)
            artifact = artifact + _scale_to_snr(x[i], gain[:, None] * source[None, :],
                                                mask, snr_db)
        out[i] = x[i] + artifact * mask.float()
        masks[i] = mask

    if squeeze_back:
        return out[0], masks[0]
    return out, masks
