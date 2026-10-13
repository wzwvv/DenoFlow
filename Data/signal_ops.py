import os
import random
from typing import Dict, Iterable
import math
import numpy as np
import torch
from scipy.signal import hilbert
import torch.nn.functional as F


DIRECT_AUGMENTATION_METHODS = {
    'Scale',
    'Noise',
    'Flip',
    'FShift',
    'DWTaug',
    'HHTaug',
}


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_random_seed = set_random_seed


def add_time_noise(x, noise_ratio=0.1, snr_db=10):
    batch_size, num_channels, num_timesteps = x.shape
    noisy_length = max(1, int(num_timesteps * noise_ratio))
    masks = torch.zeros_like(x, dtype=torch.bool)

    for batch_index in range(batch_size):
        start = random.randint(0, max(0, num_timesteps - noisy_length))
        end = start + noisy_length
        for channel_index in range(num_channels):
            signal_power = torch.mean(x[batch_index, channel_index, :] ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = torch.sqrt(noise_power)
            noise = torch.randn(end - start, device=x.device) * noise_std
            x[batch_index, channel_index, start:end] += noise
        masks[batch_index, :, start:end] = True

    return x, masks


def add_channel_noise(x, noise_ratio=0.2, snr_db=10):
    batch_size, num_channels, num_timesteps = x.shape
    masks = torch.zeros_like(x, dtype=torch.bool)
    noisy_channel_count = max(1, int(num_channels * noise_ratio))

    for batch_index in range(batch_size):
        noisy_channels = random.sample(range(num_channels), min(noisy_channel_count, num_channels))
        for channel_index in noisy_channels:
            signal_power = torch.mean(x[batch_index, channel_index, :] ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = torch.sqrt(noise_power)
            noise = torch.randn(num_timesteps, device=x.device) * noise_std
            x[batch_index, channel_index, :] += noise
            masks[batch_index, channel_index, :] = True

    return x, masks


def add_frequency_noise(x, noise_ratio=0.2, snr_db=10, freq_band=None):
    batch_size, num_channels, num_timesteps = x.shape
    masks = torch.zeros_like(x, dtype=torch.bool)
    fft_frequencies = torch.fft.fftfreq(num_timesteps).to(x.device)

    for batch_index in range(batch_size):
        band_length = max(1, int(num_timesteps * noise_ratio / 4))
        start = random.randint(0, max(0, num_timesteps // 4 - band_length))
        for channel_index in range(num_channels):
            signal_fft = torch.fft.fft(x[batch_index, channel_index, :])
            signal_power = torch.mean(torch.abs(signal_fft[: num_timesteps // 4]) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = torch.sqrt(noise_power)

            if freq_band is not None:
                low_hz, high_hz = freq_band
                mask = (torch.abs(fft_frequencies) >= low_hz) & (torch.abs(fft_frequencies) <= high_hz)
            else:
                mask = torch.zeros(num_timesteps, dtype=torch.bool, device=x.device)
                mask[start:start + band_length] = True
                if start == 0:
                    mask[-band_length:] = True
                else:
                    mask[-(start + band_length):-start] = True

            noise_real = torch.randn(num_timesteps, device=x.device) * noise_std
            noise_imag = torch.randn(num_timesteps, device=x.device) * noise_std
            noisy_fft = signal_fft + (noise_real + 1j * noise_imag) * mask
            x[batch_index, channel_index, :] = torch.fft.ifft(noisy_fft).real
            masks[batch_index, channel_index, :] = mask

    return x, masks


def _to_eeg_tensor(x):
    if torch.is_tensor(x):
        eeg = x.detach().clone().float()
    else:
        eeg = torch.tensor(x, dtype=torch.float32)
    return eeg


def _ensure_3d(x):
    """
    输入:
        [C, T] 或 [N, C, T]
    输出:
        eeg_3d, squeeze_back
    """
    eeg = _to_eeg_tensor(x)
    if eeg.ndim == 2:
        return eeg.unsqueeze(0), True
    elif eeg.ndim == 3:
        return eeg, False
    else:
        raise ValueError("x must have shape [C, T] or [N, C, T]")


def _resample_1d_torch(x, orig_fs, target_fs):
    """
    x: [T]
    返回重采样后的 1D tensor
    """
    if orig_fs == target_fs:
        return x.clone()

    x_in = x.view(1, 1, -1)  # [1, 1, T]
    new_len = max(1, int(round(x.shape[0] * float(target_fs) / float(orig_fs))))
    y = F.interpolate(x_in, size=new_len, mode='linear', align_corners=False)
    return y.view(-1)


def _pick_noise_segment(bank, target_len, fs_signal=None, fs_noise=None):
    """
    从 bank 中随机取一条噪声：
    1) 如有需要，先把噪声从 fs_noise 重采样到 fs_signal
    2) 再裁剪 / 重复到 target_len
    """
    num_segments, bank_len = bank.shape
    idx = torch.randint(0, num_segments, (1,), device=bank.device).item()
    seg = bank[idx].clone()

    # 先对齐采样率
    if fs_signal is not None and fs_noise is not None and fs_signal != fs_noise:
        seg = _resample_1d_torch(seg, orig_fs=fs_noise, target_fs=fs_signal)

    seg_len = seg.shape[0]

    # 再对齐长度
    if seg_len == target_len:
        out = seg
    elif seg_len > target_len:
        start = random.randint(0, seg_len - target_len)
        out = seg[start:start + target_len].clone()
    else:
        reps = math.ceil(target_len / seg_len)
        out = seg.repeat(reps)[:target_len].clone()

    # 标准化
    out = (out - out.mean()) / (out.std() + 1e-8)
    return out


def _build_time_mask(c, t, noise_ratio, device):
    """
    随机选连续时间窗口，加噪比例 = noise_ratio
    """
    mask = torch.zeros((c, t), dtype=torch.bool, device=device)
    span = max(1, int(round(t * noise_ratio)))
    span = min(span, t)

    start = random.randint(0, t - span)
    mask[:, start:start + span] = True
    return mask


def _build_channel_mask(c, t, noise_ratio, device):
    """
    随机选通道，加噪比例 = noise_ratio
    """
    mask = torch.zeros((c, t), dtype=torch.bool, device=device)
    k = max(1, int(round(c * noise_ratio)))
    k = min(k, c)

    ch_idx = torch.randperm(c, device=device)[:k]
    mask[ch_idx, :] = True
    return mask


def _scale_artifact_to_snr(clean, artifact, mask, snr_db):
    """
    只在 mask 区域内，将伪迹缩放到目标 SNR
    SNR = 10 * log10(P_signal / P_noise)
    """
    mask_f = mask.float()

    clean_masked = clean * mask_f
    art_masked = artifact * mask_f

    p_signal = torch.mean(clean_masked ** 2)
    p_noise = torch.mean(art_masked ** 2)

    if p_signal.item() < 1e-12 or p_noise.item() < 1e-12:
        return torch.zeros_like(artifact)

    target_ratio = 10.0 ** (snr_db / 10.0)
    scale = torch.sqrt(p_signal / (p_noise * target_ratio + 1e-12))
    return artifact * scale


# =========================================================
# 真实伪迹噪声：EMG / EOG / BOTH
# =========================================================
def _sample_real_sources(
    artifact_type,
    noise_src_num,
    target_len,
    fs_signal=None,
    fs_noise_emg=None,
    fs_noise_eog=None
):
    sources = []
    emg_bank = torch.tensor(np.load(rf'Data/EMG_all_epochs.npy', allow_pickle=True).astype(np.float32))
    eog_bank = torch.tensor(np.load(rf'Data/EOG_all_epochs.npy', allow_pickle=True).astype(np.float32))
    

    # print("emg_bank:", emg_bank.shape)
    # print("eog_bank:", eog_bank.shape)

    if artifact_type == 'emg':
        for _ in range(noise_src_num):
            sources.append(_pick_noise_segment(emg_bank, target_len, fs_signal, fs_noise_emg))

    elif artifact_type == 'eog':
        for _ in range(noise_src_num):
            sources.append(_pick_noise_segment(eog_bank, target_len, fs_signal, fs_noise_eog))

    elif artifact_type == 'both':
        # both: 分别从 EMG / EOG 库各取 src_num 条，共 2 * src_num 条
        for _ in range(noise_src_num):
            sources.append(_pick_noise_segment(emg_bank, target_len, fs_signal, fs_noise_emg))
        for _ in range(noise_src_num):
            sources.append(_pick_noise_segment(eog_bank, target_len, fs_signal, fs_noise_eog))

    else:
        raise ValueError(f"Unsupported artifact_type: {artifact_type}")

    return torch.stack(sources, dim=0)


def _add_real_artifact_noise(
    eeg,
    artifact_type='both',
    mode='time',
    noise_ratio=0.25,
    noise_src_num=1,
    snr_db=5,
    channel_gain_sigma=0.3,
    fs_signal=None,
    fs_noise_emg=None,
    fs_noise_eog=None,
):
    """
    方案 A —— 每条源各自 snr_db，再叠加。
        1) 抽源
            - artifact_type='emg' / 'eog': 各 noise_src_num 条
            - artifact_type='both'       : EMG 和 EOG 各 noise_src_num 条 (共 2*noise_src_num)
        2) 对每条源 src_k:
            (a) 采样 (C,) 的 log-normal 通道增益 gains_k = exp(N(0, channel_gain_sigma))
            (b) 单源伪迹: one_k = gains_k[:, None] * src_k[None, :]   (C, T)
            (c) 在 mask 区域内把 one_k 缩放到 SNR(one_k, clean) = snr_db
        3) 累加: artifact = Σ_k one_k_scaled
        4) noisy = eeg + artifact * mask
    注意：每条源都"单独 snr_db"，所以叠加后的全局 SNR ≈ snr_db − 10·log10(K)（独立源近似）。

    eeg: [N, C, T]
    返回:
        noisy_eeg: [N, C, T]
        masks    : [N, C, T] (bool)
    """
    if mode not in ['time', 'space']:
        raise ValueError("mode must be 'time' or 'space'")

    n, c, t = eeg.shape
    out = eeg.clone()
    masks = torch.zeros_like(eeg, dtype=torch.bool)

    for i in range(n):
        # 1) 抽源 noise_mat: [K, T]
        noise_mat = _sample_real_sources(
            artifact_type=artifact_type,
            noise_src_num=noise_src_num,
            target_len=t,
            fs_signal=fs_signal,
            fs_noise_emg=fs_noise_emg,
            fs_noise_eog=fs_noise_eog,
        ).to(eeg.device)
        k = noise_mat.shape[0]

        # 2) mask 在循环外建一次（每条源共享同一 mask 区域，SNR 才有可比性）
        if mode == 'time':
            mask = _build_time_mask(c, t, noise_ratio, eeg.device)
        else:
            mask = _build_channel_mask(c, t, noise_ratio, eeg.device)

        # 3) 逐源生成 + 单独缩放到 snr_db + 累加
        artifact = torch.zeros((c, t), device=eeg.device, dtype=eeg.dtype)
        for src_idx in range(k):
            src_k = noise_mat[src_idx]                                     # (T,)
            gains_k = torch.exp(torch.randn(c, device=eeg.device) * channel_gain_sigma)  # (C,)
            one_k = gains_k[:, None] * src_k[None, :]                      # (C, T)
            one_k = _scale_artifact_to_snr(eeg[i], one_k, mask, snr_db)
            artifact = artifact + one_k

        # 4) 注入
        out[i] = eeg[i] + artifact * mask.float()
        masks[i] = mask

    return out, masks


# =========================================================
# 第一大类：高斯噪声（你原来的 5 种）
# =========================================================
def _add_gaussian_family_noise(eeg, noise_type='time', noise_ratio=0.25, snr_db=5):
    """
    eeg: [N, C, T]
    noise_type:
        'time' / 'space' / 'freq' / 'random' / 'all'
    """
    masks = torch.zeros_like(eeg, dtype=torch.bool)

    if noise_type == 'time':
        eeg, masks = add_time_noise(eeg, noise_ratio, snr_db)

    elif noise_type == 'space':
        eeg, masks = add_channel_noise(eeg, noise_ratio, snr_db)

    elif noise_type == 'freq':
        eeg, masks = add_frequency_noise(eeg, noise_ratio, snr_db)

    elif noise_type == 'random':
        for sample_index in range(eeg.shape[0]):
            sampled_type = random.choice(['time', 'space', 'freq'])
            if sampled_type == 'time':
                eeg[sample_index:sample_index + 1], masks[sample_index:sample_index + 1] = \
                    add_time_noise(eeg[sample_index:sample_index + 1], noise_ratio, snr_db)
            elif sampled_type == 'space':
                eeg[sample_index:sample_index + 1], masks[sample_index:sample_index + 1] = \
                    add_channel_noise(eeg[sample_index:sample_index + 1], noise_ratio, snr_db)
            else:
                eeg[sample_index:sample_index + 1], masks[sample_index:sample_index + 1] = \
                    add_frequency_noise(eeg[sample_index:sample_index + 1], noise_ratio, snr_db)

    elif noise_type == 'all':
        eeg, mask_time = add_time_noise(eeg, noise_ratio, snr_db)
        eeg, mask_space = add_channel_noise(eeg, noise_ratio, snr_db)
        eeg, mask_freq = add_frequency_noise(eeg, noise_ratio, snr_db)
        masks = mask_time | mask_space | mask_freq

    else:
        raise ValueError(f"Unsupported gaussian noise_type: {noise_type}")

    return eeg, masks


# =========================================================
# 统一入口
# =========================================================
def add_noise(
    x,
    noise_family='gaussian',     # 'gaussian' or 'real'
    noise_type='time',           # gaussian family: time/space/freq/random/all
    mode='time',                 # real family: time/space
    noise_ratio=0.25,
    snr_db=5,
    noise_src_num=1,             # real family only
    artifact_type='both',        # real family only: emg/eog/both
    channel_gain_sigma=0.3,      # real family only: log-normal 通道增益 σ
    fs_signal=250,      # 你的原始 EEG
    fs_noise_emg=256,   # EMG bank 采样率
    fs_noise_eog=256    # EOG bank 采样率
):
    """
    两大类噪声入口

    1) noise_family='gaussian'
       - noise_type: 'time' / 'space' / 'freq' / 'random' / 'all'

    2) noise_family='real'
       - artifact_type: 'emg' / 'eog' / 'both'
       - mode: 'time' / 'space'
       - noise_src_num: 混入多少条独立单通道伪迹源
       - noise_ratio:
           * mode='time'  -> 污染多少比例的时间窗口
           * mode='space' -> 污染多少比例的通道
    """
    eeg, squeeze_back = _ensure_3d(x)

    if noise_family == 'gaussian':
        eeg, masks = _add_gaussian_family_noise(
            eeg,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
            snr_db=snr_db
        )

    elif noise_family == 'real':
        eeg, masks = _add_real_artifact_noise(
            eeg,
            artifact_type=artifact_type,
            mode=mode,
            noise_ratio=noise_ratio,
            noise_src_num=noise_src_num,
            snr_db=snr_db,
            channel_gain_sigma=channel_gain_sigma,
            fs_signal=fs_signal,      # 你的原始 EEG
            fs_noise_emg=fs_noise_emg,   # EMG bank 采样率
            fs_noise_eog=fs_noise_eog    # EOG bank 采样率
            
        )

    else:
        raise ValueError("noise_family must be 'gaussian' or 'real'")

    if squeeze_back:
        return eeg[0], masks[0]
    return eeg, masks


def is_direct_augmentation_method(method: str) -> bool:
    return str(method) in DIRECT_AUGMENTATION_METHODS


def apply_direct_augmentation(
    eeg: np.ndarray,
    labels: np.ndarray,
    method: str,
    sample_rate: int,
    method_cfg: Dict | None = None,
) -> np.ndarray:
    method_cfg = method_cfg or {}
    eeg = np.asarray(eeg, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)

    if eeg.ndim != 3:
        raise ValueError(f'Expected EEG to have shape (N, C, T), got {eeg.shape}')
    if eeg.shape[0] != labels.shape[0]:
        raise ValueError(f'Expected labels to match EEG samples, got {eeg.shape[0]} and {labels.shape[0]}')

    if method == 'Scale':
        return scale_augmentation(eeg, float(method_cfg.get('scale_factor', 1.05)))
    if method == 'Noise':
        return noise_augmentation(eeg, float(method_cfg.get('noise_std_divisor', 5.0)))
    if method == 'Flip':
        return flip_augmentation(eeg)
    if method == 'FShift':
        return frequency_shift_augmentation(eeg, float(method_cfg.get('freq_shift_hz', -0.2)), sample_rate)
    if method == 'DWTaug':
        return dwt_augmentation(eeg, labels, str(method_cfg.get('wavelet', 'db4')))
    if method == 'HHTaug':
        return hht_augmentation(eeg, labels)

    raise ValueError(f'Unsupported direct augmentation method: {method}')


def scale_augmentation(eeg: np.ndarray, scale_factor: float = 1.05) -> np.ndarray:
    return (eeg * scale_factor).astype(np.float32)


def noise_augmentation(eeg: np.ndarray, noise_std_divisor: float = 5.0) -> np.ndarray:
    sample_std = np.std(eeg, axis=(1, 2), keepdims=True)
    random_noise = np.random.rand(*eeg.shape).astype(np.float32) - 0.5
    augmented = eeg + random_noise * sample_std / max(noise_std_divisor, 1e-6)
    return augmented.astype(np.float32)


def flip_augmentation(eeg: np.ndarray) -> np.ndarray:
    augmented = -1.0 * eeg
    augmented = augmented - np.min(augmented, axis=(1, 2), keepdims=True)
    return augmented.astype(np.float32)


def _next_pow2(value: int) -> int:
    return int(np.ceil(np.log2(max(1, abs(value)))))


def _frequency_shift_single(sample: np.ndarray, freq_shift_hz: float, dt: float) -> np.ndarray:
    num_timesteps, num_channels = sample.shape
    padding_len = 2 ** _next_pow2(num_timesteps)
    padding = np.zeros((padding_len - num_timesteps, num_channels), dtype=np.float32)
    padded = np.vstack((sample, padding))
    analytic = hilbert(padded, axis=0)
    time_axis = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * freq_shift_hz * dt * time_axis)
    shifted = np.zeros_like(sample, dtype=np.float32)
    for channel_index in range(num_channels):
        shifted[:, channel_index] = (analytic[:, channel_index] * shift_func)[:num_timesteps].real.astype(np.float32)
    return shifted


def frequency_shift_augmentation(eeg: np.ndarray, freq_shift_hz: float = -0.2, sample_rate: int = 250) -> np.ndarray:
    dt = 1.0 / float(sample_rate)
    samples_last = np.swapaxes(eeg, 1, 2)
    shifted = np.stack([_frequency_shift_single(sample, freq_shift_hz, dt) for sample in samples_last], axis=0)
    return np.swapaxes(shifted, 1, 2).astype(np.float32)


def _iterate_class_indices(labels: np.ndarray) -> Iterable[np.ndarray]:
    for class_id in np.unique(labels):
        yield np.where(labels == class_id)[0]


def _require_pywt():
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise ImportError('DWTaug requires the `pywt` package to be installed.') from exc
    return pywt


def _require_emd():
    try:
        from pyhht import EMD  # type: ignore
    except ImportError as exc:
        raise ImportError('HHTaug requires the `pyhht` package to be installed.') from exc
    return EMD


def dwt_augmentation(eeg: np.ndarray, labels: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
    pywt = _require_pywt()
    augmented = np.empty_like(eeg, dtype=np.float32)
    num_timesteps = eeg.shape[-1]

    for class_indices in _iterate_class_indices(labels):
        class_eeg = eeg[class_indices]
        reversed_eeg = class_eeg[::-1]
        src_approx, src_detail = pywt.dwt(class_eeg, wavelet, axis=-1)
        rev_approx, rev_detail = pywt.dwt(reversed_eeg, wavelet, axis=-1)
        class_aug = pywt.idwt(src_approx, rev_detail, wavelet, axis=-1)
        augmented[class_indices] = class_aug[..., :num_timesteps].astype(np.float32)

    return augmented


def _split_imfs(imfs: np.ndarray):
    imfs = np.asarray(imfs, dtype=np.float32)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.shape[0] <= 1:
        return imfs, imfs[-1:]
    return imfs[:-1], imfs[-1:]


def hht_augmentation(eeg: np.ndarray, labels: np.ndarray) -> np.ndarray:
    EMD = _require_emd()
    augmented = np.empty_like(eeg, dtype=np.float32)

    for class_indices in _iterate_class_indices(labels):
        class_eeg = eeg[class_indices]
        reversed_eeg = class_eeg[::-1]
        class_aug = np.zeros_like(class_eeg, dtype=np.float32)
        for sample_index in range(class_eeg.shape[0]):
            for channel_index in range(class_eeg.shape[1]):
                src_imfs = EMD(class_eeg[sample_index, channel_index]).decompose()
                rev_imfs = EMD(reversed_eeg[sample_index, channel_index]).decompose()
                src_main, _ = _split_imfs(src_imfs)
                _, rev_residual = _split_imfs(rev_imfs)
                merged = np.concatenate([src_main, rev_residual], axis=0)
                class_aug[sample_index, channel_index] = np.sum(merged, axis=0, dtype=np.float32)
        augmented[class_indices] = class_aug

    return augmented.astype(np.float32)