"""
去噪场景评估指标 + CSV 写入。
与 Support/experiment.py 中下游任务的 ACC/ITR CSV 共用 results 目录。
"""
import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


# ===========================================================
# CSV 列定义（与 experiment.py 的 RESULT_COLUMNS 风格保持一致）
# ===========================================================
DENOISE_RESULT_COLUMNS = [
    'timestamp',
    'split_mode',
    'subject',
    'fold',
    'target_subject',
    'augmenter',
    'artifact_type',     # emg / eog / both（来自 augmentation_task 配置）
    'num_samples',
    'cc',
    'rrmse',
    'snr_in',            # 输入 SNR (dB)，noisy vs clean
    'snr_out',           # 输出 SNR (dB)，denoised vs clean
    'prd',               # 波形失真百分比 (%)
    'snri',              # SNR 提升量 (dB)
    'nrr',               # 噪声残留率
    'snr_in_global',     # 全局 SNR (先聚合能量再取对数)
    'snr_out_global',
    'snri_global',       # 通道级污染下唯一有意义的 SNRI
]


def get_denoise_results_path(config: Dict, filename: str = 'denoise_results.csv') -> str:
    """
    复用 experiment.py 已经创建好的 runtime results 目录。
    """
    runtime_paths = config.get('_runtime', {}).get('output_paths', {})
    result_dir = runtime_paths.get('results')
    if not result_dir:
        from Support.config import get_output_paths
        result_dir = get_output_paths(config).get('results')
    if not result_dir:
        raise KeyError('Missing resolved results dir; run _prepare_output_paths first.')
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, filename)


def initialize_denoise_results_file(config: Dict) -> str:
    path = get_denoise_results_path(config)
    if os.path.exists(path):
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=DENOISE_RESULT_COLUMNS).writeheader()
    return path


def append_denoise_result(config: Dict, row: Dict):
    path = initialize_denoise_results_file(config)
    payload = {col: row.get(col, '') for col in DENOISE_RESULT_COLUMNS}
    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=DENOISE_RESULT_COLUMNS).writerow(payload)


# ===========================================================
# 单段信号的指标
# ===========================================================
def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cal_SNR(predict, truth) -> float:
    predict = _to_np(predict)
    truth   = _to_np(truth)
    PS = np.sum(np.square(truth))
    PN = np.sum(np.square(predict - truth))
    return float(10 * np.log10(PS / (PN + 1e-12)))


def cal_SNR_global(predict, truth) -> float:
    """SNR over the whole trial, pooling energy before taking the logarithm.

    cal_SNR is applied per channel and the results averaged, which is fine when every
    channel is contaminated but breaks down when only some are. Under channel-mode
    injection roughly half the channels are untouched and score upwards of 140 dB, so the
    average is dominated by channels the denoiser never had to repair and says nothing
    about the repair itself. Pooling first weights each sample by its actual energy.
    """
    predict = _to_np(predict)
    truth = _to_np(truth)
    signal_power = np.sum(np.square(truth))
    noise_power = np.sum(np.square(predict - truth))
    return float(10 * np.log10(signal_power / (noise_power + 1e-12)))


def cal_PRD(predict, truth) -> float:
    """波形失真百分比 (%)。"""
    predict = _to_np(predict)
    truth   = _to_np(truth)
    num = np.sqrt(np.sum((predict - truth) ** 2))
    den = np.sqrt(np.sum(truth ** 2))
    return float((num / (den + 1e-12)) * 100)


def cal_SNRI(denoised, clean, noisy) -> float:
    return cal_SNR(denoised, clean) - cal_SNR(noisy, clean)


def cal_NRR(denoised, clean, noisy) -> float:
    denoised = _to_np(denoised)
    clean    = _to_np(clean)
    noisy    = _to_np(noisy)
    p_in  = np.sum((noisy - clean) ** 2)
    p_out = np.sum((denoised - clean) ** 2)
    if p_in < 1e-12:
        return 0.0
    return float(p_out / p_in)


# ===========================================================
# 批量聚合
# ===========================================================
def _flatten_to_2d(x: np.ndarray) -> np.ndarray:
    """
    把 [N, ...] 任意结构的信号扁平到 [N_total, T]，每行一段单通道时间序列。
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def compute_denoise_metrics(
    clean: np.ndarray,
    denoised: np.ndarray,
    noisy: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    clean / denoised / noisy: shape 任意，最后一维是时间。
    返回汇总指标 dict。noisy=None 时 SNRI/NRR/SNR_in 为 NaN。
    """
    clean_f    = _flatten_to_2d(clean)
    denoised_f = _flatten_to_2d(denoised)
    noisy_f    = _flatten_to_2d(noisy) if noisy is not None else None

    n = clean_f.shape[0]
    cc_l, rrmse_l, snr_out_l, prd_l = [], [], [], []
    snr_in_l, snri_l, nrr_l = [], [], []

    for i in range(n):
        c = clean_f[i]
        d = denoised_f[i]

        cc_l.append(float(np.corrcoef(c, d)[0, 1]))
        rrmse_l.append(float(
            mean_squared_error(c, d, squared=False)
            / (mean_squared_error(c, np.zeros_like(c), squared=False) + 1e-12)
        ))
        snr_out_l.append(cal_SNR(d, c))
        prd_l.append(cal_PRD(d, c))

        if noisy_f is not None:
            ns = noisy_f[i]
            snr_in_l.append(cal_SNR(ns, c))
            snri_l.append(cal_SNRI(d, c, ns))
            nrr_l.append(cal_NRR(d, c, ns))

    global_out = cal_SNR_global(denoised, clean)
    global_in = cal_SNR_global(noisy, clean) if noisy is not None else float('nan')

    out = {
        'num_samples': int(n),
        'snr_in_global': global_in,
        'snr_out_global': global_out,
        'snri_global': global_out - global_in,
        'cc':      float(np.nanmean(cc_l)),
        'rrmse':   float(np.nanmean(rrmse_l)),
        'snr_out': float(np.nanmean(snr_out_l)),
        'prd':     float(np.nanmean(prd_l)),
    }
    if noisy_f is not None:
        out['snr_in'] = float(np.nanmean(snr_in_l))
        out['snri']   = float(np.nanmean(snri_l))
        out['nrr']    = float(np.nanmean(nrr_l))
    else:
        out['snr_in'] = float('nan')
        out['snri']   = float('nan')
        out['nrr']    = float('nan')
    return out
