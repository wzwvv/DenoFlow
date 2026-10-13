"""
离线后处理脚本：在已经保存好的 .npy 输出上重新评估去噪指标。
指标实现统一在 Support/denoise_metrics.py，避免与主实验流程重复。
"""
import csv
import os
import sys
import time
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# 项目根目录加入 sys.path 以便 import Support.*
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Support.denoise_metrics import (
    DENOISE_RESULT_COLUMNS,
    cal_NRR,
    cal_PRD,
    cal_SNR,
    cal_SNRI,
)
from sklearn.metrics import mean_squared_error

plt.style.use(['science'])
l_db = 11


# ===========================================================
# CSV（与主流程同字段集）
# ===========================================================
def initialize_results_file(csv_path):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    if os.path.exists(csv_path):
        return csv_path
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=DENOISE_RESULT_COLUMNS).writeheader()
    return csv_path


def append_result(csv_path, row):
    initialize_results_file(csv_path)
    payload = {col: row.get(col, '') for col in DENOISE_RESULT_COLUMNS}
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=DENOISE_RESULT_COLUMNS).writerow(payload)


# ===========================================================
# IO
# ===========================================================
def read_mat(p):
    if os.path.exists(p):
        return scio.loadmat(p)


def read_npy(p):
    return np.load(p, allow_pickle=True)


def read_data(out_path, in_path):
    out = read_npy(out_path); inp = read_npy(in_path)
    o, i = out[0], inp[0]
    for k in range(1, out.shape[0]):
        o = np.concatenate((o, out[k]))
        i = np.concatenate((i, inp[k]))
    return o, i


def read_data_with_noisy(out_path, in_path, noisy_path):
    o, i = read_data(out_path, in_path)
    noisy = read_npy(noisy_path)
    n = noisy[0]
    for k in range(1, noisy.shape[0]):
        n = np.concatenate((n, noisy[k]))
    return o, i, n


# ===========================================================
# 评估
# ===========================================================
def _row_metrics(clean, denoised, noisy=None):
    cc    = float(np.corrcoef(clean, denoised)[0, 1])
    rrmse = float(
        mean_squared_error(clean, denoised, squared=False)
        / (mean_squared_error(clean, np.zeros_like(clean), squared=False) + 1e-12)
    )
    snr_out = cal_SNR(denoised, clean)
    prd     = cal_PRD(denoised, clean)
    snri = nrr = snr_in = float('nan')
    if noisy is not None:
        snr_in = cal_SNR(noisy, clean)
        snri   = cal_SNRI(denoised, clean, noisy)
        nrr    = cal_NRR(denoised, clean, noisy)
    return cc, rrmse, snr_in, snr_out, prd, snri, nrr


def cal_acc_and_mse(output_file, input_file, noisy_file=None,
                    artifact_type='unknown', model_name='unknown',
                    csv_path=None, split_kwargs=None):
    use_noisy = noisy_file is not None
    cc_l, rrmse_l, snr_in_l, snr_out_l, prd_l, snri_l, nrr_l = [], [], [], [], [], [], []

    for j in range(input_file.shape[0]):
        clean    = input_file[j, :]
        denoised = output_file[j, :]
        noisy    = noisy_file[j, :] if use_noisy else None
        cc, rrmse, snr_in, snr_out, prd, snri, nrr = _row_metrics(clean, denoised, noisy)
        cc_l.append(cc); rrmse_l.append(rrmse); snr_out_l.append(snr_out); prd_l.append(prd)
        if use_noisy:
            snr_in_l.append(snr_in); snri_l.append(snri); nrr_l.append(nrr)

    agg = {
        'cc':      float(np.nanmean(cc_l)),
        'rrmse':   float(np.nanmean(rrmse_l)),
        'snr_out': float(np.nanmean(snr_out_l)),
        'prd':     float(np.nanmean(prd_l)),
        'snr_in':  float(np.nanmean(snr_in_l)) if use_noisy else float('nan'),
        'snri':    float(np.nanmean(snri_l))   if use_noisy else float('nan'),
        'nrr':     float(np.nanmean(nrr_l))    if use_noisy else float('nan'),
    }

    print(f"\n=== [{artifact_type.upper()} | {model_name}] 全样本 ===")
    print(f"  CC={agg['cc']:.4f} RRMSE={agg['rrmse']:.4f} "
          f"SNRout={agg['snr_out']:.2f}dB PRD={agg['prd']:.2f}%")
    if use_noisy:
        print(f"  SNRin={agg['snr_in']:.2f}dB SNRI={agg['snri']:.2f}dB NRR={agg['nrr']:.4f}")

    if csv_path:
        sk = split_kwargs or {}
        append_result(csv_path, {
            'timestamp':      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'split_mode':     sk.get('split_mode', ''),
            'subject':        sk.get('subject_id', ''),
            'fold':           sk.get('fold_id', ''),
            'target_subject': sk.get('target_subject_id', ''),
            'augmenter':      model_name,
            'artifact_type':  artifact_type,
            'num_samples':    int(input_file.shape[0]),
            'cc':             f"{agg['cc']:.6f}",
            'rrmse':          f"{agg['rrmse']:.6f}",
            'snr_in':         f"{agg['snr_in']:.4f}",
            'snr_out':        f"{agg['snr_out']:.4f}",
            'prd':            f"{agg['prd']:.4f}",
            'snri':           f"{agg['snri']:.4f}",
            'nrr':            f"{agg['nrr']:.6f}",
        })
    return agg


# ===========================================================
# 绘图
# ===========================================================
def _plot(y, name, label, ylabel, path, clear=True):
    if clear:
        plt.clf()
    plt.plot(list(range(-5, 6)), np.array(y), '.-', label=label)
    plt.legend(frameon=True)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('Input SNR (dB)', fontsize=14)
    plt.savefig(f"{path}/{name}.png")


# ===========================================================
# 主入口（独立运行；如需写到主流程同一个 results 目录，
# 直接传 csv_path 指向 .../results/denoise_results.csv 即可）
# ===========================================================
if __name__ == "__main__":
    artifact_type = "EOG"
    dir_path  = f"./01_Denoisenet/CompareWorks/{artifact_type}/"
    save_path = "./Results/"
    os.makedirs(save_path, exist_ok=True)
    csv_path  = os.path.join(save_path, f"denoise_results_{artifact_type}.csv")
    initialize_results_file(csv_path)

    model_dirs = {
        'FCNN':      dir_path + f"/FCNN/FCNN_{artifact_type}_200_mse_0_0.0/",
        'SimpleCNN': dir_path + f"/SimpleCNN/SimpleCNN_{artifact_type}_200_mse_0_0.0/",
        'ResCNN':    dir_path + f"/ResCNN/ResCNN_{artifact_type}_200_mse_0_1/",
    }

    for name, mdir in model_dirs.items():
        out_p   = mdir + "best_output_data.npy"
        in_p    = mdir + "best_clean_data.npy"
        noisy_p = mdir + "best_noisy_data.npy"
        if os.path.exists(noisy_p):
            o, c, n = read_data_with_noisy(out_p, in_p, noisy_p)
        else:
            o, c = read_data(out_p, in_p); n = None
        cal_acc_and_mse(o, c, n,
                        artifact_type=artifact_type,
                        model_name=name,
                        csv_path=csv_path)

    print(f"\nCSV 已写入: {csv_path}")
