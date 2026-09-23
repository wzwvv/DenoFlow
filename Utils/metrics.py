"""Signal-level fidelity of a denoised trial against its reference.

Three quantities are reported: the correlation coefficient (CC), the relative root-mean-square
error (RRMSE), and the improvement in signal-to-noise ratio (SNRI), the SNR of the denoised
trial minus that of the contaminated one. All three are computed per trial and averaged.
"""
import numpy as np
import torch


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _rows(x):
    """Collapse everything but the time axis, one row per trial."""
    x = _np(x)
    return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)


def snr(predict, truth):
    """10 log10 of reference energy over residual energy, in dB."""
    predict, truth = _np(predict), _np(truth)
    ps = np.sum(np.square(truth))
    pn = np.sum(np.square(predict - truth))
    return float(10 * np.log10(ps / (pn + 1e-12)))


def snri(denoised, clean, noisy):
    return snr(denoised, clean) - snr(noisy, clean)


def rrmse(predict, truth):
    predict, truth = _np(predict), _np(truth)
    return float(np.sqrt(np.mean((predict - truth) ** 2))
                 / (np.sqrt(np.mean(truth ** 2)) + 1e-12))


def compute_metrics(clean, denoised, noisy=None):
    """Per-trial CC, RRMSE, SNR and SNRI, averaged over trials."""
    c, d = _rows(clean), _rows(denoised)
    n = _rows(noisy) if noisy is not None else None

    cc, err, out_snr, improvement = [], [], [], []
    for i in range(c.shape[0]):
        cc.append(float(np.corrcoef(c[i], d[i])[0, 1]))
        err.append(rrmse(d[i], c[i]))
        out_snr.append(snr(d[i], c[i]))
        if n is not None:
            improvement.append(snri(d[i], c[i], n[i]))

    return {
        'num_samples': int(c.shape[0]),
        'cc': float(np.nanmean(cc)),
        'rrmse': float(np.nanmean(err)),
        'snr_out': float(np.nanmean(out_snr)),
        'snri': float(np.nanmean(improvement)) if n is not None else float('nan'),
    }
