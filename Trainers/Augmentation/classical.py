"""Classical denoisers that train no network (revision baseline for R5-7).

ICA: every contaminated trial is decomposed separately in each filter-bank sub-band. The injected
artifact keeps one spatial pattern (its per-trial electrode gains) in every sub-band of a trial,
but the gains are redrawn for every trial, so a decomposition pooled over trials would not isolate
it. The components of a trial are ranked by excess kurtosis and the k most kurtotic ones are
removed before back-projection. k is the only setting. It is chosen on the training split as the
value that minimises the mean squared error to the reference trials, so no test trial affects it,
and k = 0 (no removal) is a valid outcome.
"""
import warnings

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


def _ica_variants(x, k_max, seed):
    """x: (Ne, T). Returns (k_max + 1, Ne, T): the trial with its 0..k_max most kurtotic ICs removed."""
    out = np.repeat(x[None], k_max + 1, axis=0).astype(np.float64)
    if not np.all(np.isfinite(x)) or np.allclose(x.std(axis=1), 0):
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ica = FastICA(n_components=x.shape[0], whiten="unit-variance", max_iter=1000,
                          tol=1e-4, random_state=seed)
            sources = ica.fit_transform(x.T)            # (T, Ne)
    except Exception:                                    # degenerate trial: keep it untouched
        return out
    order = np.argsort(-kurtosis(sources, axis=0, fisher=True))
    for k in range(1, k_max + 1):
        kept = sources.copy()
        kept[:, order[:k]] = 0.0
        out[k] = (kept @ ica.mixing_.T + ica.mean_).T
    return out


def _ica_all(eeg, k_max, seed):
    """eeg: (N, Nb, Ne, T) -> (k_max + 1, N, Nb, Ne, T)."""
    n, nb = eeg.shape[:2]
    out = np.empty((k_max + 1,) + eeg.shape, dtype=np.float32)
    for i in range(n):
        for b in range(nb):
            out[:, i, b] = _ica_variants(eeg[i, b], k_max, seed)
    return out


def ica_denoise(signals, method_cfg):
    """Returns (denoised train, denoised test, metrics), both (N, Nb, Ne, T)."""
    k_max = int(method_cfg.get("k_max", 3))
    seed = int(method_cfg.get("seed", 0))
    train = _ica_all(np.asarray(signals.source.train, dtype=np.float64), k_max, seed)
    clean = np.asarray(signals.clean.train, dtype=np.float64)
    errors = [float(np.mean((train[k] - clean) ** 2)) for k in range(k_max + 1)]
    k_best = int(np.argmin(errors))
    test = _ica_all(np.asarray(signals.source.test, dtype=np.float64), k_max, seed)[k_best]
    metrics = {"ica_k": float(k_best)}
    metrics.update({f"ica_train_mse_k{k}": e for k, e in enumerate(errors)})
    print(f"[ICA] k chosen on the training split = {k_best}; train MSE by k = "
          + ", ".join(f"{e:.4f}" for e in errors), flush=True)
    return train[k_best], test.astype(np.float32), metrics
