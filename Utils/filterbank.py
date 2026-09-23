"""Filter bank applied to both the reference and the contaminated trial.

Each sub-band is a Chebyshev type-I band-pass with 0.5 dB passband ripple and 40 dB stopband
attenuation, run forward and backward so the response is zero-phase. The k-th sub-band passes
from its own lower edge up to 80 Hz. Sub-bands are stacked along the channel axis, so a trial
of Ne electrodes and Nb sub-bands reaches the network as Nc = Ne * Nb rows.
"""
import numpy as np
from scipy import signal

# lower passband / stopband edges of each sub-band, in Hz
SUBBAND_EDGES = {
    'nakanishi2015': ([9, 18, 27, 36, 45, 54], [7, 15, 19, 28, 37, 46]),
    'benchmark': ([8, 16, 24, 32, 40, 48], [6, 12, 18, 26, 34, 42]),
}

HIGHCUT_PASS, HIGHCUT_STOP = 80, 90
GPASS, GSTOP, RIPPLE = 3, 40, 0.5


def apply_filter_bank(eeg, dataset, sample_rate, num_banks=3):
    """Split trials into sub-bands.

    Args:
        eeg: (N, Ne, T).
        dataset: key into SUBBAND_EDGES.
        num_banks: number of sub-bands; 0 returns the input unfiltered.

    Returns:
        (N, Nb, Ne, T), float32.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    n, ne, t = eeg.shape
    out = np.zeros((n, max(num_banks, 1), ne, t), dtype=np.float32)
    if not num_banks:
        out[:, 0] = eeg
        return out

    passband, stopband = SUBBAND_EDGES[dataset]
    nyquist = sample_rate / 2.0
    for index in range(num_banks):
        wp = [passband[index] / nyquist, HIGHCUT_PASS / nyquist]
        ws = [stopband[index] / nyquist, HIGHCUT_STOP / nyquist]
        order, wn = signal.cheb1ord(wp, ws, GPASS, GSTOP)
        b, a = signal.cheby1(order, RIPPLE, wn, 'bandpass')
        padlen = 3 * (max(len(b), len(a)) - 1)
        out[:, index] = signal.filtfilt(b, a, eeg, padlen=padlen)[:, :, :t]
    return out


def flatten(eeg):
    """(N, Nb, Ne, T) -> (N, Nb*Ne, T), the layout the field network reads."""
    n, nb, ne, t = eeg.shape
    return eeg.reshape(n, nb * ne, t).astype(np.float32)


def unflatten(eeg, num_banks, num_channels):
    """Inverse of `flatten`."""
    n, _, t = eeg.shape
    return eeg.reshape(n, num_banks, num_channels, t).astype(np.float32)
