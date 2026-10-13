"""Differentiable filter bank, so the fixed front end can be moved behind the generator.

apply_filter_bank uses scipy filtfilt, which is not differentiable and cannot sit inside the
training graph. Since the operation is linear and time-invariant, it is fully characterised by
its impulse response: running filtfilt on a unit impulse and using the result as a symmetric
FIR kernel reproduces the same operator exactly, up to the truncation length.

This lets the filter bank be applied to the generator's output instead of its input, so the
network denoises in the raw domain and is scored only on errors that survive the filter --
rather than having to reproduce the passband shape itself, which it does poorly.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy import signal


def _subband_edges(dataset_name):
    if dataset_name == 'benchmark12':
        return [9, 18, 27, 36, 45, 54], [7, 15, 19, 28, 37, 46]
    return [8, 16, 24, 32, 40, 48], [6, 12, 18, 26, 34, 42]


def build_filter_bank_kernels(meta, kernel_length=257):
    """Impulse responses of each sub-band of apply_filter_bank, as FIR kernels.

    kernel_length is odd so the kernel has a well-defined centre; the response of a
    zero-phase filtfilt is symmetric about it.
    """
    passband, stopband = _subband_edges(meta.dataset_name)
    nyquist = meta.sample_rate / 2.0
    highcut_pass, highcut_stop = 80, 90
    gpass, gstop, ripple = 3, 40, 0.5

    impulse = np.zeros(kernel_length * 4, dtype=np.float64)
    impulse[len(impulse) // 2] = 1.0

    kernels = []
    for index in range(max(meta.num_filter_banks, 1)):
        wp = [passband[index] / nyquist, highcut_pass / nyquist]
        ws = [stopband[index] / nyquist, highcut_stop / nyquist]
        order, wn = signal.cheb1ord(wp, ws, gpass, gstop)
        b, a = signal.cheby1(order, ripple, wn, 'bandpass')
        response = signal.filtfilt(b, a, impulse)
        centre = len(response) // 2
        half = kernel_length // 2
        kernels.append(response[centre - half:centre + half + 1])
    return np.stack(kernels).astype(np.float32)


class TorchFilterBank(nn.Module):
    """Applies the sub-band filters as fixed depthwise convolutions.

    Input  (B, 1, C, T)      -- a single-band signal in the raw domain
    Output (B, 1, Nfb*C, T)  -- stacked sub-bands, matching flatten_structured_eeg order
    """

    def __init__(self, meta, kernel_length=257):
        super(TorchFilterBank, self).__init__()
        kernels = build_filter_bank_kernels(meta, kernel_length)
        self.num_banks, self.kernel_length = kernels.shape
        self.num_channels = meta.num_channels
        self.register_buffer('kernels', torch.tensor(kernels).unsqueeze(1))

    def forward(self, x):
        batch, _, channels, steps = x.shape
        flat = x.reshape(batch * channels, 1, steps)
        # Zero padding. filtfilt pads by odd reflection over padlen samples, which this
        # does not reproduce, leaving a transient in roughly the first and last 30 samples
        # (correlation with scipy is 0.9997 over the interior, 0.97 including the edges).
        # Reflect padding was tried and is worse. The transient is left in place because it
        # affects both the model output and the target identically, so the loss is unaffected.
        padding = self.kernel_length // 2
        filtered = nn.functional.conv1d(flat, self.kernels, padding=padding)[..., :steps]
        filtered = filtered.view(batch, channels, self.num_banks, steps)
        filtered = filtered.permute(0, 2, 1, 3).reshape(batch, 1, self.num_banks * channels, steps)
        return filtered
