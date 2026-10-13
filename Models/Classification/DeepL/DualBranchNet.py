"""Dual-branch time-frequency network used both as a discriminator and as a decoder.

The discriminator in the adversarial framework is an SSVEPNet backbone, which operates
entirely in the time domain. That is a poor match for the signal: an SSVEP class is defined
by a narrow-band response at a known stimulus frequency and its harmonics, and a
time-domain convolution has to spend capacity rediscovering that. Adding an explicit
spectral branch gives the network the representation the paradigm is defined in.

The same module serves two purposes. With a single category head it is a decoder that can be
compared against FBCCA, TRCA, EEGNet, CCNN and SSVEPNet on equal terms. With an additional
domain head it is the discriminator of the generative framework. Sharing one implementation
keeps the claim honest: the classifier we report is the discriminator we train.
"""
import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class TemporalBranch(nn.Module):
    """The existing SSVEPNet path: spatial filtering, temporal filtering, recurrent pooling."""

    def __init__(self, num_channels, num_timesteps, dropout=0.5):
        super(TemporalBranch, self).__init__()
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(1, num_channels * 2, kernel_size=(num_channels, 1), max_norm=1.0),
            nn.BatchNorm2d(num_channels * 2), nn.PReLU(), nn.Dropout(dropout))
        self.temporal = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=(1, 10),
                      stride=(1, 2), padding=(0, 4)),
            nn.BatchNorm2d(num_channels * 4), nn.PReLU(), nn.Dropout(dropout))
        self.rnn = nn.LSTM(num_channels * 4, num_channels * 4, batch_first=True, bidirectional=True)
        # Pool the recurrent output to a fixed number of segments before flattening. Keeping
        # every timestep, as SSVEPNet does, makes the first dense layer dominate the parameter
        # count; a win obtained that way would be a statement about capacity, not architecture.
        self.pool_steps = 16
        self.pool = nn.AdaptiveAvgPool1d(self.pool_steps)
        self.out_dim = num_channels * 8 * self.pool_steps

    def forward(self, x):
        h = self.temporal(self.spatial(x)).squeeze(2).permute(0, 2, 1)
        h, _ = self.rnn(h)
        return self.pool(h.permute(0, 2, 1)).flatten(1)


class SpectralBranch(nn.Module):
    """Real and imaginary parts of the trial spectrum, restricted to the stimulus band.

    Restricting to the band that can carry an SSVEP response keeps the input small and stops
    the branch from fitting broadband artifact energy, which is the very thing the generator
    is trying to remove.
    """

    def __init__(self, num_channels, num_timesteps, sample_rate=250.0,
                 low_hz=3.0, high_hz=45.0, dropout=0.5):
        super(SpectralBranch, self).__init__()
        freqs = torch.fft.rfftfreq(num_timesteps, d=1.0 / sample_rate)
        band = ((freqs >= low_hz) & (freqs <= high_hz)).nonzero().flatten()
        self.register_buffer('band_index', band)
        self.num_bins = int(band.numel())
        self.net = nn.Sequential(
            nn.Conv1d(num_channels * 2, num_channels * 4, kernel_size=5, padding=2),
            nn.LayerNorm([num_channels * 4, self.num_bins]), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(num_channels * 4, num_channels * 4, kernel_size=5, padding=2),
            nn.LayerNorm([num_channels * 4, self.num_bins]), nn.GELU(), nn.Dropout(dropout),
            # Reduce channels but keep every frequency bin: which bin carries the response is
            # exactly the information this branch exists to supply, so pooling over frequency
            # would defeat its purpose.
            nn.Conv1d(num_channels * 4, 8, kernel_size=1))
        self.out_dim = 8 * self.num_bins

    def forward(self, x):
        spectrum = torch.fft.rfft(x.squeeze(1), dim=-1)[..., self.band_index]
        feature = torch.cat([spectrum.real, spectrum.imag], dim=1)
        return self.net(feature).flatten(1)


class DualBranchNet(nn.Module):
    """Time-frequency network. `num_domain_outputs=1` adds the adversarial head."""

    def __init__(self, num_channels, num_timesteps, num_classes, sample_rate=250.0,
                 dropout=0.5, with_domain_head=False):
        super(DualBranchNet, self).__init__()
        self.temporal_branch = TemporalBranch(num_channels, num_timesteps, dropout)
        self.spectral_branch = SpectralBranch(num_channels, num_timesteps, sample_rate,
                                              dropout=dropout)
        fused = self.temporal_branch.out_dim + self.spectral_branch.out_dim
        hidden1 = max(fused // 10, 64)
        hidden2 = max(hidden1 // 5, 32)
        self.head = nn.Sequential(
            nn.Linear(fused, hidden1), nn.PReLU(),
            nn.Linear(hidden1, hidden2), nn.PReLU(), nn.Dropout(dropout))
        self.category = nn.Linear(hidden2, num_classes)
        self.domain = nn.Sequential(nn.Linear(hidden2, 1), nn.Sigmoid()) if with_domain_head else None

    def forward(self, x):
        feature = self.head(torch.cat([self.temporal_branch(x), self.spectral_branch(x)], dim=1))
        logits = self.category(feature)
        if self.domain is None:
            return logits
        return {'cls': torch.softmax(logits, dim=1), 'adv': self.domain(feature)}
