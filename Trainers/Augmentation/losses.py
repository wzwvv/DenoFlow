"""Extra generator objectives for DenoNet v2.

Two additions, both motivated by concrete reviewer objections:

harmonic_spectral_loss
    The paper claims DenoNet "preserves frequency-specific neural information",
    but no loss term actually enforces that. This weights the reconstruction
    error in the frequency domain, boosting the bins at each trial's stimulus
    fundamental and its harmonics. Makes the central claim mechanistic instead
    of incidental, and gives the spectral-preservation evidence reviewers asked
    for (ENG-R3#3 / R6#7, JBHI-R1#5).

info_nce_loss
    The existing latent alignment matches channel/temporal means and variances
    between the noisy and clean encoder paths. That is an aggregate constraint
    and is largely redundant with the reconstruction loss, which is why sweeping
    its weight 1 -> 100 barely moves any metric. InfoNCE instead pulls the noisy
    embedding of a trial towards the clean embedding of *that same trial* while
    pushing it away from every other trial in the batch, i.e. a trial-level
    constraint. The negatives explicitly penalise copying the input or
    collapsing onto a per-class template (ENG-R4#9/#10, R6#4).
    Operates on the same statistics the KL version used, so "KL moment matching
    vs InfoNCE" is a clean apples-to-apples swap with no architecture change.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def harmonic_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    freqs,
    sample_rate: float,
    num_harmonics: int = 5,
    bandwidth_hz: float = 0.4,
    harmonic_boost: float = 4.0,
    mode: str = "complex",
) -> torch.Tensor:
    """Harmonic-weighted spectral error.

    pred/target: (B, 1, C, T) or (B, C, T); labels: (B,) class indices.

    mode="complex" (default): error on the COMPLEX spectrum, so phase is preserved.
    mode="logmag": the original magnitude-only form. Measured to be actively harmful
        (-8.3 pp FBCCA, -7.4 pp TRCA at severe noise) precisely because it pins the
        magnitude at the harmonics while letting the phase drift, and FBCCA/TRCA are
        correlation-based and therefore phase-sensitive. Kept only for the ablation
        that documents this.
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)

    num_timesteps = target.shape[-1]
    pred_fft = torch.fft.rfft(pred, dim=-1)
    true_fft = torch.fft.rfft(target, dim=-1)

    fft_freqs = torch.fft.rfftfreq(num_timesteps, d=1.0 / sample_rate).to(target.device)
    class_freqs = torch.as_tensor(freqs, device=target.device, dtype=target.dtype)[labels]

    weights = torch.ones((labels.shape[0], fft_freqs.shape[0]), device=target.device, dtype=target.dtype)
    for harmonic_index in range(1, num_harmonics + 1):
        centers = class_freqs.unsqueeze(1) * harmonic_index
        mask = (fft_freqs.unsqueeze(0) - centers).abs() <= bandwidth_hz
        weights = weights + mask.to(target.dtype) * (harmonic_boost / harmonic_index)

    if mode == "logmag":
        # NOTE: .abs() on a complex tensor has an undefined gradient at 0; only safe
        # because it is never the default path.
        err = (torch.log1p(pred_fft.abs()) - torch.log1p(true_fft.abs())).pow(2)
    else:
        diff = pred_fft - true_fft
        # squared modulus written out so no abs() appears in the graph
        err = diff.real.pow(2) + diff.imag.pow(2)
        # scale-free: normalise by the target's spectral energy per trial
        energy = (true_fft.real.pow(2) + true_fft.imag.pow(2)).mean(dim=(-2, -1), keepdim=True)
        err = err / (energy + 1e-8)

    return (err * weights.unsqueeze(1)).mean()


def _kl_diag_gaussian(mean_a, logvar_a, mean_r, logvar_r):
    """KL between diagonal Gaussians, same form the original alignment uses."""
    var_a = torch.exp(logvar_a)
    var_r = torch.exp(logvar_r)
    return 0.5 * torch.mean(logvar_r - logvar_a + (var_a + (mean_a - mean_r).pow(2)) / var_r - 1)


def grouped_channel_alignment(feat_noisy, feat_clean, groups: int = 16):
    """Channel-axis alignment at group resolution instead of a single global mean.

    The original term averages the whole feature axis into ONE value per timestep,
    so any configuration with the right global mean satisfies it. Splitting the axis
    into `groups` blocks keeps far more of the channel structure while staying a
    cheap moment constraint. feat: (B, C, T).
    """
    b, c, t = feat_noisy.shape
    g = max(1, min(groups, c))
    while c % g:
        g -= 1
    fn = feat_noisy.reshape(b, g, c // g, t)
    fc = feat_clean.reshape(b, g, c // g, t)
    mn, vn = fn.mean(dim=2), fn.var(dim=2, unbiased=False).clamp_min(1e-8)
    mc, vc = fc.mean(dim=2), fc.var(dim=2, unbiased=False).clamp_min(1e-8)
    return _kl_diag_gaussian(mn, vn.log(), mc, vc.log())


def multi_depth_temporal_alignment(feats_noisy, feats_clean):
    """Temporal-direction KL applied at several encoder depths, not just the last one.

    The separated ablation showed the temporal direction carries the entire effect of
    the original alignment, while the channel direction is inert (and alone, worse than
    no alignment at all). This applies the direction that works at x1/x2/x3 as well as
    the transformer output, on the theory that constraining only the deepest layer
    leaves the earlier ones free to drift.
    Each feat: (B, C, T) with different C and T per depth.
    """
    total = None
    for fn, fc in zip(feats_noisy, feats_clean):
        mn = fn.mean(dim=-1, keepdim=True)
        vn = fn.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
        mc = fc.mean(dim=-1, keepdim=True)
        vc = fc.var(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-8)
        term = _kl_diag_gaussian(mn, vn.log(), mc, vc.log())
        total = term if total is None else total + term
    n = max(1, len(feats_noisy))
    return total / n if total is not None else torch.zeros(())


def coral_alignment(feat_noisy, feat_clean):
    """Align the cross-channel COVARIANCE of the encoder features.

    The existing loss matches per-axis marginal means/variances, which says nothing
    about how channels co-vary. CORAL constrains exactly that joint structure, so it
    is a genuinely different constraint rather than a reparametrisation of the same one.
    feat: (B, C, T) -> time-pooled to (B, C), covariance over the batch.
    """
    zn = feat_noisy.mean(dim=-1)
    zc = feat_clean.mean(dim=-1)

    def _cov(z):
        z = z - z.mean(dim=0, keepdim=True)
        return z.t() @ z / max(1, z.shape[0] - 1)

    d = zn.shape[1]
    return (_cov(zn) - _cov(zc)).pow(2).sum() / (4.0 * d * d)


def projected_infonce(projection, feat_noisy, feat_clean, temperature: float = 0.1,
                      labels=None, exclude_same_class: bool = False, logit_clamp: float = 30.0):
    """Trial-level InfoNCE on a learned projection of the FULL feature map.

    Distinct from the earlier attempt, which ran InfoNCE on the same collapsed
    statistics the KL used -- that changed how the representations were compared but
    not what was compared, so it never tested the diagnosis. Here the projection sees
    the time-pooled per-channel features (C values, not 1).
    """
    zn = F.normalize(projection(feat_noisy.mean(dim=-1)), dim=1, eps=1e-6)
    zc = F.normalize(projection(feat_clean.mean(dim=-1)), dim=1, eps=1e-6)
    logits = (zn @ zc.t() / max(temperature, 1e-3)).clamp(-logit_clamp, logit_clamp)
    targets = torch.arange(logits.shape[0], device=logits.device)
    if exclude_same_class and labels is not None:
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, -logit_clamp)
    return F.cross_entropy(logits, targets)


def _embed(*stats: torch.Tensor) -> torch.Tensor:
    """Flatten the encoder statistics into one embedding per trial.

    IMPORTANT: pass the log-variance statistics as well, not only the means.
    The mean statistics are averages over a BatchNorm'd feature axis, so they sit
    extremely close to zero; normalising a near-zero vector divides by ~0 and
    detonates the gradients (observed: generator NaN at ~epoch 140, which then
    poisoned the discriminator through loss_d_cls). The log-variances are O(1)
    and keep the embedding well conditioned.
    """
    return torch.cat([s.flatten(start_dim=1) for s in stats], dim=1)


def info_nce_loss(
    noisy_stats,
    clean_stats,
    temperature: float = 0.1,
    labels: torch.Tensor | None = None,
    exclude_same_class: bool = False,
    logit_clamp: float = 30.0,
) -> torch.Tensor:
    """Trial-level InfoNCE between the noisy and clean encoder statistics.

    noisy_stats / clean_stats: sequences of stat tensors (means AND log-variances).
    Positive pair = (noisy_i, clean_i). Negatives = clean_j, j != i.
    With exclude_same_class=False the negatives include same-class trials, so the
    objective demands trial-specific alignment rather than a per-class template.
    """
    z_noisy = F.normalize(_embed(*noisy_stats), dim=1, eps=1e-6)
    z_clean = F.normalize(_embed(*clean_stats), dim=1, eps=1e-6)

    logits = (z_noisy @ z_clean.t() / max(temperature, 1e-3)).clamp(-logit_clamp, logit_clamp)
    targets = torch.arange(logits.shape[0], device=logits.device)

    if exclude_same_class and labels is not None:
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, -logit_clamp)

    return F.cross_entropy(logits, targets)
