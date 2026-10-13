"""Generative denoisers conditioned on the contaminated observation.

The diffusion path already in the repository generates a trial from a class label and
never sees the signal it is supposed to clean, so it cannot function as a denoiser. The
models here are conditioned on the contaminated trial throughout, which is what makes the
comparison against the adversarial generator meaningful.

All of them reuse the encoder-decoder of EEGDenoiseGeneratorv2 unchanged apart from the
first convolution, which takes two input planes (the current iterate and the conditioning
observation) instead of one. Holding the backbone fixed is deliberate: whatever difference
appears is attributable to the generative formulation rather than to capacity or wiring.
"""
import math

import torch
import torch.nn as nn

from Models.Augmentation.generators import EEGTransformer


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], self.dim - emb.shape[-1], device=t.device)], dim=-1)
        return self.mlp(emb)


class CrossSkip(nn.Module):
    """Fuse a decoder feature map with the encoder map of the same scale by attention.

    Query is the decoder side, key and value the encoder side, both read as sequences over
    time. Output keeps the decoder's channel count, so the following deconvolution sees half
    the channels that concatenation would give it.
    """

    def __init__(self, channels, nhead, dropout):
        super().__init__()
        # The skip widths follow the channel multiplier, so the narrowest one is 2*Nc and is
        # not always divisible by the requested head count -- 54 on Benchmark against 48 on
        # Nakanishi2015. Fall back to the largest head count that does divide it.
        while channels % nhead:
            nhead -= 1
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(channels, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, decoded, encoded):
        q = decoded.squeeze(2).permute(0, 2, 1)
        kv = encoded.squeeze(2).permute(0, 2, 1)
        fused, _ = self.attn(q, kv, kv)
        fused = self.norm(q + fused)
        return fused.permute(0, 2, 1).unsqueeze(2)


def _apply_sequence_block(block, x):
    """Run a sequence module over a (B, C, 1, T) feature map."""
    seq = x.squeeze(2).permute(0, 2, 1)
    return block(seq).permute(0, 2, 1).unsqueeze(2)


class ConditionalFieldNet(nn.Module):
    """Backbone shared by the flow and diffusion variants.

    Input is the pair (iterate, condition) stacked as two planes; output has the shape of a
    single trial. Time enters at the bottleneck, added to the Transformer input, which is
    where the existing generator already carries its global context.

    The encoder collapses the Nc rows to one, halves the time axis `temporal_layers` times,
    and the decoder mirrors it with a skip at every scale.
    """

    def __init__(self, Nc, Nt, dropout=0.5, num_classes=0, arch=None):
        super(ConditionalFieldNet, self).__init__()
        arch = arch or {}
        w = int(arch.get('width', 4))
        kt = int(arch.get('temporal_kernel', 16))
        pt = (kt - 2) // 2          # stride 2 with this padding halves the time axis
        n_sp = int(arch.get('spatial_layers', 1))
        n_tp = int(arch.get('temporal_layers', 2))
        bands = int(arch.get('bands', 3))
        c1 = w * Nc
        chans = [c1 * (2 ** i) for i in range(n_tp + 1)]     # per scale, index 0 = full rate

        self.Nc = Nc
        self.n_tp = n_tp

        def block(mod, out_ch):
            return nn.Sequential(mod, nn.BatchNorm2d(out_ch), nn.PReLU(), nn.Dropout(dropout))

        # ---- encoder -------------------------------------------------------
        if n_sp == 1:
            self.spatial = block(nn.Conv2d(2, c1, (Nc, 1), (Nc, 1)), c1)
        else:
            if Nc % bands:
                raise ValueError(f"spatial_layers=2 needs bands={bands} to divide Nc={Nc}")
            mid, rows = max(c1 // 2, 1), Nc // bands
            self.spatial = nn.Sequential(
                block(nn.Conv2d(2, mid, (bands, 1), (bands, 1)), mid),
                block(nn.Conv2d(mid, c1, (rows, 1), (rows, 1)), c1))
        self.temporal = nn.ModuleList([
            block(nn.Conv2d(chans[i], chans[i + 1], (1, kt), (1, 2), (0, pt)), chans[i + 1])
            for i in range(n_tp)])

        # time axis length at each scale, needed to size the transposed convolutions
        lengths = [Nt]
        for _ in range(n_tp):
            lengths.append((lengths[-1] + 2 * pt - kt) // 2 + 1)
        self.lengths = lengths

        # ---- bottleneck ----------------------------------------------------
        nhead = int(arch.get('nhead', 12))
        dim_ff = int(arch.get('dim_feedforward', 2048))
        self.transformer_layer = EEGTransformer(
            Nc=Nc, dropout=dropout, d_model=chans[-1],
            num_layers=int(arch.get('num_layers', 4)),
            nhead=nhead, dim_feedforward=dim_ff)

        # Attention one scale above the bottleneck, on both sides so the encoder and the
        # decoder operate on the same set of scales.
        scale2 = int(arch.get('scale2_layers', 0))
        if scale2 and n_tp >= 2:
            self.enc_attn2 = EEGTransformer(Nc=Nc, dropout=dropout, d_model=chans[-2],
                                            num_layers=scale2, nhead=nhead,
                                            dim_feedforward=dim_ff)
            self.dec_attn2 = EEGTransformer(Nc=Nc, dropout=dropout, d_model=chans[-2],
                                            num_layers=scale2, nhead=nhead,
                                            dim_feedforward=dim_ff)
        else:
            self.enc_attn2 = self.dec_attn2 = None

        self.skip_mode = str(arch.get('skip_mode', 'concat'))
        if self.skip_mode == 'cross':
            self.skips = nn.ModuleList([CrossSkip(c, nhead, dropout) for c in chans])
        else:
            self.skips = None
        fan = 1 if self.skip_mode == 'cross' else 2

        self.time_embedding = TimeEmbedding(chans[-1])
        self.label_embedding = nn.Embedding(num_classes, chans[-1]) if num_classes else None

        # Path embedding at every scale, off by default. Each site gets its own linear from the
        # shared embedding to that scale's channel count, added as a per-channel bias. Zero-
        # initialised so 'multi' starts identical to 'bottleneck' and learns the injection.
        self.time_mode = str(arch.get('time_mode', 'bottleneck'))
        if self.time_mode == 'multi':
            emb = chans[-1]
            self.time_spatial = nn.Linear(emb, c1)
            self.time_enc = nn.ModuleList([nn.Linear(emb, chans[i + 1]) for i in range(n_tp)])
            self.time_dec = nn.ModuleList([nn.Linear(emb, chans[n_tp - j - 1]) for j in range(n_tp)])
            self.time_spatial_up = nn.Linear(emb, c1)
            for m in (self.time_spatial, self.time_spatial_up, *self.time_enc, *self.time_dec):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # ---- decoder -------------------------------------------------------
        # stage j lifts scale n_tp-j to scale n_tp-j-1; output_padding makes the transposed
        # convolution land on exactly the length the encoder had there
        ups = []
        for j in range(n_tp):
            lo, hi = n_tp - j - 1, n_tp - j
            op = lengths[lo] - 2 * lengths[hi]
            ups.append(block(nn.ConvTranspose2d(fan * chans[hi], chans[lo], (1, kt), (1, 2),
                                                (0, pt), output_padding=(0, op)), chans[lo]))
        self.up = nn.ModuleList(ups)

        if n_sp == 1:
            self.spatial_up = block(nn.ConvTranspose2d(fan * c1, c1, (Nc, 1), (Nc, 1)), c1)
        else:
            mid, rows = max(c1 // 2, 1), Nc // bands
            self.spatial_up = nn.Sequential(
                block(nn.ConvTranspose2d(fan * c1, mid, (rows, 1), (rows, 1)), mid),
                block(nn.ConvTranspose2d(mid, c1, (bands, 1), (bands, 1)), c1))
        self.final_conv = nn.Conv2d(c1, 1, (1, 1), (1, 1))

    def _fuse(self, decoded, encoded, level):
        if self.skips is not None:
            return self.skips[level](decoded, encoded)
        return torch.cat([decoded, encoded], dim=1)

    def forward(self, iterate, condition, t, labels=None):
        x = torch.cat([iterate, condition], dim=1)          # (B, 2, Nc, T)
        te = self.time_embedding(t)                         # shared path embedding
        multi = self.time_mode == 'multi'
        bias = lambda proj: proj(te).view(te.shape[0], -1, 1, 1)
        h = self.spatial(x)
        if multi:
            h = h + bias(self.time_spatial)
        skips = [h]
        for i, layer in enumerate(self.temporal):
            h = layer(h)
            if multi:
                h = h + bias(self.time_enc[i])
            if self.enc_attn2 is not None and i == self.n_tp - 2:
                h = _apply_sequence_block(self.enc_attn2, h)
                skips.append(h)
                continue
            skips.append(h)

        seq = h.squeeze(2).permute(0, 2, 1)
        cond = te
        if self.label_embedding is not None and labels is not None:
            cond = cond + self.label_embedding(labels)
        seq = seq + cond.unsqueeze(1)
        h = self.transformer_layer(seq).permute(0, 2, 1).unsqueeze(2)

        for j, layer in enumerate(self.up):
            level = self.n_tp - j
            h = layer(self._fuse(h, skips[level], level))
            if multi:
                h = h + bias(self.time_dec[j])
            if self.dec_attn2 is not None and j == 0 and self.n_tp >= 2:
                h = _apply_sequence_block(self.dec_attn2, h)
        h = self.spatial_up(self._fuse(h, skips[0], 0))
        if multi:
            h = h + bias(self.time_spatial_up)
        return self.final_conv(h)


class FlowDenoiser(nn.Module):
    """Rectified flow along the straight path from contaminated to reference trial.

    Training draws t ~ U(0,1), forms x_t = (1-t)*noisy + t*clean and regresses the constant
    velocity clean - noisy. Sampling integrates that field forward from the observation with
    Euler steps, so one step is an ordinary residual predictor and additional steps refine it;
    the step count is therefore a knob that trades compute against reconstruction, not a
    property fixed at training time.
    """

    def __init__(self, Nc, Nt, dropout=0.5, num_classes=0, steps=4, arch=None):
        super(FlowDenoiser, self).__init__()
        net_cls = AltFieldNet if str((arch or {}).get('conv_order', '')) == 'alt'             else ConditionalFieldNet
        self.net = net_cls(Nc, Nt, dropout=dropout, num_classes=num_classes, arch=arch)
        self.steps = steps

    def velocity_target(self, clean, noisy, labels=None):
        t = torch.rand(clean.shape[0], device=clean.device)
        shape = (-1,) + (1,) * (clean.dim() - 1)
        x_t = (1.0 - t.view(shape)) * noisy + t.view(shape) * clean
        return self.net(x_t, noisy, t, labels), clean - noisy

    def rollout(self, noisy, labels=None, steps=None):
        """Euler integration that keeps the graph, for losses defined on the sampled output.

        Separate from sample() because that one is decorated with no_grad for inference:
        calling it during training silently detaches every term computed from its output.
        """
        steps = steps or self.steps
        x = noisy
        for index in range(steps):
            t = torch.full((noisy.shape[0],), index / steps, device=noisy.device)
            x = x + self.net(x, noisy, t, labels) / steps
        return x

    @torch.no_grad()
    def sample(self, noisy, labels=None, steps=None):
        return self.rollout(noisy, labels, steps)

    def forward(self, noisy, labels=None):
        return self.sample(noisy, labels)


class AltFieldNet(nn.Module):
    """Interleaved spatial/temporal field network: S1, T1, S2, T2 and its mirror.

    Same input/output contract as ConditionalFieldNet -- forward(iterate, condition, t, labels)
    returns a (B, 1, Nc, T) velocity -- so FlowDenoiser can swap one for the other. The spatial
    collapse is split into a bands stage (Nc -> Nc/bands) and a rows stage (Nc/bands -> 1), with
    a temporal downsample between and after, so a temporal filter sees the electrode axis before
    it is gone. The decoder mirrors with a concat skip at every stage; the length bookkeeping is
    the base network's, so kt = 2*pt + 2 makes each transposed convolution land on the encoder's
    length exactly.
    """

    def __init__(self, Nc, Nt, dropout=0.5, num_classes=0, arch=None):
        super().__init__()
        arch = arch or {}
        w = int(arch.get('width', 4))
        kt = int(arch.get('temporal_kernel', 16))
        pt = (kt - 2) // 2
        bands = int(arch.get('bands', 3))
        if Nc % bands:
            raise ValueError(f"conv_order=alt needs bands={bands} to divide Nc={Nc}")
        rows = Nc // bands
        c1 = w * Nc
        c2, c4 = 2 * c1, 4 * c1
        self.Nc = Nc

        def block(mod, out_ch):
            return nn.Sequential(mod, nn.BatchNorm2d(out_ch), nn.PReLU(), nn.Dropout(dropout))

        self.s1 = block(nn.Conv2d(2, c1, (bands, 1), (bands, 1)), c1)
        self.t1 = block(nn.Conv2d(c1, c2, (1, kt), (1, 2), (0, pt)), c2)
        self.s2 = block(nn.Conv2d(c2, c2, (rows, 1), (rows, 1)), c2)
        self.t2 = block(nn.Conv2d(c2, c4, (1, kt), (1, 2), (0, pt)), c4)

        L0 = Nt
        L1 = (L0 + 2 * pt - kt) // 2 + 1
        L2 = (L1 + 2 * pt - kt) // 2 + 1

        nhead = int(arch.get('nhead', 12))
        self.transformer_layer = EEGTransformer(
            Nc=Nc, dropout=dropout, d_model=c4,
            num_layers=int(arch.get('num_layers', 4)), nhead=nhead,
            dim_feedforward=int(arch.get('dim_feedforward', 2048)))
        self.time_embedding = TimeEmbedding(c4)
        self.label_embedding = nn.Embedding(num_classes, c4) if num_classes else None

        self.t2_up = block(nn.ConvTranspose2d(2 * c4, c2, (1, kt), (1, 2), (0, pt),
                                              output_padding=(0, L1 - 2 * L2)), c2)
        self.s2_up = block(nn.ConvTranspose2d(2 * c2, c2, (rows, 1), (rows, 1)), c2)
        self.t1_up = block(nn.ConvTranspose2d(2 * c2, c1, (1, kt), (1, 2), (0, pt),
                                              output_padding=(0, L0 - 2 * L1)), c1)
        self.s1_up = block(nn.ConvTranspose2d(2 * c1, c1, (bands, 1), (bands, 1)), c1)
        self.final_conv = nn.Conv2d(c1, 1, (1, 1), (1, 1))

    def forward(self, iterate, condition, t, labels=None):
        x = torch.cat([iterate, condition], dim=1)
        e1 = self.s1(x)
        e2 = self.t1(e1)
        e3 = self.s2(e2)
        e4 = self.t2(e3)
        seq = e4.squeeze(2).permute(0, 2, 1)
        cond = self.time_embedding(t)
        if self.label_embedding is not None and labels is not None:
            cond = cond + self.label_embedding(labels)
        seq = seq + cond.unsqueeze(1)
        b = self.transformer_layer(seq).permute(0, 2, 1).unsqueeze(2)
        d = self.t2_up(torch.cat([b, e4], dim=1))
        d = self.s2_up(torch.cat([d, e3], dim=1))
        d = self.t1_up(torch.cat([d, e2], dim=1))
        d = self.s1_up(torch.cat([d, e1], dim=1))
        return self.final_conv(d)


class DiffusionDenoiser(nn.Module):
    """Conditional DDPM in x0-prediction form, conditioned on the contaminated trial.

    x0-prediction rather than epsilon-prediction because the reconstruction loss the rest of
    the framework reports is defined on the signal, and predicting the signal directly keeps
    the training objective on the same footing as the deterministic baseline.
    """

    def __init__(self, Nc, Nt, dropout=0.5, num_classes=0, num_steps=200,
                 infer_steps=10, artifact_forward=False):
        super(DiffusionDenoiser, self).__init__()
        self.net = ConditionalFieldNet(Nc, Nt, dropout=dropout, num_classes=num_classes)
        self.num_steps = num_steps
        self.infer_steps = infer_steps
        betas = torch.linspace(1e-4, 0.02, num_steps)
        self.register_buffer('alphas_bar', torch.cumprod(1.0 - betas, dim=0))
        # Standard DDPM corrupts with Gaussian noise, so the reverse process learns to
        # invert Gaussian noising. The contamination here is recorded EMG and EOG, which
        # is neither Gaussian nor stationary, so that is a mismatch between what the model
        # is trained to undo and what it must actually undo. With artifact_forward the
        # forward process instead uses the artifact that actually contaminated the trial,
        # which is available without extra bookkeeping as noisy - clean.
        self.artifact_forward = artifact_forward

    def _corruption(self, clean, noisy):
        """Unit-RMS corruption direction for the forward process.

        Normalised per trial so the noise schedule keeps its usual meaning: without this the
        magnitude of the artifact, which is set by the injection SNR, would silently rescale
        every alpha in the schedule.
        """
        if not self.artifact_forward:
            return torch.randn_like(clean)
        residual = noisy - clean
        rms = residual.flatten(1).pow(2).mean(dim=1).sqrt()
        rms = rms.view((-1,) + (1,) * (clean.dim() - 1)).clamp_min(1e-8)
        return residual / rms

    def predict_x0(self, clean, noisy, labels=None):
        step = torch.randint(0, self.num_steps, (clean.shape[0],), device=clean.device)
        a_bar = self.alphas_bar[step].view((-1,) + (1,) * (clean.dim() - 1))
        x_t = a_bar.sqrt() * clean + (1.0 - a_bar).sqrt() * self._corruption(clean, noisy)
        return self.net(x_t, noisy, step.float() / self.num_steps, labels)

    def rollout(self, noisy, labels=None):
        """Single differentiable x0 estimate at t=0, used for losses on the sampled output.

        Backpropagating through the full reverse chain would cost one activation set per
        step for no benefit here: the terms that need a gradient act on the final estimate.
        """
        zero = torch.zeros(noisy.shape[0], device=noisy.device)
        return self.net(noisy, noisy, zero, labels)

    @torch.no_grad()
    def sample(self, noisy, labels=None):
        # Start from the observation rather than pure Gaussian: the observation is already a
        # perturbed version of the target, so seeding with it keeps the trajectory short.
        x = noisy
        grid = torch.linspace(self.num_steps - 1, 0, self.infer_steps).long().to(noisy.device)
        for step in grid:
            batch_step = step.repeat(noisy.shape[0]).float() / self.num_steps
            x0 = self.net(x, noisy, batch_step, labels)
            a_bar = self.alphas_bar[step]
            # At sampling time the true artifact is unknown; the observation's own residual
            # against the current estimate is the closest available stand-in.
            direction = self._corruption(x0, noisy) if self.artifact_forward else torch.randn_like(x0)
            x = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * direction
        return self.net(x, noisy, torch.zeros(noisy.shape[0], device=noisy.device), labels)

    def forward(self, noisy, labels=None):
        return self.sample(noisy, labels)
