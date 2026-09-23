"""DenoFlow: the velocity field and its Euler integrator.

The field network regresses the velocity of the straight path from a contaminated trial to
its reference, and denoising integrates that field forward from the observation. The network
is an encoder-decoder that interleaves spatial and temporal convolutions around a Transformer
bottleneck; the contaminated trial is carried as a second input plane through every layer and
the path position enters once, at the bottleneck.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, T, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, C)
        return x + self.pe[:, :x.size(1)]


class EEGTransformer(nn.Module):
    def __init__(self, Nc, dropout=0.5, num_layers=4, nhead=12, dim_feedforward=1024,
                 d_model=None):
        super().__init__()
        d_model = d_model or 8 * Nc
        if d_model % nhead:
            raise ValueError(f"nhead={nhead} does not divide d_model={d_model}")

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (B, T, C)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        return x


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
        self.net = AltFieldNet(Nc, Nt, dropout=dropout, num_classes=num_classes, arch=arch)
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
