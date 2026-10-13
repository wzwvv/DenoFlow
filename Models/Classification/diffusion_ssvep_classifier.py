import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def _valid_nhead(d_model, max_head=8):
    nhead = max(1, min(max_head, d_model // 16))
    while d_model % nhead != 0:
        nhead -= 1
    return max(1, nhead)


class NoiseScheduler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]
        while sqrt_ab.dim() < x0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ReferenceSignalBank(nn.Module):
    def __init__(self, Fs, Nt, freqs, num_harmonics=5):
        super().__init__()
        self.Nf = len(freqs)
        self.num_harmonics = num_harmonics
        self.ref_channels = 2 * num_harmonics

        t = np.arange(Nt) / Fs
        ref = np.zeros((self.Nf, self.ref_channels, Nt), dtype=np.float32)
        for fi, freq in enumerate(freqs):
            for harmonic in range(1, num_harmonics + 1):
                ref[fi, 2 * (harmonic - 1), :] = np.sin(2 * np.pi * harmonic * freq * t)
                ref[fi, 2 * (harmonic - 1) + 1, :] = np.cos(2 * np.pi * harmonic * freq * t)

        self.register_buffer('templates', torch.tensor(ref, dtype=torch.float32))


class DiffusionSSVEPClassifier(nn.Module):
    def __init__(
        self,
        C_eeg,
        Nt,
        Fs,
        freqs,
        Nf,
        T_diff=200,
        beta_start=1e-4,
        beta_end=0.02,
        num_harmonics=5,
        dropout=0.5,
        use_input_norm=True,
        use_proto_fusion=False,
        logit_mix=0.5,
        score_temperature=8.0,
        timestep_sampling='uniform',
        timestep_power=2.0,
        classifier_mode='attn',
    ):
        super().__init__()
        self.Nf = Nf
        self.C_eeg = C_eeg
        self.Nt = Nt
        self.T_diff = T_diff
        self.use_input_norm = use_input_norm
        self.use_proto_fusion = use_proto_fusion
        self.logit_mix = logit_mix
        self.timestep_sampling = timestep_sampling
        self.timestep_power = timestep_power
        self.classifier_mode = classifier_mode

        d_model = 8 * C_eeg
        self.d_model = d_model
        nhead = _valid_nhead(d_model)

        self.scheduler = NoiseScheduler(T=T_diff, beta_start=beta_start, beta_end=beta_end)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.ref_bank = ReferenceSignalBank(Fs=Fs, Nt=Nt, freqs=freqs, num_harmonics=num_harmonics)
        ref_input_dim = 2 * num_harmonics * Nt
        self.ref_proj = nn.Sequential(
            nn.Linear(ref_input_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ref_norm = nn.LayerNorm(d_model)

        self.raw_cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(C_eeg * Nt),
            nn.Linear(C_eeg * Nt, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, Nf),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2 * C_eeg, kernel_size=(C_eeg, 1), stride=(C_eeg, 1)),
            nn.GroupNorm(_get_num_groups(2 * C_eeg), 2 * C_eeg),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2 * C_eeg, 4 * C_eeg, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.GroupNorm(_get_num_groups(4 * C_eeg), 4 * C_eeg),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4 * C_eeg, d_model, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.GroupNorm(_get_num_groups(d_model), d_model),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(256, d_model * 2),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.eeg_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(score_temperature))))

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                d_model * 2,
                4 * C_eeg,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1 if Nt % 4 else 0),
            ),
            nn.GroupNorm(_get_num_groups(4 * C_eeg), 4 * C_eeg),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                8 * C_eeg,
                2 * C_eeg,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1 if Nt % 2 else 0),
            ),
            nn.GroupNorm(_get_num_groups(2 * C_eeg), 2 * C_eeg),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * C_eeg, 2 * C_eeg, kernel_size=(C_eeg, 1), stride=(C_eeg, 1)),
            nn.GroupNorm(_get_num_groups(2 * C_eeg), 2 * C_eeg),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.final_conv = nn.Conv2d(2 * C_eeg, 1, kernel_size=(1, 1))

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def normalize_input(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std

    def sample_timesteps(self, batch_size, device):
        if self.timestep_sampling == 'low_noise':
            noise = torch.rand(batch_size, device=device)
            scaled = noise.pow(self.timestep_power)
            return (scaled * (self.T_diff - 1)).long()
        return torch.randint(0, self.T_diff, (batch_size,), device=device, dtype=torch.long)

    def build_reference_embeddings(self):
        refs_flat = self.ref_bank.templates.reshape(self.Nf, -1)
        return self.ref_norm(self.ref_proj(refs_flat))

    def predict_x0(self, x_t, t, noise_pred):
        sqrt_ab = self.scheduler.sqrt_alpha_bar[t]
        sqrt_1_ab = self.scheduler.sqrt_one_minus_alpha_bar[t]
        while sqrt_ab.dim() < x_t.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)
        return (x_t - sqrt_1_ab * noise_pred) / sqrt_ab.clamp_min(1e-6)

    def _combine_logits(self, raw_logits, attn_logits, proto_logits):
        if self.classifier_mode == 'raw':
            return raw_logits
        if self.classifier_mode == 'raw_proto':
            return raw_logits + self.logit_mix * proto_logits
        if self.classifier_mode == 'hybrid':
            return raw_logits + attn_logits + self.logit_mix * proto_logits
        if self.classifier_mode == 'attn_proto':
            return attn_logits + self.logit_mix * proto_logits
        return attn_logits

    def forward(self, x_t, t, return_aux=False):
        batch_size = x_t.shape[0]
        raw_logits = self.raw_cls_head(x_t)
        t_emb = self.time_mlp(t)

        e1 = self.spatial_conv(x_t)
        e2 = self.temp_conv1(e1)
        e3 = self.temp_conv2(e2)

        seq = e3.squeeze(2).permute(0, 2, 1)
        seq = seq + t_emb.unsqueeze(1)
        trans_out = self.transformer(seq)

        ref_bank = self.build_reference_embeddings()
        ref_emb = ref_bank.unsqueeze(0).expand(batch_size, -1, -1)

        cross_out, _ = self.cross_attn(ref_emb, trans_out, trans_out)
        cross_out = self.cross_norm(ref_emb + cross_out)
        attn_logits = self.cls_head(cross_out).squeeze(-1)

        pooled = trans_out.mean(dim=1)
        eeg_token = self.eeg_head(pooled)
        proto_logits = torch.matmul(
            F.normalize(eeg_token, dim=-1),
            F.normalize(ref_bank, dim=-1).transpose(0, 1),
        )
        proto_logits = proto_logits * self.logit_scale.exp().clamp(1.0, 100.0)

        logits = self._combine_logits(raw_logits, attn_logits, proto_logits)

        trans_4d = trans_out.permute(0, 2, 1).unsqueeze(2)
        u1 = self.deconv1(torch.cat([trans_4d, e3], dim=1))
        u2 = self.deconv2(torch.cat([u1, e2], dim=1))
        u3 = self.spatial_deconv(torch.cat([u2, e1], dim=1))
        noise_pred = self.final_conv(u3)

        if not return_aux:
            return logits, noise_pred

        return logits, noise_pred, {
            'raw_logits': raw_logits,
            'attn_logits': attn_logits,
            'proto_logits': proto_logits,
            'pooled_features': pooled,
            'reference_embeddings': ref_bank,
        }

    def training_step(self, x, labels):
        x0 = self.normalize_input(x) if self.use_input_norm else x
        t = self.sample_timesteps(x.shape[0], x.device)
        x_t, noise = self.scheduler.q_sample(x0, t)
        logits, noise_pred, aux = self.forward(x_t, t, return_aux=True)
        aux.update({
            'logits': logits,
            'noise_pred': noise_pred,
            'noise': noise,
            'x_t': x_t,
            'x0': x0,
            't': t,
            'labels': labels,
        })
        return aux

    @torch.no_grad()
    def classify(self, x):
        self.eval()
        x0 = self.normalize_input(x) if self.use_input_norm else x
        t = torch.zeros(x0.shape[0], device=x0.device, dtype=torch.long)
        logits, _ = self.forward(x0, t)
        predictions = logits.argmax(dim=1)
        return predictions, logits
