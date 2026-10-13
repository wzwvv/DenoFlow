"""EEGDfus (Huang et al., IEEE JBHI 2024): conditional DDPM for single-channel EEG denoising.

Ported from the official implementation, https://github.com/XYH0118/EEGDfus (commit a19a652,
files denoising_model_eegdnet.py, DDPM.py, utils.py, config/base.yaml). Changes for this repository:
  * the Transformer width d_model, hard-coded to the 512-sample EEGdenoiseNet segment, becomes the
    segment length T of the filter-bank rows (250 on Benchmark, 256 on Nakanishi2015);
  * each filter-bank row is denoised as an independent single-channel segment with shared weights,
    like the other single-channel baselines, and is scaled by the standard deviation of the
    contaminated row (the only quantity available at test time) and rescaled afterwards;
  * the epoch count is set by the caller; the step-decay point keeps the original 1500/4000 ratio.
Everything else follows the official code: dual-branch network with FiLM bridges (64 features,
three encoder layers per branch), 500-step linear schedule from 1e-4 to 0.02, L1 noise-prediction
loss with sum reduction and the continuous noise level of the official p_losses, Adam at 1e-3,
gradient clipping at 1, and 500-step ancestral sampling conditioned on the contaminated segment.
"""
from functools import partial
from math import log as ln

import numpy as np
import torch
import torch.nn as nn

D_K = D_V = 64   # attention dimensions of the official code
N_HEADS = 1


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding.unsqueeze(-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_Q = nn.Linear(d_model, D_K * N_HEADS, bias=False)
        self.W_K = nn.Linear(d_model, D_K * N_HEADS, bias=False)
        self.W_V = nn.Linear(d_model, D_V * N_HEADS, bias=False)
        self.fc = nn.Linear(N_HEADS * D_V, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        b = x.size(0)
        q = self.W_Q(x).view(b, -1, N_HEADS, D_K).transpose(1, 2)
        k = self.W_K(x).view(b, -1, N_HEADS, D_K).transpose(1, 2)
        v = self.W_V(x).view(b, -1, N_HEADS, D_V).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(D_K), dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, N_HEADS * D_V)
        return self.ln(self.fc(context) + x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.ReLU(),
                                nn.Linear(d_ff, d_model, bias=False))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.ln(self.fc(x) + x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, x):
        return self.pos_ffn(self.enc_self_attn(x))


class FiLM(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.fc_gamma = nn.Linear(condition_dim, input_dim)
        self.fc_beta = nn.Linear(condition_dim, input_dim)

    def forward(self, x, condition):
        return self.fc_gamma(condition) * x + self.fc_beta(condition)


class DualBranchDenoisingModel(nn.Module):
    def __init__(self, seq_len, feats=64):
        super().__init__()
        d_model = d_ff = seq_len
        conv = lambda: nn.Sequential(Conv1d(1, feats, 3, padding=1), Conv1d(feats, feats, 3, padding=1))
        self.stream_x = nn.ModuleList([conv()] + [EncoderLayer(d_model, d_ff) for _ in range(3)])
        self.stream_cond = nn.ModuleList([conv()] + [EncoderLayer(d_model, d_ff) for _ in range(3)])
        self.embed = PositionalEncoding(feats)
        self.bridge = nn.ModuleList([FiLM(d_model, 1) for _ in range(4)])
        self.conv_out = nn.Sequential(Conv1d(feats, feats, 3, padding=1), Conv1d(feats, 1, 3, padding=1))

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))
        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x
        return self.conv_out(cond)


class EEGDfus(nn.Module):
    """Conditional DDPM of the official DDPM.py (linear schedule only)."""

    def __init__(self, seq_len, feats=64, num_steps=500, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = DualBranchDenoisingModel(seq_len, feats)
        self.num_steps = num_steps
        self.loss_func = nn.L1Loss(reduction="sum")
        betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
        alphas_cumprod = np.cumprod(1.0 - betas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))
        t = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("sqrt_recip_alphas_cumprod", t(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", t(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_log_variance_clipped", t(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2",
                             t((1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)))

    def p_losses(self, clean, noisy):
        b = clean.shape[0]
        step = np.random.randint(1, self.num_steps + 1)
        level = torch.tensor(np.random.uniform(self.sqrt_alphas_cumprod_prev[step - 1],
                                               self.sqrt_alphas_cumprod_prev[step], size=b),
                             dtype=torch.float32, device=clean.device).view(b, 1)
        noise = torch.randn_like(clean)
        x_noisy = level.view(-1, 1, 1) * clean + (1 - level.view(-1, 1, 1) ** 2).sqrt() * noise
        return self.loss_func(noise, self.model(x_noisy, noisy, level))

    @torch.no_grad()
    def denoise(self, noisy):
        b = noisy.shape[0]
        x = torch.randn_like(noisy)
        for i in reversed(range(self.num_steps)):
            level = torch.full((b, 1), float(self.sqrt_alphas_cumprod_prev[i + 1]), device=noisy.device)
            x_recon = self.sqrt_recip_alphas_cumprod[i] * x - self.sqrt_recipm1_alphas_cumprod[i] * self.model(x, noisy, level)
            mean = self.posterior_mean_coef1[i] * x_recon + self.posterior_mean_coef2[i] * x
            if i > 0:
                x = mean + torch.randn_like(x) * (0.5 * self.posterior_log_variance_clipped[i]).exp()
            else:
                x = mean
        return x
