import math

import numpy as np
import torch
import torch.nn as nn


class ReferenceSignalBank(nn.Module):
    def __init__(self, Fs, Nt, freqs, num_harmonics=5):
        super().__init__()
        self.Nf = len(freqs)
        self.num_harmonics = num_harmonics
        self.ref_channels = 2 * num_harmonics
        t = np.arange(Nt) / Fs
        ref = np.zeros((self.Nf, self.ref_channels, Nt), dtype=np.float32)
        for fi, freq in enumerate(freqs):
            for h in range(1, num_harmonics + 1):
                ref[fi, 2 * (h - 1), :] = np.sin(2 * np.pi * h * freq * t)
                ref[fi, 2 * (h - 1) + 1, :] = np.cos(2 * np.pi * h * freq * t)
        self.register_buffer('templates', torch.tensor(ref, dtype=torch.float32))


class FFTLayer(nn.Module):
    def __init__(self, sample_rate, window_sec, resolution=0.2, start_freq=8, end_freq=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.resolution = resolution
        self.start_freq = start_freq
        self.end_freq = end_freq

    def forward(self, x):
        n_fft = round(self.sample_rate / self.resolution)
        fft_start = int(round(self.start_freq / self.resolution))
        fft_end = int(round(self.end_freq / self.resolution)) + 1
        sample_point = int(self.sample_rate * self.window_sec)
        fft_result = torch.fft.fft(x, n=n_fft) / (sample_point / 2)
        real_part = torch.real(fft_result[:, :, fft_start:fft_end - 1])
        imag_part = torch.imag(fft_result[:, :, fft_start:fft_end - 1])
        return torch.cat([real_part, imag_part], dim=-1)


class ReferenceProjection(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class ReferenceCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, ref_queries, kv_sequence):
        q = self.norm_q(ref_queries)
        kv = self.norm_kv(kv_sequence)
        attn_out, _ = self.cross_attn(q, kv, kv)
        logits = self.out_proj(attn_out).squeeze(-1)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class EnhancedSSVEPformer(nn.Module):
    def __init__(
        self,
        knowledge_channels,
        num_timesteps,
        num_classes,
        sample_rate,
        start_frequency,
        frequency_step,
        raw_channels=0,
        d_model=128,
        depth=2,
        num_heads=4,
        dropout=0.5,
        num_harmonics=5,
        resolution=0.2,
        end_freq=64,
        ref_sources=('sincos',),
        raw_template_dim=0,
        trca_template_dim=0,
        tdca_template_dim=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.knowledge_channels = knowledge_channels
        self.raw_channels = raw_channels
        self.total_input_channels = knowledge_channels + raw_channels
        self.ref_sources = list(ref_sources)

        if self.total_input_channels <= 0:
            raise ValueError('EnhancedSSVEPformer requires at least one input channel.')

        freqs = [start_frequency + i * frequency_step for i in range(num_classes)]
        window_sec = num_timesteps / sample_rate
        fft_dim = 2 * round((end_freq - start_frequency) / resolution)
        token_num = self.total_input_channels * 2

        self.fft_layer = FFTLayer(sample_rate, window_sec, resolution, start_frequency, end_freq)
        self.signal_embed = nn.Sequential(
            nn.Conv1d(self.total_input_channels, token_num, kernel_size=1),
            nn.LayerNorm(fft_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.signal_proj = nn.Linear(fft_dim, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(depth)
        ])

        self.ref_cross_attns = nn.ModuleDict()
        self.ref_projections = nn.ModuleDict()

        if 'sincos' in self.ref_sources:
            self.ref_bank = ReferenceSignalBank(sample_rate, num_timesteps, freqs, num_harmonics)
            sincos_dim = 2 * num_harmonics * num_timesteps
            self.ref_projections['sincos'] = ReferenceProjection(sincos_dim, d_model)
            self.ref_cross_attns['sincos'] = ReferenceCrossAttention(d_model, num_heads, dropout)

        if 'raw_template' in self.ref_sources and raw_template_dim > 0:
            self.ref_projections['raw_template'] = ReferenceProjection(raw_template_dim, d_model)
            self.ref_cross_attns['raw_template'] = ReferenceCrossAttention(d_model, num_heads, dropout)

        if 'trca' in self.ref_sources and trca_template_dim > 0:
            self.ref_projections['trca'] = ReferenceProjection(trca_template_dim, d_model)
            self.ref_cross_attns['trca'] = ReferenceCrossAttention(d_model, num_heads, dropout)

        if 'tdca' in self.ref_sources and tdca_template_dim > 0:
            self.ref_projections['tdca'] = ReferenceProjection(tdca_template_dim, d_model)
            self.ref_cross_attns['tdca'] = ReferenceCrossAttention(d_model, num_heads, dropout)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 4, num_classes),
        )

        num_ref_heads = len([source for source in self.ref_sources if source in self.ref_cross_attns])
        self.gate_logits = nn.Parameter(torch.zeros(1 + num_ref_heads)) if num_ref_heads > 0 else None
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_reference_templates(self, source_name, templates):
        buf_name = f'ref_templates_{source_name}'
        if isinstance(templates, np.ndarray):
            templates = torch.tensor(templates, dtype=torch.float32)
        self.register_buffer(buf_name, templates)

    def _build_spectral_tokens(self, knowledge_input, raw_input):
        if knowledge_input.ndim != 3:
            raise ValueError(f'Expected knowledge_input to have shape (B, C, T), got {tuple(knowledge_input.shape)}')

        batch_size = knowledge_input.shape[0]
        combined_inputs = [knowledge_input]

        if self.raw_channels > 0:
            if raw_input is None:
                raise ValueError('raw_input is required when raw_channels > 0.')
            if raw_input.ndim == 4:
                raw_flat = raw_input.reshape(batch_size, -1, raw_input.shape[-1])
            elif raw_input.ndim == 3:
                raw_flat = raw_input
            else:
                raise ValueError(f'Expected raw_input to have shape (B, C, T) or (B, F, C, T), got {tuple(raw_input.shape)}')
            if raw_flat.shape[-1] != knowledge_input.shape[-1]:
                raise ValueError(
                    f'Expected raw_input and knowledge_input to share the same time dimension, '
                    f'got {raw_flat.shape[-1]} and {knowledge_input.shape[-1]}'
                )
            combined_inputs.append(raw_flat)

        spectral_input = torch.cat(combined_inputs, dim=1)
        if spectral_input.shape[1] != self.total_input_channels:
            raise ValueError(
                f'Expected concatenated spectral input to have {self.total_input_channels} channels, '
                f'got {spectral_input.shape[1]}'
            )

        fft_feat = self.fft_layer(spectral_input)
        signal_tokens = self.signal_embed(fft_feat)
        signal_tokens = self.signal_proj(signal_tokens)
        return signal_tokens

    def forward(self, knowledge_input, raw_input=None):
        batch_size = knowledge_input.shape[0]
        spectral_tokens = self._build_spectral_tokens(knowledge_input, raw_input)
        tokens = [self.cls_token.expand(batch_size, -1, -1), spectral_tokens]

        x = torch.cat(tokens, dim=1)
        for block in self.transformer:
            x = block(x)

        cls_logits = self.cls_head(x[:, 0])
        active_sources = [source for source in self.ref_sources if source in self.ref_cross_attns]
        if not active_sources:
            return cls_logits

        ref_logits_list = []
        for source in active_sources:
            if source == 'sincos':
                ref_flat = self.ref_bank.templates.flatten(1)
            else:
                buf_name = f'ref_templates_{source}'
                ref_flat = getattr(self, buf_name)

            ref_embedded = self.ref_projections[source](ref_flat)
            ref_embedded = ref_embedded.unsqueeze(0).expand(batch_size, -1, -1)
            ref_logit = self.ref_cross_attns[source](ref_embedded, x)
            ref_logits_list.append(ref_logit)

        gate = torch.softmax(self.gate_logits, dim=0)
        final_logits = gate[0] * cls_logits
        for index, ref_logit in enumerate(ref_logits_list):
            final_logits = final_logits + gate[index + 1] * ref_logit
        return final_logits
