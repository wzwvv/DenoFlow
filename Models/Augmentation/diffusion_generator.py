import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Augmentation.generators import EEGTransformer


class NoiseScheduler:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
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

    def predict_x0(self, x_t, t, predicted_noise):
        alpha_bar_t = self.alpha_bar[t]
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        return (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition):
        t_tensor = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor, condition)
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        sqrt_recip_alpha = self.sqrt_recip_alpha[t]
        beta = self.betas[t]
        mean = sqrt_recip_alpha * (x_t - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition):
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, condition)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, condition, inference_steps=50, eta=0.0):
        step_size = max(1, self.T // max(inference_steps, 1))
        timesteps = list(reversed(range(0, self.T, step_size)))
        x = torch.randn(shape, device=self.device)

        for index, t in enumerate(timesteps):
            t_prev = timesteps[index + 1] if index + 1 < len(timesteps) else 0
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t_tensor, condition)

            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma ** 2) * predicted_noise
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            if float(sigma) > 0.0 and t_prev > 0:
                x = x + sigma * torch.randn_like(x)

        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ReferenceSignalBank(nn.Module):
    def __init__(self, sample_rate, num_timesteps, freqs, num_harmonics=5):
        super().__init__()
        time_axis = np.arange(num_timesteps, dtype=np.float32) / float(sample_rate)
        templates = np.zeros((len(freqs), 2 * num_harmonics, num_timesteps), dtype=np.float32)
        for class_index, freq in enumerate(freqs):
            for harmonic in range(1, num_harmonics + 1):
                phase = 2 * np.pi * harmonic * float(freq) * time_axis
                templates[class_index, 2 * (harmonic - 1), :] = np.sin(phase)
                templates[class_index, 2 * (harmonic - 1) + 1, :] = np.cos(phase)
        self.register_buffer('templates', torch.tensor(templates, dtype=torch.float32))


class KnowledgeConditionEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        num_signal_channels,
        num_timesteps,
        sample_rate,
        freqs,
        condition_dim,
        num_harmonics=5,
        dropout=0.1,
        use_reference_condition=True,
        use_class_mean_condition=True,
    ):
        super().__init__()
        self.use_reference_condition = use_reference_condition
        self.use_class_mean_condition = use_class_mean_condition
        self.label_embedding = nn.Embedding(num_classes, condition_dim)
        self.reference_bank = ReferenceSignalBank(sample_rate, num_timesteps, freqs, num_harmonics)
        self.reference_projection = nn.Sequential(
            nn.Linear(2 * num_harmonics * num_timesteps, condition_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.mean_projection = nn.Sequential(
            nn.Linear(num_signal_channels * num_timesteps, condition_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )
        self.register_buffer(
            'class_mean_bank',
            torch.zeros(num_classes, num_signal_channels, num_timesteps, dtype=torch.float32),
        )

    def set_class_mean_bank(self, class_mean_bank):
        if class_mean_bank.dim() == 4 and class_mean_bank.shape[1] == 1:
            class_mean_bank = class_mean_bank.squeeze(1)
        if class_mean_bank.dim() != 3:
            raise ValueError('class_mean_bank must have shape (num_classes, channels, timesteps).')
        if tuple(class_mean_bank.shape) != tuple(self.class_mean_bank.shape):
            raise ValueError(
                f'class_mean_bank shape mismatch: expected {tuple(self.class_mean_bank.shape)}, got {tuple(class_mean_bank.shape)}'
            )
        self.class_mean_bank.copy_(class_mean_bank.to(self.class_mean_bank.device, dtype=self.class_mean_bank.dtype))

    def forward(self, labels):
        condition = self.label_embedding(labels)
        if self.use_reference_condition:
            reference = self.reference_bank.templates[labels].flatten(1)
            condition = condition + self.reference_projection(reference)
        if self.use_class_mean_condition:
            class_means = self.class_mean_bank[labels].flatten(1)
            condition = condition + self.mean_projection(class_means)
        return self.fusion(condition)


class EEGDiffusionDenoiser(nn.Module):
    def __init__(
        self,
        num_signal_channels,
        num_timesteps,
        num_classes,
        sample_rate,
        freqs,
        num_harmonics=5,
        dropout=0.5,
        use_reference_condition=True,
        use_class_mean_condition=True,
    ):
        super().__init__()
        self.num_signal_channels = num_signal_channels
        condition_dim = 8 * num_signal_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(condition_dim),
            nn.Linear(condition_dim, condition_dim * 2),
            nn.GELU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )
        self.condition_encoder = KnowledgeConditionEncoder(
            num_classes=num_classes,
            num_signal_channels=num_signal_channels,
            num_timesteps=num_timesteps,
            sample_rate=sample_rate,
            freqs=freqs,
            condition_dim=condition_dim,
            num_harmonics=num_harmonics,
            dropout=dropout,
            use_reference_condition=use_reference_condition,
            use_class_mean_condition=use_class_mean_condition,
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2 * num_signal_channels, kernel_size=(num_signal_channels, 1), stride=(num_signal_channels, 1)),
            nn.BatchNorm2d(2 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2 * num_signal_channels, 4 * num_signal_channels, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(4 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4 * num_signal_channels, 8 * num_signal_channels, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(8 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.transformer_layer = EEGTransformer(Nc=num_signal_channels, dropout=dropout)

        deconv1_output_padding = (0, 0) if num_timesteps % 4 == 0 else (0, 1)
        deconv2_output_padding = (0, 0) if num_timesteps % 2 == 0 else (0, 1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                16 * num_signal_channels,
                4 * num_signal_channels,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=deconv1_output_padding,
            ),
            nn.BatchNorm2d(4 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                8 * num_signal_channels,
                2 * num_signal_channels,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=deconv2_output_padding,
            ),
            nn.BatchNorm2d(2 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * num_signal_channels, 2 * num_signal_channels, kernel_size=(num_signal_channels, 1), stride=(num_signal_channels, 1)),
            nn.BatchNorm2d(2 * num_signal_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.final_conv = nn.Conv2d(2 * num_signal_channels, 1, kernel_size=(1, 1), stride=(1, 1))

    def set_class_mean_bank(self, class_mean_bank):
        self.condition_encoder.set_class_mean_bank(class_mean_bank)

    def forward(self, x_t, t, labels):
        time_condition = self.time_mlp(t)
        knowledge_condition = self.condition_encoder(labels)
        fused_condition = time_condition + knowledge_condition

        x1 = self.spatial_conv(x_t)
        x2 = self.temp_conv1(x1)
        x3 = self.temp_conv2(x2)

        bottleneck = x3.squeeze(2).permute(0, 2, 1)
        bottleneck = bottleneck + fused_condition.unsqueeze(1)
        transformer_out = self.transformer_layer(bottleneck)
        transformer_out = transformer_out.permute(0, 2, 1).unsqueeze(2)
        fused_bottleneck = torch.cat([transformer_out, x3], dim=1)

        upsampled_1 = self.deconv1(fused_bottleneck)
        skip_1 = torch.cat([upsampled_1, x2], dim=1)
        upsampled_2 = self.deconv2(skip_1)
        skip_2 = torch.cat([upsampled_2, x1], dim=1)
        upsampled_3 = self.spatial_deconv(skip_2)
        return self.final_conv(upsampled_3)


class DiffusionGeneratorWrapper(nn.Module):
    def __init__(
        self,
        num_signal_channels,
        num_timesteps,
        num_classes,
        sample_rate,
        freqs,
        T=200,
        beta_start=1e-4,
        beta_end=0.02,
        inference_steps=50,
        num_harmonics=5,
        dropout=0.5,
        use_reference_condition=True,
        use_class_mean_condition=True,
        use_ddim=True,
    ):
        super().__init__()
        self.num_signal_channels = num_signal_channels
        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.denoiser = EEGDiffusionDenoiser(
            num_signal_channels=num_signal_channels,
            num_timesteps=num_timesteps,
            num_classes=num_classes,
            sample_rate=sample_rate,
            freqs=freqs,
            num_harmonics=num_harmonics,
            dropout=dropout,
            use_reference_condition=use_reference_condition,
            use_class_mean_condition=use_class_mean_condition,
        )
        self.scheduler = NoiseScheduler(T=T, beta_start=beta_start, beta_end=beta_end)
        self.inference_steps = inference_steps
        self.use_ddim = use_ddim

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def set_class_mean_bank(self, class_mean_bank):
        self.denoiser.set_class_mean_bank(class_mean_bank)

    def _format_output(self, output):
        feat = output.squeeze(1)
        mean1 = feat.mean(dim=2, keepdim=True)
        variance1 = feat.var(dim=2, keepdim=True, unbiased=True)
        log_var1 = torch.log(variance1 + 1e-8)
        mean2 = feat.mean(dim=1, keepdim=True)
        variance2 = feat.var(dim=1, keepdim=True, unbiased=True)
        log_var2 = torch.log(variance2 + 1e-8)
        return mean1, log_var1, mean2, log_var2, output

    @torch.no_grad()
    def sample(self, labels):
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.scheduler.device)
        shape = (labels.shape[0], 1, self.num_signal_channels, self.num_timesteps)
        self.denoiser.eval()
        if self.use_ddim:
            output = self.scheduler.ddim_sample(self.denoiser, shape, labels, inference_steps=self.inference_steps)
        else:
            output = self.scheduler.p_sample_loop(self.denoiser, shape, labels)
        return self._format_output(output)

    def forward(self, labels):
        return self.sample(labels)
