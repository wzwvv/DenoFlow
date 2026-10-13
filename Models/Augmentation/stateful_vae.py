import torch
import torch.nn as nn

from Models.Augmentation.generators import EEGTransformer


class _VariationalHead(nn.Module):
    def __init__(self, feature_channels: int, latent_dim: int, bottleneck_steps: int):
        super().__init__()
        self.mu = nn.Linear(feature_channels, latent_dim)
        self.logvar = nn.Linear(feature_channels, latent_dim)
        self.to_map = nn.Linear(latent_dim, feature_channels * bottleneck_steps)
        self.feature_channels = feature_channels
        self.bottleneck_steps = bottleneck_steps

    def forward(self, pooled_features: torch.Tensor):
        mu = self.mu(pooled_features)
        logvar = self.logvar(pooled_features)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_map = self.to_map(z).view(z.shape[0], self.feature_channels, 1, self.bottleneck_steps)
        return mu, logvar, z_map


class _BaseLatentVAEGenerator(nn.Module):
    def __init__(self, num_channels: int, num_timesteps: int, dropout: float = 0.5, latent_dim: int = 128):
        super().__init__()
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.latent_dim = latent_dim

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2 * num_channels, kernel_size=(num_channels, 1), stride=(num_channels, 1)),
            nn.BatchNorm2d(2 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2 * num_channels, 4 * num_channels, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(4 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4 * num_channels, 8 * num_channels, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(8 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_channels, num_timesteps)
            bottleneck_steps = self.temp_conv2(self.temp_conv1(self.spatial_conv(dummy))).shape[-1]
        self.bottleneck_steps = bottleneck_steps

        self.feature_channels = self._get_feature_channels(num_channels)
        self.variational_head = _VariationalHead(self.feature_channels, latent_dim, bottleneck_steps)

        deconv1_in_channels = self.feature_channels + 8 * num_channels
        deconv1_out_padding = 0 if num_timesteps % 4 == 0 else 1
        deconv2_out_padding = 0 if num_timesteps % 2 == 0 else 1

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                deconv1_in_channels,
                4 * num_channels,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, deconv1_out_padding),
            ),
            nn.BatchNorm2d(4 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                8 * num_channels,
                2 * num_channels,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, deconv2_out_padding),
            ),
            nn.BatchNorm2d(2 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * num_channels, 2 * num_channels, kernel_size=(num_channels, 1), stride=(num_channels, 1)),
            nn.BatchNorm2d(2 * num_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.final_conv = nn.Conv2d(2 * num_channels, 1, kernel_size=(1, 1), stride=(1, 1))

    def _get_feature_channels(self, num_channels: int) -> int:
        raise NotImplementedError

    def _bottleneck_map(self, x3: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _align_time(first: torch.Tensor, second: torch.Tensor):
        target_steps = min(first.shape[-1], second.shape[-1])
        return first[..., :target_steps], second[..., :target_steps]

    def forward(self, x: torch.Tensor):
        x1 = self.spatial_conv(x)
        x2 = self.temp_conv1(x1)
        x3 = self.temp_conv2(x2)

        bottleneck_map = self._bottleneck_map(x3)
        pooled = bottleneck_map.squeeze(2).mean(dim=-1)
        mu, logvar, latent_map = self.variational_head(pooled)

        e0 = torch.cat([latent_map, x3], dim=1)
        u1 = self.deconv1(e0)
        u1, x2 = self._align_time(u1, x2)
        e1 = torch.cat([u1, x2], dim=1)

        u2 = self.deconv2(e1)
        u2, x1 = self._align_time(u2, x1)
        e2 = torch.cat([u2, x1], dim=1)

        u3 = self.spatial_deconv(e2)
        output = self.final_conv(u3)
        output = output[..., :x.shape[-1]]
        return mu, logvar, mu, logvar, output


class CNNLatentVAEGenerator(_BaseLatentVAEGenerator):
    def __init__(self, num_channels: int, num_timesteps: int, dropout: float = 0.5, latent_dim: int = 128):
        self.bottleneck = None
        super().__init__(num_channels, num_timesteps, dropout=dropout, latent_dim=latent_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8 * num_channels, 16 * num_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16 * num_channels),
            nn.PReLU(),
        )

    def _get_feature_channels(self, num_channels: int) -> int:
        return 16 * num_channels

    def _bottleneck_map(self, x3: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x3)


class LSTMLatentVAEGenerator(_BaseLatentVAEGenerator):
    def __init__(self, num_channels: int, num_timesteps: int, dropout: float = 0.5, latent_dim: int = 128):
        self.bi_lstm = None
        super().__init__(num_channels, num_timesteps, dropout=dropout, latent_dim=latent_dim)
        self.bi_lstm = nn.LSTM(
            input_size=8 * num_channels,
            hidden_size=8 * num_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def _get_feature_channels(self, num_channels: int) -> int:
        return 16 * num_channels

    def _bottleneck_map(self, x3: torch.Tensor) -> torch.Tensor:
        seq = x3.squeeze(2).permute(0, 2, 1)
        seq_out, _ = self.bi_lstm(seq)
        return seq_out.permute(0, 2, 1).unsqueeze(2)


class TransformerLatentVAEGenerator(_BaseLatentVAEGenerator):
    def __init__(self, num_channels: int, num_timesteps: int, dropout: float = 0.5, latent_dim: int = 128):
        self.transformer = None
        super().__init__(num_channels, num_timesteps, dropout=dropout, latent_dim=latent_dim)
        self.transformer = EEGTransformer(Nc=num_channels, dropout=dropout)

    def _get_feature_channels(self, num_channels: int) -> int:
        return 8 * num_channels

    def _bottleneck_map(self, x3: torch.Tensor) -> torch.Tensor:
        seq = x3.squeeze(2).permute(0, 2, 1)
        seq_out = self.transformer(seq)
        return seq_out.permute(0, 2, 1).unsqueeze(2)
