import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGDenoiseGenerator_NoSeq(nn.Module):
    """
        CNN Baseline
    """

    def __init__(self, Nc, dropout=0.5):
        super().__init__()
        self.Nc = Nc

        # ---------- Encoder ----------
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2*Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2*Nc, 4*Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4*Nc, 8*Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)),
            nn.BatchNorm2d(8*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # ---------- Bottleneck（纯 CNN） ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(8*Nc, 16*Nc, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16*Nc),
            nn.PReLU()
        )

        # ---------- Decoder ----------
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                16*Nc, 4*Nc,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1)
            ),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                8*Nc, 2*Nc,
                kernel_size=(1, 16),
                stride=(1, 2),
                padding=(0, 7),
                output_padding=(0, 1)
            ),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4*Nc, 2*Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.final_conv = nn.Conv2d(2*Nc, 1, kernel_size=(1, 1))

    @staticmethod
    def _align_time(a, b):
        """裁剪到相同时间长度"""
        Ta, Tb = a.shape[-1], b.shape[-1]
        T = min(Ta, Tb)
        return a[..., :T], b[..., :T]

    def forward(self, x):
        # x: (B, 1, Nc, T)

        # -------- Encoder --------
        x1 = self.spatial_conv(x)     # (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)      # (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)      # (B, 8Nc, 1, T/4)

        # -------- Bottleneck --------
        b = self.bottleneck(x3)       # (B, 16Nc, 1, T/4)

        # -------- Statistics（保持你原接口）--------
        feat = b.squeeze(2)           # (B, C, T')
        mean1 = feat.mean(dim=2, keepdim=True)
        variance1 = feat.var(dim=2, keepdim=True, unbiased=True)
        mean2 = feat.mean(dim=1, keepdim=True)
        variance2 = feat.var(dim=1, keepdim=True, unbiased=True)

        # -------- Decoder --------
        u1 = self.deconv1(b)
        u1, x2a = self._align_time(u1, x2)
        e1 = torch.cat([u1, x2a], dim=1)

        u2 = self.deconv2(e1)
        u2, x1a = self._align_time(u2, x1)
        e2 = torch.cat([u2, x1a], dim=1)

        u3 = self.spatial_deconv(e2)
        output = self.final_conv(u3)

        return mean1, variance1, mean2, variance2, output

class EEGDenoiseGenerator(nn.Module):
    def __init__(self, Nc,Nt, dropout=0.5):
        super(EEGDenoiseGenerator, self).__init__()
        self.Nc = Nc
        self.dropout = dropout

        # ----------- Encoder ----------- #
        # Block 1: Spatial Conv
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 2*Nc, kernel_size=(Nc,1), stride=(Nc,1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 2: Temporal Conv (1)
        self.temp_conv1 = nn.Sequential(
            nn.Conv2d(2*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7)),
            nn.BatchNorm2d(4*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 3: Temporal Conv (2)
        self.temp_conv2 = nn.Sequential(
            nn.Conv2d(4*Nc, 8*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7)),
            nn.BatchNorm2d(8*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 4: Bi-LSTM
        self.bi_lstm = nn.LSTM(
            input_size=8*Nc,
            hidden_size=8*Nc,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        # ----------- Decoder ----------- #
        # Block 5: Temporal Transpose Conv
        if Nt % 4 == 0:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,0)),
                nn.BatchNorm2d(4*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16*Nc, 4*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,1)),
                nn.BatchNorm2d(4*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 6: Temporal Transpose Conv
        if Nt % 2 == 0:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8*Nc, 2*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,0)),
                nn.BatchNorm2d(2*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8*Nc, 2*Nc, kernel_size=(1,16), stride=(1,2), padding=(0,7), output_padding=(0,1)),
                nn.BatchNorm2d(2*Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 7: Spatial Transpose Conv
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4*Nc, 2*Nc, kernel_size=(Nc,1), stride=(Nc,1)),
            nn.BatchNorm2d(2*Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 8: Final Temporal Conv
        self.final_conv = nn.Conv2d(2*Nc, 1, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        # x shape: (B, 1, Nc, T)

        # ----------- Encoding Path ----------- #
        x1 = self.spatial_conv(x)             # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)              # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)              # -> (B, 8Nc, 1, T/4)

        # Prepare for Bi-LSTM
        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        lstm_out, _ = self.bi_lstm(x3_squeezed)       # -> (B, T/4, 16Nc)
        lstm_out = self.avgpool(lstm_out)
        mean1 = torch.mean(lstm_out.squeeze(2), dim=1, keepdim=True)  # 形状变为 (B, C, 1)

        # 2. 计算方差 (无偏估计)
        variance1 = torch.var(lstm_out.squeeze(2), dim=1, keepdim=True, unbiased=False)  # 形状变为 (B, C, 1)
        
        mean2 = torch.mean(lstm_out.squeeze(2), dim=-1, keepdim=True)  # 形状变为 (B,  1)

        # 2. 计算方差 (无偏估计)
        variance2 = torch.var(lstm_out.squeeze(2), dim=-1, keepdim=True, unbiased=False)  # 形状变为 (B, C, 1)
        lstm_out = lstm_out.permute(0, 2, 1).unsqueeze(2)  # -> (B, 16Nc, 1, T/4)

        e0 = torch.cat([lstm_out, x3], dim=1)         # -> (B, 24Nc, 1, T/4)

        # ----------- Decoding Path ----------- #
        u1 = self.deconv1(e0)                         # -> (B, 4Nc, 1, T/2)
        e1 = torch.cat([u1, x2], dim=1)               # -> (B, 8Nc, 1, T/2)
        

        u2 = self.deconv2(e1)                      # -> (B, 2Nc, 1, T)
        e2 = torch.cat([u2, x1], dim=1)               # -> (B, 4Nc, 1, T)
        u3 = self.spatial_deconv(e2)                  # -> (B, 2Nc, Nc, T)
        output = self.final_conv(u3)                  # -> (B, 1, Nc, T)

        return mean1,variance1,mean2,variance2,output
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

class SpectralGate(nn.Module):
    """Adaptive spectral gate applied to the bottleneck representation.

    The filter bank in front of the network is fixed and identical for every trial;
    this is its trial-adaptive counterpart. Bottleneck features are moved to the rFFT
    domain, multiplied by a complex gain that combines a learned per-(channel, bin)
    prior with a modulation predicted from the trial itself, and transformed back.

    The residual scale is initialised to zero, so at initialisation the gate is exactly
    the identity and the model is a strict superset of the ungated one.
    """

    def __init__(self, channels, num_steps):
        super(SpectralGate, self).__init__()
        self.num_bins = num_steps // 2 + 1
        hidden = max(channels // 4, 8)
        self.gain_real = nn.Parameter(torch.ones(channels, self.num_bins))
        self.gain_imag = nn.Parameter(torch.zeros(channels, self.num_bins))
        self.context = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.PReLU(),
            nn.Linear(hidden, self.num_bins),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, T_bottleneck)
        spectrum = torch.fft.rfft(x, dim=-1)
        modulation = torch.sigmoid(self.context(x.mean(dim=-1)))          # (B, F)
        gain = torch.complex(self.gain_real, self.gain_imag)              # (C, F)
        gated = spectrum * gain.unsqueeze(0) * modulation.unsqueeze(1)
        filtered = torch.fft.irfft(gated, n=x.shape[-1], dim=-1)
        return x + self.residual_scale * filtered


class ArtifactMaskGate(nn.Module):
    """Predict where the trial is corrupted, and gate the skip path accordingly.

    The injector corrupts either a contiguous time span across all channels or a
    subset of channels across the whole trial, so corruption is localised in a way
    the plain U-Net has no way to represent: its skip connections carry detail from
    the noisy input indiscriminately, clean region or not.

    Two gates are produced from one predicted (B, 1, C, T) corruption map:
      * a temporal profile that suppresses the skip path where the trial is corrupted,
        forcing those samples to be rebuilt from the bottleneck instead of copied;
      * a per-channel reliability weight applied before the spatial convolution, which
        is the last point at which channel identity still exists (the spatial conv
        collapses it), so channel-mode corruption can be down-weighted at all.

    The predicted map is also returned so it can be supervised against the injector's
    ground-truth mask and reported as a detection score.
    """

    def __init__(self, num_channels, hidden=16):
        super(ArtifactMaskGate, self).__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(hidden),
            nn.PReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(num_channels, 1), padding=0),
            nn.BatchNorm2d(hidden),
            nn.PReLU(),
        )
        self.to_map = nn.Conv2d(hidden, num_channels, kernel_size=(1, 9), padding=(0, 4))
        self.gate_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, 1, C, T) -> logits (B, 1, C, T)
        context = self.detector(x)                       # (B, hidden, 1, T)
        logits = self.to_map(context)                    # (B, C, 1, T)
        logits = logits.permute(0, 2, 1, 3)              # (B, 1, C, T)
        probability = torch.sigmoid(logits)
        # gate_strength starts at zero: an untrained gate is exactly the identity.
        strength = torch.sigmoid(self.gate_strength) * 2.0 - 1.0
        temporal = probability.amax(dim=2, keepdim=True)             # (B, 1, 1, T)
        channel = probability.mean(dim=3, keepdim=True)               # (B, 1, C, 1)
        temporal_gate = 1.0 - strength * temporal
        channel_gate = 1.0 - strength * channel
        return logits, temporal_gate, channel_gate


class SubbandSpatialConv(nn.Module):
    """Spatial convolution that keeps the filter-bank sub-bands separable.

    flatten_structured_eeg stacks the sub-bands into the channel axis, so the default
    spatial convolution -- one kernel of height Nfb*Nc, stride Nfb*Nc -- mixes every
    sub-band and every electrode in a single stride at layer one. Whatever the filter
    bank separated is gone from that point on, which is odd for SSVEP: the whole reason
    FBCCA uses a filter bank is that the fundamental and its harmonics sit in different
    sub-bands with different SNR, and are worth weighting differently.

    Here one spatial kernel of height Nc is shared across sub-bands and applied to each
    of them, then the per-band responses are scaled by a learned weight and concatenated.
    The output shape is identical to the default layer, so nothing downstream changes.
    The band weights start at the classical FBCCA profile w(n) = n^-1.25 + 0.25.

    It is also far smaller: 2*Nc^2 weights instead of 2*(Nfb*Nc)^2, so any gain cannot
    be attributed to added capacity.
    """

    def __init__(self, num_banks, num_channels):
        super(SubbandSpatialConv, self).__init__()
        self.num_banks = num_banks
        self.num_channels = num_channels
        self.shared = nn.Conv2d(
            1, 2 * num_channels, kernel_size=(num_channels, 1), stride=(num_channels, 1)
        )
        weights = torch.tensor(
            [float(n) ** -1.25 + 0.25 for n in range(1, num_banks + 1)], dtype=torch.float32
        )
        self.band_weight = nn.Parameter(weights)

    def forward(self, x):
        # x: (B, 1, Nfb*Nc, T) with row index fb*Nc + c
        batch, _, _, steps = x.shape
        bands = x.view(batch, self.num_banks, self.num_channels, steps)
        bands = bands.reshape(batch * self.num_banks, 1, self.num_channels, steps)
        responses = self.shared(bands)                                  # (B*Nfb, 2Nc, 1, T)
        responses = responses.view(batch, self.num_banks, 2 * self.num_channels, 1, steps)
        responses = responses * self.band_weight.view(1, -1, 1, 1, 1)
        return responses.reshape(batch, self.num_banks * 2 * self.num_channels, 1, steps)


class MultiScaleTemporalConv(nn.Module):
    """Parallel temporal convolutions at several kernel lengths, concatenated.

    The single fixed length of 16 samples is shorter than one stimulus period for the lower
    half of both frequency sets: at 250 Hz an 8 Hz target has a period of 31.3 samples, so a
    16-tap kernel spans half of one oscillation and the layer never sees a complete cycle at
    full temporal resolution. Running 16, 32 and 64 taps side by side covers roughly one
    period at the top of the range, one at the bottom, and several cycles respectively.

    Padding is set to (k-2)/2 for every branch, which makes all branches produce exactly the
    output length of the original stride-2 layer, so nothing downstream changes shape.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(16, 32, 64), stride=2):
        super(MultiScaleTemporalConv, self).__init__()
        counts = [out_channels // len(kernel_sizes)] * len(kernel_sizes)
        counts[-1] += out_channels - sum(counts)
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, count, kernel_size=(1, k), stride=(1, stride),
                      padding=(0, (k - 2) // 2))
            for k, count in zip(kernel_sizes, counts)
        ])

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        steps = min(out.shape[-1] for out in outputs)
        return torch.cat([out[..., :steps] for out in outputs], dim=1)

class EEGDenoiseGeneratorv2(nn.Module):
    def __init__(self, Nc, Nt, dropout=0.5, spectral_gate=False, mask_gate=False,
                 num_filter_banks=0, multiscale_kernels=None):
        super(EEGDenoiseGeneratorv2, self).__init__()
        self.Nc = Nc
        self.dropout = dropout

        # ----------- Encoder ----------- #
        # Block 1: Spatial Conv
        # num_filter_banks > 0 swaps the all-mixing layer-one kernel for a sub-band
        # separable one; Nc here is already Nfb*Nc_raw, so the raw count is recovered.
        if num_filter_banks and num_filter_banks > 1 and Nc % num_filter_banks == 0:
            spatial_front = SubbandSpatialConv(num_filter_banks, Nc // num_filter_banks)
        else:
            spatial_front = nn.Conv2d(1, 2 * Nc, kernel_size=(Nc, 1), stride=(Nc, 1))
        self.spatial_conv = nn.Sequential(
            spatial_front,
            nn.BatchNorm2d(2 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 2: Temporal Conv (1)
        temporal1 = (MultiScaleTemporalConv(2 * Nc, 4 * Nc, tuple(multiscale_kernels))
                     if multiscale_kernels else
                     nn.Conv2d(2 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)))
        self.temp_conv1 = nn.Sequential(
            temporal1,
            nn.BatchNorm2d(4 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 3: Temporal Conv (2)
        temporal2 = (MultiScaleTemporalConv(4 * Nc, 8 * Nc, tuple(multiscale_kernels))
                     if multiscale_kernels else
                     nn.Conv2d(4 * Nc, 8 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7)))
        self.temp_conv2 = nn.Sequential(
            temporal2,
            nn.BatchNorm2d(8 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # self.transformer_layer = TransformerEncoder(
        #     TransformerEncoderLayer(
        #         d_model=8 * Nc,
        #         nhead=12,
        #         dim_feedforward=1024,
        #         dropout=dropout,
        #         batch_first=True  # 使用(B, T, C)格式
        #     ),
        #     num_layers=4
        # )
        self.transformer_layer = EEGTransformer(Nc=Nc,dropout=dropout)
        self.avgpool = nn.AvgPool1d(kernel_size=2)

        # ----------- Decoder ----------- #
        # Block 5: Temporal Transpose Conv
        if Nt % 4 == 0:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 0)),
                nn.BatchNorm2d(4 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(16 * Nc, 4 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 1)),
                nn.BatchNorm2d(4 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 6: Temporal Transpose Conv
        if Nt % 2 == 0:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 0)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(8 * Nc, 2 * Nc, kernel_size=(1, 16), stride=(1, 2), padding=(0, 7),
                                   output_padding=(0, 1)),
                nn.BatchNorm2d(2 * Nc),
                nn.PReLU(),
                nn.Dropout(dropout)
            )

        # Block 7: Spatial Transpose Conv
        self.spatial_deconv = nn.Sequential(
            nn.ConvTranspose2d(4 * Nc, 2 * Nc, kernel_size=(Nc, 1), stride=(Nc, 1)),
            nn.BatchNorm2d(2 * Nc),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        # Block 8: Final Temporal Conv
        self.final_conv = nn.Conv2d(2 * Nc, 1, kernel_size=(1, 1), stride=(1, 1))

        # Probe the encoder so the gate matches the true bottleneck length: the two
        # stride-2 stages floor it, so T=256 -> 64 but T=250 -> 62.
        self.spectral_gate = None
        if spectral_gate:
            # The probe must not run through BatchNorm in training mode, or the all-zero
            # input is folded into the running statistics before training even starts.
            encoder = nn.Sequential(self.spatial_conv, self.temp_conv1, self.temp_conv2)
            was_training = encoder.training
            encoder.eval()
            with torch.no_grad():
                steps = encoder(torch.zeros(1, 1, Nc, Nt)).shape[-1]
            encoder.train(was_training)
            self.spectral_gate = SpectralGate(8 * Nc, steps)

        self.mask_gate = ArtifactMaskGate(Nc) if mask_gate else None

    def forward(self, x):
        # x shape: (B, 1, Nc, T)

        # ----------- Encoding Path ----------- #
        self.last_mask_logits = None
        temporal_gate = None
        if self.mask_gate is not None:
            mask_logits, temporal_gate, channel_gate = self.mask_gate(x)
            self.last_mask_logits = mask_logits
            x = x * channel_gate

        x1 = self.spatial_conv(x)  # -> (B, 2Nc, 1, T)
        x2 = self.temp_conv1(x1)  # -> (B, 4Nc, 1, T/2)
        x3 = self.temp_conv2(x2)  # -> (B, 8Nc, 1, T/4)

        if temporal_gate is not None:
            # Skip tensors are already channel-collapsed, so only the temporal profile
            # can act here; resample it to each stage's own time resolution.
            x1 = x1 * F.interpolate(temporal_gate.squeeze(2), size=x1.shape[-1],
                                    mode='linear', align_corners=False).unsqueeze(2)
            x2 = x2 * F.interpolate(temporal_gate.squeeze(2), size=x2.shape[-1],
                                    mode='linear', align_corners=False).unsqueeze(2)
            x3 = x3 * F.interpolate(temporal_gate.squeeze(2), size=x3.shape[-1],
                                    mode='linear', align_corners=False).unsqueeze(2)

        x3_squeezed = x3.squeeze(2).permute(0, 2, 1)  # -> (B, T/4, 8Nc)
        transformer_out= self.transformer_layer(x3_squeezed)  # -> (B, T/4, 16Nc)
        #lstm_out = self.avgpool(lstm_out)
        transformer_out = transformer_out.permute(0, 2, 1).unsqueeze(2)  # -> (B, 16Nc, 1, T/4)

        if self.spectral_gate is not None:
            transformer_out = self.spectral_gate(transformer_out.squeeze(2)).unsqueeze(2)

        e0 = torch.cat([transformer_out, x3], dim=1)  # -> (B, 24Nc, 1, T/4)
        # expose the full feature map so alignment losses can use more than the
        # collapsed marginals below; overwritten every forward, so no accumulation
        self.last_features = transformer_out.squeeze(2)  # (B, 8Nc, T/4)
        # encoder stack at three depths, for multi-depth alignment
        self.last_multi = (x1.squeeze(2), x2.squeeze(2), x3.squeeze(2), transformer_out.squeeze(2))
        mean1 = torch.mean(transformer_out.squeeze(2), dim=1, keepdim=True)  # 形状变为 (B, C, 1)

        # 2. 计算方差 (无偏估计)
        variance1 = torch.var(transformer_out.squeeze(2), dim=1, keepdim=True, unbiased=True)  # 形状变为 (B, C, 1)
        
        mean2 = torch.mean(transformer_out.squeeze(2), dim=-1, keepdim=True)  # 形状变为 (B,  1)

        # 2. 计算方差 (无偏估计)
        variance2 = torch.var(transformer_out.squeeze(2), dim=-1, keepdim=True, unbiased=True)  # 形状变为 (B, C, 1)

        # 3. 计算对数方差 (添加小常数避免 log(0))
        log_variance1 = torch.log(variance1 + 1e-8)  # 形状保持 (B, C, 1)
        log_variance2 = torch.log(variance2 + 1e-8)  # 形状保持 (B, C, 1)
        # ----------- Decoding Path ----------- #
        u1 = self.deconv1(e0)  # -> (B, 4Nc, 1, T/2)
        e1 = torch.cat([u1, x2], dim=1)  # -> (B, 8Nc, 1, T/2)

        u2 = self.deconv2(e1)  # -> (B, 2Nc, 1, T)
        e2 = torch.cat([u2, x1], dim=1)  # -> (B, 4Nc, 1, T)
        u3 = self.spatial_deconv(e2)  # -> (B, 2Nc, Nc, T)
        output = self.final_conv(u3)  # -> (B, 1, Nc, T)

        return mean1,log_variance1,mean2,log_variance2,output
    