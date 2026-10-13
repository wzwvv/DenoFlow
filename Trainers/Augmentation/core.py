import os
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Data.core import AugmentationOutput, ExperimentSignals, SignalSet, flatten_structured_eeg
from Data.signal_ops import apply_direct_augmentation, is_direct_augmentation_method
from Models.Augmentation import discriminators as Discriminator
from Models.Augmentation.torch_filterbank import TorchFilterBank
from Models.Augmentation.CT_DCENet.models import CT_DCENet
from Models.Augmentation.EEGDNet.Network_structure import DeT
from Models.Augmentation.GCTNet.models import DuoCL, GCTNet, basemodel
from Models.Augmentation.generators import EEGDenoiseGenerator, EEGDenoiseGenerator_NoSeq, EEGDenoiseGeneratorv2
from Models.Augmentation.stateful_vae import CNNLatentVAEGenerator
from Support.config import (
    get_augmentation_method_config,
    get_default_batch_size,
    get_experiment_config,
    get_output_paths,
)
from Trainers.Augmentation import diffusion as DiffusionTrainer
from Trainers.Augmentation.generative import (
    build_generative_model,
    is_generative_method,
    run_generative_inference,
    train_generative_denoiser,
)
from Trainers.Augmentation import legacy as LegacyTrainer
from Trainers.Augmentation.losses import (
    coral_alignment,
    grouped_channel_alignment,
    multi_depth_temporal_alignment,
    harmonic_spectral_loss,
    info_nce_loss,
    projected_infonce,
)
from Support.utils import finish_progress_line, print_progress_line, require_gradient


# TOED-GAN (Zeng et al., J. Neural Eng. 21:066003) shares this repository's CNN-LSTM
# backbone and two-head discriminator, so it is reproduced by fixing that generator to the
# published objective rather than by adding a parallel implementation. The weights are pinned
# here instead of left to a config file: a baseline that can be mis-specified by editing YAML
# is not a faithful reproduction of anyone's method.
#   L_G = L_ca + 0.3 * L_adv        (no reconstruction term)
#   L_D = L_feat + 0.5 * L_adv
TOEDGAN_SPEC = {
    'backbone': 'CNN-LSTM_GAN',
    'reconstruction_weight': 0.0,
    'classification_weight': 1.0,
    'adversarial_weight': 0.3,
    'd_adversarial_weight': 0.5,
    'kl_weight': 0.0,
    'epochs': 150,
    'generator_lr': 1e-3,
    'discriminator_lr': 1e-3,
}


def resolve_baseline_spec(method: str, method_cfg: Dict) -> Tuple[str, Dict]:
    """Map a published-baseline name to (internal method, config with pinned weights)."""
    if str(method).upper() != 'TOEDGAN':
        return method, method_cfg
    merged = dict(method_cfg)
    merged.update({k: v for k, v in TOEDGAN_SPEC.items() if k != 'backbone'})
    return TOEDGAN_SPEC['backbone'], merged


STATEFUL_METHOD_ALIASES = {
    'CNN-BASE': 'CNN-BASE',
    'CNN-GAN': 'CNN-GAN',
    'CNN-VAE': 'CNN-VAE',
    'CNN-GAN-VAE': 'CNN-GAN-VAE',
    'CNN-LSTM': 'CNN-LSTM_BASE',
    'CNN-LSTM_BASE': 'CNN-LSTM_BASE',
    'CNN-LSTM_GAN': 'CNN-LSTM_GAN',
    'CNN-LSTM_VAE': 'CNN-LSTM_VAE',
    'CNN-LSTM_GAN-VAE': 'CNN-LSTM_GAN-VAE',
    'CNN-TRANS': 'CNN-TRANS_BASE',
    'CNN-TRANS_BASE': 'CNN-TRANS_BASE',
    'CNN-TRANS_GAN': 'CNN-TRANS_GAN',
    'CNN-TRANS_VAE': 'CNN-TRANS_VAE',
    'CNN-TRANS_GAN-VAE': 'CNN-TRANS_GAN-VAE',
    'CNN-Trans': 'CNN-TRANS_BASE',
    'CNN-Trans_BASE': 'CNN-TRANS_BASE',
    'CNN-Trans_GAN': 'CNN-TRANS_GAN',
    'CNN-Trans_VAE': 'CNN-TRANS_VAE',
    'CNN-Trans_GAN-VAE': 'CNN-TRANS_GAN-VAE',
    'GAN': 'CNN-TRANS_GAN-VAE',
}
STATEFUL_BACKBONES = ('CNN-LSTM', 'CNN-TRANS', 'CNN')
STATEFUL_MODES = {'BASE', 'GAN', 'VAE', 'GAN-VAE'}
DIFFUSION_FAMILY = {'Diffusion', 'DiffusionGAN'}
GCT_FAMILY = {'GCTNet', 'SimpleCNN', 'FCNN', 'ResCNN', 'GeneratorCNN', 'GeneratorTransformer', 'NovelCNN', 'DuoCL'}
ONE_DIMENSIONAL_INPUT_METHODS = {'GCTNet', 'SimpleCNN', 'ResCNN', 'GeneratorCNN', 'GeneratorTransformer', 'NovelCNN', 'DuoCL'}
FLAT_INPUT_METHODS = {'FCNN'}


def normalize_stateful_method_name(method: str) -> str:
    if method in STATEFUL_METHOD_ALIASES:
        return STATEFUL_METHOD_ALIASES[method]
    normalized = method.replace('CNN-Trans', 'CNN-TRANS').upper()
    return STATEFUL_METHOD_ALIASES.get(normalized, normalized)


def parse_stateful_generator_method(method: str) -> Tuple[str, str]:
    normalized = normalize_stateful_method_name(method)
    for backbone in STATEFUL_BACKBONES:
        prefix = f'{backbone}_'
        if normalized.startswith(prefix):
            mode = normalized[len(prefix):]
            if mode in STATEFUL_MODES:
                return backbone, mode
            
    raise ValueError(f'Unsupported stateful augmentation method: {method}')


def is_stateful_generator_method(method: str) -> bool:
    try:
        parse_stateful_generator_method(method)
        return True
    except ValueError:
        return False


def build_stateful_generator(method: str, flat_channels: int, num_timesteps: int, method_cfg: Dict,
                            num_filter_banks: int = 0):
    backbone, mode = parse_stateful_generator_method(method)
    dropout = float(method_cfg.get('dropout', 0.5))
    latent_dim = int(method_cfg.get('latent_dim', 128))
    requires_vae = 'VAE' in mode

    if requires_vae and backbone == 'CNN':
        return CNNLatentVAEGenerator(flat_channels, num_timesteps, dropout=dropout, latent_dim=latent_dim)
    if backbone == 'CNN':
        return EEGDenoiseGenerator_NoSeq(flat_channels, dropout=dropout)
    if backbone == 'CNN-LSTM':
        return EEGDenoiseGenerator(flat_channels, num_timesteps, dropout=dropout)
    return EEGDenoiseGeneratorv2(
        flat_channels,
        num_timesteps,
        dropout=dropout,
        spectral_gate=bool(method_cfg.get('spectral_gate', False)),
        mask_gate=bool(method_cfg.get('mask_gate', False)),
        num_filter_banks=num_filter_banks if method_cfg.get('subband_spatial', False) else 0,
        multiscale_kernels=method_cfg.get('multiscale_kernels'),
    )


def _standard_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _to_logvar(backbone: str, variance_like: torch.Tensor) -> torch.Tensor:
    if backbone == 'CNN-LSTM':
        return torch.log(variance_like.clamp_min(1e-8))
    return variance_like


def _kl_pair(mean1, logvar1, ref_mean1, ref_logvar1):
    """KL for a single alignment direction (channel-wise OR temporal-wise)."""
    var1 = torch.exp(logvar1)
    ref_var1 = torch.exp(ref_logvar1)
    return 0.5 * torch.mean(ref_logvar1 - logvar1 + (var1 + (mean1 - ref_mean1).pow(2)) / ref_var1 - 1)


def _reference_kl_per_trial(
    backbone,
    mean1, varlike1, mean2, varlike2,
    ref_mean1, ref_varlike1, ref_mean2, ref_varlike2,
):
    """Same divergence as _reference_kl_divergence but reduced per trial, not globally.

    Needed to gate the term by reference quality: the batch mean cannot express "align this
    trial but not that one", and scaling the pooled scalar only changes the step size.
    """
    def flat_mean(t):
        return t.reshape(t.shape[0], -1).mean(dim=1)

    logvar1 = _to_logvar(backbone, varlike1)
    logvar2 = _to_logvar(backbone, varlike2)
    ref_logvar1 = _to_logvar(backbone, ref_varlike1)
    ref_logvar2 = _to_logvar(backbone, ref_varlike2)

    term1 = ref_logvar1 - logvar1 + (torch.exp(logvar1) + (mean1 - ref_mean1).pow(2)) / torch.exp(ref_logvar1) - 1
    term2 = ref_logvar2 - logvar2 + (torch.exp(logvar2) + (mean2 - ref_mean2).pow(2)) / torch.exp(ref_logvar2) - 1
    return 0.5 * flat_mean(term1) + 0.5 * flat_mean(term2)


def _reference_kl_divergence(
    backbone: str,
    mean1: torch.Tensor,
    varlike1: torch.Tensor,
    mean2: torch.Tensor,
    varlike2: torch.Tensor,
    ref_mean1: torch.Tensor,
    ref_varlike1: torch.Tensor,
    ref_mean2: torch.Tensor,
    ref_varlike2: torch.Tensor,
) -> torch.Tensor:
    logvar1 = _to_logvar(backbone, varlike1)
    logvar2 = _to_logvar(backbone, varlike2)
    ref_logvar1 = _to_logvar(backbone, ref_varlike1)
    ref_logvar2 = _to_logvar(backbone, ref_varlike2)

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    ref_var1 = torch.exp(ref_logvar1)
    ref_var2 = torch.exp(ref_logvar2)

    kl_loss1 = 0.5 * torch.mean(ref_logvar1 - logvar1 + (var1 + (mean1 - ref_mean1).pow(2)) / ref_var1 - 1)
    kl_loss2 = 0.5 * torch.mean(ref_logvar2 - logvar2 + (var2 + (mean2 - ref_mean2).pow(2)) / ref_var2 - 1)
    return kl_loss1 + kl_loss2


def train_stateful_generator(
    train_loader,
    test_loader,
    device: str,
    generator,
    method: str,
    flat_channels: int,
    num_timesteps: int,
    num_classes: int,
    method_cfg: Dict,
    meta=None,
):
    backbone, mode = parse_stateful_generator_method(method)
    use_adv = 'GAN' in mode
    use_kl = 'VAE' in mode
    use_latent_kl = (use_kl and backbone == 'CNN')
    use_reference_kl = (use_kl and backbone in {'CNN-LSTM', 'CNN-TRANS'})

    generator_lr = float(method_cfg.get('generator_lr', 1e-3))
    discriminator_lr = float(method_cfg.get('discriminator_lr', 1e-3))
    generator_weight_decay = float(method_cfg.get('generator_weight_decay', 1e-4))
    discriminator_weight_decay = float(method_cfg.get('discriminator_weight_decay', 1e-4))
    epochs = int(method_cfg.get('epochs', 300))
    eval_interval = int(method_cfg.get('eval_interval', 10))
    reconstruction_weight = float(method_cfg.get('reconstruction_weight', 1.0))
    classification_weight = float(method_cfg.get('classification_weight', 1.0))
    adversarial_weight = float(method_cfg.get('adversarial_weight', 1.0))
    kl_weight = float(method_cfg.get('kl_weight', 0.1))
    reconstruction_loss = str(method_cfg.get('reconstruction_loss', 'l1')).lower()
    use_scheduler = bool(method_cfg.get('use_scheduler', True))
    # --- DenoNet v2 additions (default 0.0 -> disabled, fully backward compatible) ---
    spectral_weight = float(method_cfg.get('spectral_weight', 0.0))
    spectral_num_harmonics = int(method_cfg.get('spectral_num_harmonics', 5))
    spectral_bandwidth_hz = float(method_cfg.get('spectral_bandwidth_hz', 0.4))
    spectral_harmonic_boost = float(method_cfg.get('spectral_harmonic_boost', 4.0))
    spectral_mode = str(method_cfg.get('spectral_mode', 'complex'))
    contrastive_weight = float(method_cfg.get('contrastive_weight', 0.0))
    contrastive_temperature = float(method_cfg.get('contrastive_temperature', 0.1))
    contrastive_exclude_same_class = bool(method_cfg.get('contrastive_exclude_same_class', False))
    grad_clip = float(method_cfg.get('grad_clip', 0.0))
    # TOED-GAN (Zeng 2024) uses different adversarial weights for D (lambda_D=0.5)
    # and G (lambda_G=0.3). Falls back to adversarial_weight, so existing configs are unchanged.
    d_adversarial_weight = float(method_cfg.get('d_adversarial_weight', adversarial_weight))
    # Supervision for the artifact-mask detector head. 0 keeps the gate but trains it
    # end-to-end from the main objective only, which isolates architecture from supervision.
    mask_weight = float(method_cfg.get('mask_weight', 0.0))
    # Per-trial confidence weighting. The reference is a real recording, not a clean
    # signal, and some references are poor exemplars of their own class. The category
    # head's probability of the true class on the reference is a cheap per-trial estimate
    # of how much that reference deserves to be imitated; terms that pull the output
    # towards it (reconstruction, adversarial) are scaled by it.
    reference_confidence = float(method_cfg.get('reference_confidence', 0.0))
    confidence_floor = float(method_cfg.get('confidence_floor', 0.25))
    confidence_warmup = int(method_cfg.get('confidence_warmup', 30))
    la_mode = str(method_cfg.get('la_mode', 'moment')).lower()
    la_groups = int(method_cfg.get('la_groups', 16))
    la_proj_dim = int(method_cfg.get('la_proj_dim', 128))
    la_temperature = float(method_cfg.get('la_temperature', 0.1))
    # Gate the alignment term by reference quality. The term pulls the contaminated
    # encoding towards the reference encoding, so when the reference trial is itself a
    # poor exemplar of its class that target is the wrong thing to move towards. This is
    # a different claim from weighting reconstruction: a reference the discriminator
    # misreads is still a valid waveform to reconstruct, just not a representation worth
    # aligning to.
    #   'hard' : zero the term on trials whose reference is misclassified
    #   'soft' : weight each trial by its reference's probability of the true class
    la_gate_mode = str(method_cfg.get('la_gate_mode', 'off')).lower()
    la_gate_warmup = int(method_cfg.get('la_gate_warmup', 30))
    # Move the filter bank behind the generator: it denoises in the raw domain and the
    # fixed bank is applied to its output before the loss, so the network is scored only
    # on error that survives the filter instead of also having to reproduce the passband
    # shape. Requires unfiltered_source, otherwise the input is already band-split.
    post_filter = bool(method_cfg.get('post_filter', False))
    filter_bank_layer = TorchFilterBank(meta).to(device) if (post_filter and meta is not None) else None

    generator = generator.to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(flat_channels, num_timesteps, num_classes)
    ).to(device)

    la_projection = None
    if la_mode == 'proj':
        # EEGTransformer uses d_model = 8*Nc; the inline comment in generators.py saying 16Nc is wrong
        la_feature_dim = 8 * flat_channels
        la_projection = nn.Sequential(
            nn.Linear(la_feature_dim, la_proj_dim),
            nn.PReLU(),
            nn.Linear(la_proj_dim, la_proj_dim),
        ).to(device)
    generator_params = list(generator.parameters()) + (
        list(la_projection.parameters()) if la_projection is not None else [])
    optimizer_g = torch.optim.Adam(generator_params, lr=generator_lr, weight_decay=generator_weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, weight_decay=discriminator_weight_decay)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=max(1, epochs * max(1, len(train_loader))), eta_min=5e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=max(1, epochs * max(1, len(train_loader))), eta_min=5e-6)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_recon = F.l1_loss if reconstruction_loss == 'l1' else F.mse_loss
    safe_log = lambda tensor: torch.log(tensor + 1e-8)

    latest_fake_acc = 0.0
    latest_real_acc = 0.0
    nonfinite_batches = 0
    best_fake_acc = -1.0
    best_real_acc = 0.0
    best_state = deepcopy(generator.state_dict())

    for epoch_index in range(epochs):
        generator.train()
        discriminator.train()
        for batch in train_loader:
            clean, source, labels = batch[0], batch[1], batch[2]
            true_mask = batch[3].to(device) if len(batch) > 3 else None
            clean = clean.float().to(device)
            source = source.float().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

            src_stat1, src_stat2, src_stat3, src_stat4, generated = generator(source)
            if filter_bank_layer is not None:
                generated = filter_bank_layer(generated[:, :, :meta.num_channels, :])
            d_real = discriminator(clean.detach())
            d_fake = discriminator(generated.detach())

            reference_probability = d_real['cls'][range(len(labels)), labels]
            loss_d_cls = -safe_log(reference_probability).mean() - safe_log(d_fake['cls'][range(len(labels)), labels]).mean()
            loss_d_adv = -safe_log(d_real['adv']).mean() - safe_log(1 - d_fake['adv']).mean() if use_adv else torch.zeros((), device=device)
            loss_d = d_adversarial_weight * loss_d_adv
            if classification_weight > 0:
                # added conditionally: 0 * inf = nan, and loss_d_cls can reach inf
                # once its head stops receiving gradient
                loss_d = loss_d + classification_weight * loss_d_cls

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()
            if use_scheduler:
                scheduler_d.step()

            src_stat1, src_stat2, src_stat3, src_stat4, generated = generator(source)
            if filter_bank_layer is not None:
                # The generator sees Nfb*Nc rows carrying the same raw trial; collapse to the
                # raw channel count before filtering, then let the bank produce the sub-bands.
                raw_out = generated[:, :, :meta.num_channels, :]
                generated = filter_bank_layer(raw_out)
            src_features = getattr(generator, 'last_features', None)
            src_multi = getattr(generator, 'last_multi', None)
            # Must be grabbed here, like src_features above: the reference forward below
            # runs the same generator on the clean signal under no_grad and overwrites
            # this attribute, which would leave the mask loss with no gradient at all.
            src_mask_logits = getattr(generator, 'last_mask_logits', None)
            d_fake_for_g = discriminator(generated)
            loss_recon = criterion_recon(generated, clean)
            loss_cls = criterion_ce(d_fake_for_g['cls'], labels)
            loss_adv = -safe_log(d_fake_for_g['adv']).mean() if use_adv else torch.zeros((), device=device)

            ref_stats = None
            ref_features = None
            ref_multi = None
            if use_reference_kl or contrastive_weight > 0:
                with torch.no_grad():
                    ref_stats = generator(clean)
                    ref_features = getattr(generator, 'last_features', None)
                    ref_multi = getattr(generator, 'last_multi', None)

            if use_latent_kl:
                loss_kl = _standard_kl_divergence(src_stat1, src_stat2)
            elif use_reference_kl and ref_stats is not None:
                if la_mode in ('channel', 'temporal'):
                    # ENG-R4#9 asked for the two alignment directions to be ablated separately
                    lv = lambda t: _to_logvar(backbone, t)
                    if la_mode == 'channel':
                        loss_kl = _kl_pair(src_stat1, lv(src_stat2), ref_stats[0], lv(ref_stats[1]))
                    else:
                        loss_kl = _kl_pair(src_stat3, lv(src_stat4), ref_stats[2], lv(ref_stats[3]))
                elif la_mode == 'multi' and src_multi is not None and ref_multi is not None:
                    loss_kl = multi_depth_temporal_alignment(src_multi, ref_multi)
                elif la_mode == 'grouped' and src_features is not None and ref_features is not None:
                    loss_kl = grouped_channel_alignment(src_features, ref_features, groups=la_groups)
                elif la_mode == 'coral' and src_features is not None and ref_features is not None:
                    loss_kl = coral_alignment(src_features, ref_features)
                elif la_mode == 'proj' and la_projection is not None and src_features is not None and ref_features is not None:
                    loss_kl = projected_infonce(la_projection, src_features, ref_features,
                                                temperature=la_temperature, labels=labels)
                elif la_mode == 'none':
                    loss_kl = torch.zeros((), device=device)
                elif la_gate_mode != 'off' and epoch_index >= la_gate_warmup:
                    per_trial = _reference_kl_per_trial(
                        backbone, src_stat1, src_stat2, src_stat3, src_stat4,
                        ref_stats[0], ref_stats[1], ref_stats[2], ref_stats[3],
                    )
                    # Detached: the generator must not be able to switch its own alignment
                    # off by degrading the discriminator's read of the reference.
                    if la_gate_mode == 'hard':
                        gate = (d_real['cls'].argmax(dim=1) == labels).float().detach()
                    else:
                        gate = reference_probability.detach()
                    loss_kl = (gate * per_trial).sum() / gate.sum().clamp_min(1e-8)
                else:
                    loss_kl = _reference_kl_divergence(
                        backbone, src_stat1, src_stat2, src_stat3, src_stat4,
                        ref_stats[0], ref_stats[1], ref_stats[2], ref_stats[3],
                    )
            else:
                loss_kl = torch.zeros((), device=device)

            # harmonic-weighted spectral fidelity (SSVEP fundamental + harmonics)
            loss_spec = torch.zeros((), device=device)
            if spectral_weight > 0 and meta is not None:
                loss_spec = harmonic_spectral_loss(
                    generated, clean, labels, meta.freqs, meta.sample_rate,
                    num_harmonics=spectral_num_harmonics,
                    bandwidth_hz=spectral_bandwidth_hz,
                    harmonic_boost=spectral_harmonic_boost,
                    mode=spectral_mode,
                )

            # trial-level InfoNCE on the same statistics the KL alignment used
            loss_con = torch.zeros((), device=device)
            if contrastive_weight > 0 and ref_stats is not None:
                loss_con = info_nce_loss(
                    (src_stat1, src_stat2, src_stat3, src_stat4),
                    (ref_stats[0], ref_stats[1], ref_stats[2], ref_stats[3]),
                    temperature=contrastive_temperature,
                    labels=labels,
                    exclude_same_class=contrastive_exclude_same_class,
                )

            if mask_weight > 0 and src_mask_logits is not None and true_mask is not None:
                loss_mask = F.binary_cross_entropy_with_logits(src_mask_logits, true_mask)
            else:
                loss_mask = torch.zeros((), device=device)

            if reference_confidence > 0 and epoch_index >= confidence_warmup:
                # Detached: the generator must not be able to lower its own loss by making the
                # discriminator doubt the reference. Renormalised to mean one so the effective
                # step size does not drift as the discriminator sharpens.
                weights = reference_probability.detach()
                weights = confidence_floor + (1.0 - confidence_floor) * weights
                weights = weights / weights.mean().clamp_min(1e-8)
                weights = weights.view((-1,) + (1,) * (generated.dim() - 1))
                loss_recon = (weights * (generated - clean).abs().pow(
                    1 if reconstruction_loss == 'l1' else 2)).mean()
                loss_adv = (-safe_log(d_fake_for_g['adv']) * weights.view(-1, 1)).mean() \
                    if use_adv else loss_adv

            loss_g = adversarial_weight * loss_adv
            for weight, term in (
                (reconstruction_weight, loss_recon),
                (classification_weight, loss_cls),
                (kl_weight, loss_kl),
                (spectral_weight, loss_spec),
                (contrastive_weight, loss_con),
                (mask_weight, loss_mask),
            ):
                if weight > 0:
                    require_gradient(term, 'generator term', weight)
                    loss_g = loss_g + weight * term

            if not torch.isfinite(loss_g):
                nonfinite_batches += 1
                optimizer_g.zero_grad(set_to_none=True)
                continue

            optimizer_g.zero_grad()
            loss_g.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer_g.step()
            if use_scheduler:
                scheduler_g.step()

        if (epoch_index + 1) % eval_interval == 0 or epoch_index + 1 == epochs:
            generator.eval()
            discriminator.eval()
            fake_correct = 0
            real_correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    clean, source, labels = batch[0], batch[1], batch[2]
                    clean = clean.float().to(device)
                    source = source.float().to(device)
                    labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)
                    _, _, _, _, generated = generator(source)
                    d_fake = discriminator(generated)
                    d_real = discriminator(clean)
                    fake_correct += (d_fake['cls'].argmax(dim=1) == labels).sum().item()
                    real_correct += (d_real['cls'].argmax(dim=1) == labels).sum().item()
                    total += labels.shape[0]
            latest_fake_acc = fake_correct / max(total, 1)
            latest_real_acc = real_correct / max(total, 1)
            # if latest_fake_acc >= best_fake_acc:
            best_fake_acc = latest_fake_acc
            best_real_acc = latest_real_acc
            best_state = deepcopy(generator.state_dict())
            print_progress_line(f'stateful/{method}', epoch_index + 1, epochs, fake=latest_fake_acc, real=latest_real_acc)
    finish_progress_line(f'[stateful/{method}] best_fake={best_fake_acc:.4f} best_real={best_real_acc:.4f}')
    generator.load_state_dict(best_state)
    if nonfinite_batches:
        print(f'[stateful/{method}] WARNING: skipped {nonfinite_batches} non-finite generator batches')
    return generator, {
        'generator_nonfinite_batches': float(nonfinite_batches),
        'generator_fake_cls_acc': float(latest_fake_acc),
        'generator_real_cls_acc': float(latest_real_acc),
        'generator_best_fake_cls_acc': float(best_fake_acc),
        'generator_best_real_cls_acc': float(best_real_acc),
    }


def _canonical_method_name(method: str) -> str:
    return normalize_stateful_method_name(method) if is_stateful_generator_method(method) else method


def _build_pair_dataloaders(signals: ExperimentSignals, batch_size: int, include_masks: bool = False):
    clean_train = torch.tensor(flatten_structured_eeg(signals.clean.train), dtype=torch.float32).unsqueeze(1)
    clean_test = torch.tensor(flatten_structured_eeg(signals.clean.test), dtype=torch.float32).unsqueeze(1)
    source_train = torch.tensor(flatten_structured_eeg(signals.source.train), dtype=torch.float32).unsqueeze(1)
    source_test = torch.tensor(flatten_structured_eeg(signals.source.test), dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor(signals.train_labels.astype(np.int64))
    test_labels = torch.tensor(signals.test_labels.astype(np.int64))

    train_tensors = [clean_train, source_train, train_labels]
    test_tensors = [clean_test, source_test, test_labels]
    # Masks are appended only when a method asks for them, so the three-tuple contract
    # the legacy trainers (GCTNet / EEGDNet / CT-DCENet) rely on is untouched.
    if include_masks and signals.masks is not None:
        train_tensors.append(torch.tensor(signals.masks.train).float().unsqueeze(1))
        test_tensors.append(torch.tensor(signals.masks.test).float().unsqueeze(1))
    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    test_dataset = torch.utils.data.TensorDataset(*test_tensors)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _build_gct_generator(method: str, device: str, num_timesteps: int):
    if method == 'SimpleCNN':
        return basemodel.SimpleCNN(num_timesteps).to(device)
    if method == 'FCNN':
        return basemodel.FCNN(num_timesteps).to(device)
    if method == 'ResCNN':
        return basemodel.ResCNN(num_timesteps).to(device)
    if method == 'GCTNet':
        return GCTNet.Generator(num_timesteps).to(device)
    if method == 'GeneratorCNN':
        return GCTNet.GeneratorCNN(num_timesteps).to(device)
    if method == 'GeneratorTransformer':
        return GCTNet.GeneratorTransformer(num_timesteps).to(device)
    if method == 'NovelCNN':
        return basemodel.NovelCNN(num_timesteps).to(device)
    if method == 'DuoCL':
        return DuoCL.DuoCL(num_timesteps).to(device)
    raise ValueError(f'Unsupported GCT family method: {method}')


def _run_inference(
    method: str,
    generator,
    source_structured: np.ndarray,
    labels: np.ndarray,
    meta,
    device: str,
    batch_size: int,
) -> np.ndarray:
    canonical_method = _canonical_method_name(method)

    generator.eval()
    with torch.no_grad():
        if canonical_method in DIFFUSION_FAMILY:
            label_tensor = torch.tensor(labels.astype(np.int64), dtype=torch.long, device=device)
            generated_chunks = []
            for start_index in range(0, label_tensor.shape[0], max(1, batch_size)):
                end_index = start_index + max(1, batch_size)
                _, _, _, _, output = generator(label_tensor[start_index:end_index])
                generated_chunks.append(output.squeeze(1).cpu())
            generated = torch.cat(generated_chunks, dim=0)
            flat_channels = meta.num_filter_banks * meta.num_channels
            num_timesteps = meta.num_timesteps
        else:
            flat_source = torch.tensor(flatten_structured_eeg(source_structured), dtype=torch.float32).unsqueeze(1).to(device)
            flat_batch_size, _, flat_channels, num_timesteps = flat_source.shape

            if is_stateful_generator_method(canonical_method):
                _, _, _, _, output = generator(flat_source)
                generated = output.squeeze(1)
            elif canonical_method == 'CT_DCENet':
                flattened = flat_source.squeeze(1).reshape(-1, num_timesteps)
                generated = generator.denoise(flattened).reshape(flat_batch_size, flat_channels, num_timesteps)
            elif canonical_method == 'EEGDNet':
                flattened = flat_source.squeeze(1).reshape(-1, num_timesteps)
                generated = generator(flattened).reshape(flat_batch_size, flat_channels, num_timesteps)
            elif canonical_method in GCT_FAMILY:
                flattened = flat_source.squeeze(1).reshape(-1, num_timesteps)
                if canonical_method in ONE_DIMENSIONAL_INPUT_METHODS:
                    prediction = generator(flattened.unsqueeze(1))
                elif canonical_method in FLAT_INPUT_METHODS:
                    prediction = generator(flattened)
                else:
                    raise ValueError(f'Unsupported GCT family inference path for {canonical_method}.')
                generated = prediction.reshape(flat_batch_size, flat_channels, num_timesteps)
            else:
                raise ValueError(f'Unsupported augmentation method: {canonical_method}')

    return generated.cpu().numpy().reshape(-1, meta.num_filter_banks, meta.num_channels, num_timesteps).astype(np.float32)


def _safe_output_name(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in str(value)).strip('-_.') or 'default'


def _maybe_save_state(config, method: str, state_dict, split_tag: str, component: str):
    """Same as _maybe_save_checkpoint but for a component held as a bare state dict."""
    experiment_cfg = get_experiment_config(config)
    if not experiment_cfg.get('save_checkpoints', False):
        return
    checkpoint_root = get_output_paths(config).get('checkpoints')
    if not checkpoint_root:
        return
    split_dir = os.path.join(checkpoint_root, _safe_output_name(split_tag))
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(
        split_dir, f'{_safe_output_name(method)}__{_safe_output_name(component)}.pt')
    torch.save({'method': method, 'component': component, 'state_dict': state_dict}, path)


def _maybe_save_checkpoint(config, method: str, generator, split_tag: str, component: str = 'generator'):
    """Persist one component of one method for one split.

    The file name carries both the method and the component, so a method with more than one
    trained part (a field network and a classifier, or a generator and a discriminator) does
    not overwrite itself. Runs are already separated by the run name in the directory above.
    """
    experiment_cfg = get_experiment_config(config)
    if not experiment_cfg.get('save_checkpoints', False):
        return
    checkpoint_root = get_output_paths(config).get('checkpoints')
    if not checkpoint_root:
        return
    split_dir = os.path.join(checkpoint_root, _safe_output_name(split_tag))
    os.makedirs(split_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        split_dir, f'{_safe_output_name(method)}__{_safe_output_name(component)}.pt')
    torch.save({'method': method, 'component': component,
                'state_dict': generator.state_dict()}, checkpoint_path)

def train_augmentation_method(method: str, signals: ExperimentSignals, config, device: str, split_tag: str) -> AugmentationOutput:
    canonical_method = _canonical_method_name(method)
    method_cfg = get_augmentation_method_config(config, canonical_method)
    canonical_method, method_cfg = resolve_baseline_spec(canonical_method, method_cfg)
    flat_channels = signals.meta.num_filter_banks * signals.meta.num_channels
    metrics: Dict[str, float] = {}

    # Classical baseline (revision): per-trial, per-sub-band ICA with kurtosis-ranked removal.
    if str(canonical_method).upper() == 'ICA':
        from Trainers.Augmentation.classical import ica_denoise
        generated_train, generated_test, metrics = ica_denoise(signals, method_cfg)
        return AugmentationOutput(
            method='ICA',
            generated=SignalSet(train=generated_train, test=generated_test),
            metrics=metrics,
        )

    # EEGDfus baseline (revision): official conditional DDPM, one filter-bank row per segment.
    if str(canonical_method).upper() == 'EEGDFUS':
        from Trainers.Augmentation.eegdfus_trainer import eegdfus_denoise
        generated_train, generated_test, metrics = eegdfus_denoise(signals, method_cfg, device)
        return AugmentationOutput(
            method='EEGDfus',
            generated=SignalSet(train=generated_train, test=generated_test),
            metrics=metrics,
        )

    if is_direct_augmentation_method(canonical_method):
        train_flat = flatten_structured_eeg(signals.source.train)
        test_flat = flatten_structured_eeg(signals.source.test)
        generated_train_flat = apply_direct_augmentation(
            train_flat,
            signals.train_labels,
            canonical_method,
            signals.meta.sample_rate,
            method_cfg,
        )
        if bool(method_cfg.get('apply_to_test', False)):
            generated_test_flat = apply_direct_augmentation(
                test_flat,
                signals.test_labels,
                canonical_method,
                signals.meta.sample_rate,
                method_cfg,
            )
        else:
            generated_test_flat = test_flat.copy()

        generated_train = generated_train_flat.reshape(
            generated_train_flat.shape[0],
            signals.meta.num_filter_banks,
            signals.meta.num_channels,
            signals.meta.num_timesteps,
        ).astype(np.float32)
        generated_test = generated_test_flat.reshape(
            generated_test_flat.shape[0],
            signals.meta.num_filter_banks,
            signals.meta.num_channels,
            signals.meta.num_timesteps,
        ).astype(np.float32)
        metrics = {
            'direct_aug_train_samples': float(generated_train.shape[0]),
            'direct_aug_test_samples': float(generated_test.shape[0]),
            'apply_to_test': float(bool(method_cfg.get('apply_to_test', False))),
        }
        return AugmentationOutput(
            method=canonical_method,
            generated=SignalSet(train=generated_train, test=generated_test),
            metrics=metrics,
        )

    batch_size = int(method_cfg.get('batch_size', get_default_batch_size(config)))
    train_loader, test_loader = _build_pair_dataloaders(
        signals, batch_size, include_masks=bool(method_cfg.get('mask_gate', False))
    )

    if is_generative_method(canonical_method):
        model = build_generative_model(
            canonical_method, flat_channels, signals.meta.num_timesteps,
            signals.meta.num_classes, method_cfg,
        )
        reuse = str(method_cfg.get('load_field_from', '') or '')
        if reuse:
            field_path = os.path.join(reuse, 'checkpoints', _safe_output_name(split_tag),
                                      f'{_safe_output_name(canonical_method)}__field.pt')
            if not os.path.exists(field_path):
                raise FileNotFoundError(f'no field checkpoint to reuse at {field_path}')
            model.load_state_dict(torch.load(field_path, map_location="cpu")['state_dict'])
            model = model.to(device)
            cls_path = field_path.replace('__field.pt', '__classifier.pt')
            if os.path.exists(cls_path):
                model.classifier_state = torch.load(cls_path, map_location="cpu")['state_dict']
            metrics = {'reused_from': 1.0}
            print(f'[{canonical_method}] reusing weights from {field_path}')
        else:
            model, metrics = train_generative_denoiser(
                train_loader, test_loader, device, model, canonical_method,
                flat_channels, signals.meta.num_timesteps, signals.meta.num_classes, method_cfg,
            )
        generated_train = run_generative_inference(
            model, canonical_method, signals.source.train, signals.train_labels,
            signals.meta, device, batch_size)
        generated_test = run_generative_inference(
            model, canonical_method, signals.source.test, signals.test_labels,
            signals.meta, device, batch_size)
        _maybe_save_checkpoint(config, canonical_method, model, split_tag, 'field')
        if getattr(model, 'classifier_state', None) is not None:
            _maybe_save_state(config, canonical_method, model.classifier_state,
                              split_tag, 'classifier')
        return AugmentationOutput(
            method=canonical_method,
            generated=SignalSet(train=generated_train, test=generated_test),
            metrics=metrics,
        )

    if is_stateful_generator_method(canonical_method):
        generator = build_stateful_generator(
            canonical_method, flat_channels, signals.meta.num_timesteps, method_cfg,
            num_filter_banks=signals.meta.num_filter_banks,
        )
        generator, metrics = train_stateful_generator(
            train_loader,
            test_loader,
            device,
            generator,
            canonical_method,
            flat_channels,
            signals.meta.num_timesteps,
            signals.meta.num_classes,
            method_cfg,
            meta=signals.meta,
        )
    elif canonical_method in DIFFUSION_FAMILY:
        if canonical_method == 'Diffusion':
            fake_acc, real_acc, generator, _ = DiffusionTrainer.train_diffusion_generator(train_loader, test_loader, signals.meta, method_cfg, device)
        else:
            fake_acc, real_acc, generator, _ = DiffusionTrainer.train_diffusion_gan_generator(train_loader, test_loader, signals.meta, method_cfg, device)
        metrics = {
            'generator_fake_cls_acc': float(fake_acc),
            'generator_real_cls_acc': float(real_acc),
        }
    elif canonical_method == 'CT_DCENet':
        generator = CT_DCENet(
            signal_len=signals.meta.num_timesteps,
            head_type_list=method_cfg.get('head_types', [0, 0, 1, 1]),
            block_num=int(method_cfg.get('num_blocks', 5)),
        ).to(device)
        generator, _ = LegacyTrainer.train_ct_dcenet_with_cls(
            train_loader,
            test_loader,
            device,
            generator,
            signals.meta.num_classes,
            epochs_stage1=int(method_cfg.get('stage1_epochs', 100)),
            epochs_stage2=int(method_cfg.get('stage2_epochs', 100)),
            classification_weight=float(method_cfg.get('classification_weight', 1.0)),
        )
    elif canonical_method == 'EEGDNet':
        generator = DeT(
            seq_len=signals.meta.num_timesteps,
            patch_len=int(method_cfg.get('patch_length', 64)),
            depth=int(method_cfg.get('depth', 6)),
            heads=int(method_cfg.get('num_heads', 1)),
        ).to(device)
        generator = LegacyTrainer.train_eegdnet(
            train_loader,
            test_loader,
            device,
            generator,
            signals.meta.num_classes,
            epochs=int(method_cfg.get('epochs', 500)),
            lr=float(method_cfg.get('lr', 1e-3)),
            lr_d=float(method_cfg.get('discriminator_lr', 1e-4)),
            weight_decay_d=float(method_cfg.get('discriminator_weight_decay', 0.0)),
            betas=tuple(method_cfg.get('betas', [0.5, 0.9])),
            eps=float(method_cfg.get('eps', 1e-8)),
            classification_weight=float(method_cfg.get('classification_weight', 1.0)),
        )
    elif canonical_method in GCT_FAMILY:
        generator = _build_gct_generator(canonical_method, device, signals.meta.num_timesteps)
        generator, _, fake_acc, real_acc = LegacyTrainer.train_gctnet(
            train_loader,
            test_loader,
            generator,
            int(method_cfg.get('epochs', 200)),
            signals.meta.num_classes,
            device,
            classification_weight=float(method_cfg.get('classification_weight', 0.05)),
            adversarial_weight=float(method_cfg.get('adversarial_weight', 0.05)),
            feature_weight=float(method_cfg.get('feature_weight', 0.05)),
        )
        metrics = {
            'generator_fake_cls_acc': float(fake_acc),
            'generator_real_cls_acc': float(real_acc),
        }
    else:
        raise ValueError(f'Unsupported augmentation method: {canonical_method}')

    generated_train = _run_inference(canonical_method, generator, signals.source.train, signals.train_labels, signals.meta, device, batch_size)
    generated_test = _run_inference(canonical_method, generator, signals.source.test, signals.test_labels, signals.meta, device, batch_size)
    _maybe_save_checkpoint(config, canonical_method, generator, split_tag)

    return AugmentationOutput(
        method=canonical_method,
        generated=SignalSet(train=generated_train, test=generated_test),
        metrics=metrics,
    )



