"""Training loop for the conditional generative denoisers.

Kept separate from train_stateful_generator because the objective differs in kind: the
network predicts a velocity or a denoised iterate at a sampled time rather than the signal
in one shot, so the reconstruction term is defined on that prediction. Adversarial and class
supervision are optional and, when enabled, act on the sampled output so they see exactly
what downstream decoding will see.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from Models.Augmentation import discriminators as Discriminator
from Models.Augmentation.generative import DiffusionDenoiser, FlowDenoiser
from Support.utils import finish_progress_line, print_progress_line, require_gradient

GENERATIVE_METHODS = {
    'FLOW': ('flow', False, False),
    'FLOW_CLS': ('flow', False, True),
    'FLOW_GAN': ('flow', True, True),
    'DIFF': ('diffusion', False, False),
    'DIFF_CLS': ('diffusion', False, True),
    'DIFF_GAN': ('diffusion', True, True),
    # Same model, but the forward process corrupts with the recorded artifact instead of
    # Gaussian noise, so the reverse process inverts the degradation actually present.
    'DIFF_REAL': ('diffusion', False, False),
    'DIFF_REAL_CLS': ('diffusion', False, True),
}


def is_generative_method(method):
    return str(method).upper() in GENERATIVE_METHODS


def build_generative_model(method, flat_channels, num_timesteps, num_classes, method_cfg):
    family, _, use_cls = GENERATIVE_METHODS[str(method).upper()]
    dropout = float(method_cfg.get('dropout', 0.5))
    # The field network is given no label conditioning: at test time the label is what the
    # decoders are being asked for, so conditioning on it would put the answer in the signal.
    # use_cls still selects the classifier and its loss, which only reads labels in training.
    classes = 0
    if family == 'flow':
        arch = {k: method_cfg[k] for k in
                ('width', 'temporal_kernel', 'num_layers', 'nhead', 'dim_feedforward',
                 'scale2_layers', 'skip_mode', 'spatial_layers', 'temporal_layers',
                 'time_mode', 'conv_order')
                if k in method_cfg}
        return FlowDenoiser(flat_channels, num_timesteps, dropout=dropout,
                            num_classes=classes, steps=int(method_cfg.get('flow_steps', 4)),
                            arch=arch)
    return DiffusionDenoiser(flat_channels, num_timesteps, dropout=dropout, num_classes=classes,
                             num_steps=int(method_cfg.get('diffusion_steps', 200)),
                             infer_steps=int(method_cfg.get('inference_steps', 10)),
                             artifact_forward=str(method).upper().startswith('DIFF_REAL'))


def train_generative_denoiser(train_loader, test_loader, device, model, method,
                              flat_channels, num_timesteps, num_classes, method_cfg):
    family, use_adv, use_cls = GENERATIVE_METHODS[str(method).upper()]
    epochs = int(method_cfg.get('epochs', 300))
    lr = float(method_cfg.get('generator_lr', 1e-3))
    weight_decay = float(method_cfg.get('generator_weight_decay', 1e-4))
    adv_weight = float(method_cfg.get('adversarial_weight', 0.3))
    cls_weight = float(method_cfg.get('classification_weight', 1.0))
    sample_weight = float(method_cfg.get('sample_weight', 1.0))
    # 'ce' keeps the earlier behaviour: the class term reads the softmax output and
    # nn.CrossEntropyLoss softmaxes it again, which flattens it to a constant.
    cls_loss_form = str(method_cfg.get('cls_loss_form', 'nll'))
    # The velocity term carries a weight only so that the ablation can remove it;
    # every configuration in use leaves it at one.
    flow_weight = float(method_cfg.get('flow_weight', 1.0))
    grad_clip = float(method_cfg.get('grad_clip', 1.0))
    eval_interval = int(method_cfg.get('eval_interval', 10))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * max(1, len(train_loader))), eta_min=5e-6)

    discriminator = None
    if use_adv or use_cls:
        discriminator = Discriminator.Spectral_Normalization(
            Discriminator.ESNet(flat_channels, num_timesteps, num_classes)).to(device)
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=float(method_cfg.get('discriminator_lr', 1e-3)),
            weight_decay=float(method_cfg.get('discriminator_weight_decay', 1e-4)))
    criterion_ce = nn.CrossEntropyLoss()
    safe_log = lambda z: torch.log(z + 1e-8)

    final_metric = float('inf')
    latest_fake_acc = latest_real_acc = 0.0

    for epoch in range(epochs):
        model.train()
        # The classifier holds an LSTM, and cudnn refuses a backward pass on one left in eval
        # mode by the periodic evaluation below.
        if discriminator is not None:
            discriminator.train()
        for batch in train_loader:
            clean, source, labels = batch[0].float().to(device), batch[1].float().to(device), batch[2]
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)
            label_arg = None  # the field network is unconditioned; see above

            if family == 'flow':
                # With the velocity term switched off, its forward pass is skipped as well: run in
                # training mode it would still update the BatchNorm statistics with path points x_t.
                if flow_weight > 0:
                    pred, target = model.velocity_target(clean, source, label_arg)
                    loss = flow_weight * F.mse_loss(pred, target)
                else:
                    loss = torch.zeros((), device=clean.device)
            else:
                loss = F.mse_loss(model.predict_x0(clean, source, label_arg), clean)

            # A term on the actually-sampled output: the field loss alone leaves the
            # multi-step trajectory unconstrained, and it is the trajectory that is decoded.
            if sample_weight > 0 or use_adv or use_cls:
                generated = model.rollout(
                    source, label_arg,
                    int(method_cfg.get('train_sample_steps', 2))) if family == 'flow' \
                    else model.rollout(source, label_arg)
                require_gradient(generated, 'sampled output', 1.0)
                if sample_weight > 0:
                    loss = loss + sample_weight * F.mse_loss(generated, clean)

                d_weighted = use_adv or (use_cls and cls_weight > 0)
                if discriminator is not None and d_weighted:
                    d_real = discriminator(clean.detach())
                    d_fake = discriminator(generated.detach())
                    loss_d = torch.zeros((), device=device)
                    if use_adv:
                        loss_d = loss_d + adv_weight * (
                            -safe_log(d_real['adv']).mean() - safe_log(1 - d_fake['adv']).mean())
                    if use_cls and cls_weight > 0:
                        loss_d = loss_d + cls_weight * (
                            (-safe_log(d_real['cls'][range(len(labels)), labels]).mean()
                             - safe_log(d_fake['cls'][range(len(labels)), labels]).mean())
                            if cls_loss_form == 'ce' else
                            (F.cross_entropy(d_real['cls_logits'], labels)
                             + F.cross_entropy(d_fake['cls_logits'], labels)))
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

                    d_for_g = discriminator(generated)
                    if use_adv:
                        loss = loss + adv_weight * (-safe_log(d_for_g['adv']).mean())
                    if use_cls and cls_weight > 0:
                        cls_term = (criterion_ce(d_for_g['cls'], labels)
                                    if cls_loss_form == 'ce' else
                                    F.cross_entropy(d_for_g['cls_logits'], labels))
                        loss = loss + cls_weight * cls_term

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
            model.eval()
            if discriminator is not None:
                discriminator.eval()
            total_error, count = 0.0, 0
            fake_correct, real_correct, total_trials = 0, 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    clean, source = batch[0].float().to(device), batch[1].float().to(device)
                    all_labels = torch.as_tensor(batch[2].reshape(batch[2].shape[0]),
                                                 dtype=torch.int64).to(device)
                    labels = all_labels if use_cls else None
                    generated = model.sample(source, labels)
                    total_error += F.mse_loss(generated, clean, reduction='sum').item()
                    count += clean.numel()
                    # The classifier is evaluated on the same held-out trials as the downstream
                    # decoders, so its accuracy is directly comparable with theirs.
                    if discriminator is not None:
                        fake_correct += (discriminator(generated)['cls'].argmax(1) == all_labels).sum().item()
                        real_correct += (discriminator(clean)['cls'].argmax(1) == all_labels).sum().item()
                        total_trials += all_labels.numel()
            metric = total_error / max(count, 1)
            if total_trials:
                latest_fake_acc = fake_correct / total_trials
                latest_real_acc = real_correct / total_trials
            # Reported only; the weights are never chosen by it.
            final_metric = metric
            print_progress_line(method, epoch + 1, epochs, test_mse=metric)
    finish_progress_line()

    model.classifier_state = discriminator.state_dict() if discriminator is not None else None
    return model, {'generator_final_test_mse': float(final_metric),
                   'generator_fake_cls_acc': float(latest_fake_acc),
                   'generator_real_cls_acc': float(latest_real_acc)}


@torch.no_grad()
def run_generative_inference(model, method, source_structured, labels, meta, device,
                            batch_size, label_mode='true'):
    from Data.core import flatten_structured_eeg
    _, _, use_cls = GENERATIVE_METHODS[str(method).upper()]
    model.eval()
    flat = torch.tensor(flatten_structured_eeg(source_structured), dtype=torch.float32).unsqueeze(1)
    label_tensor = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.int64)
    outputs = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start:start + batch_size].to(device)
        chunk_labels = None
        outputs.append(model.sample(chunk, chunk_labels).squeeze(1).cpu().numpy())
    generated = np.concatenate(outputs, axis=0)
    num_samples, _, num_timesteps = generated.shape
    return generated.reshape(num_samples, meta.num_filter_banks, meta.num_channels,
                             num_timesteps).astype(np.float32)
