import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.Classification.diffusion_ssvep_classifier import DiffusionSSVEPClassifier
from Support.config import get_diffusion_classifier_config, get_signal_config, get_training_config
from Support.utils import finish_progress_line, print_progress_line


def compute_harmonic_spectral_loss(
    x_pred,
    x_true,
    labels,
    freqs,
    sample_rate,
    num_harmonics,
    bandwidth_hz=0.4,
    harmonic_boost=4.0,
):
    pred_fft = torch.fft.rfft(x_pred.squeeze(1), dim=-1)
    true_fft = torch.fft.rfft(x_true.squeeze(1), dim=-1)
    pred_mag = torch.log1p(pred_fft.abs())
    true_mag = torch.log1p(true_fft.abs())

    fft_freqs = torch.fft.rfftfreq(x_true.shape[-1], d=1.0 / sample_rate).to(x_true.device)
    class_freqs = torch.as_tensor(freqs, device=x_true.device, dtype=x_true.dtype)[labels]
    weights = torch.ones((labels.shape[0], fft_freqs.shape[0]), device=x_true.device, dtype=x_true.dtype)

    for harmonic_index in range(1, num_harmonics + 1):
        centers = class_freqs.unsqueeze(1) * harmonic_index
        mask = (fft_freqs.unsqueeze(0) - centers).abs() <= bandwidth_hz
        weights = weights + mask.to(x_true.dtype) * (harmonic_boost / harmonic_index)

    return ((pred_mag - true_mag).pow(2) * weights.unsqueeze(1)).mean()


def train_diffusion_classifier(train_dataset, test_dataset, meta, config, device):
    classifier_cfg = get_diffusion_classifier_config(config)
    schedule_cfg = classifier_cfg.get('schedule', {})
    training_cfg = classifier_cfg.get('training', {})
    model_cfg = classifier_cfg.get('model', {})
    signal_cfg = get_signal_config(config)
    default_training_cfg = get_training_config(config)

    sample_rate = meta.sample_rate
    num_classes = meta.num_classes
    num_timesteps = meta.num_timesteps
    flat_channels = meta.num_filter_banks * meta.num_channels
    freqs = meta.freqs

    num_diffusion_steps = int(schedule_cfg.get('num_steps', 200))
    beta_start = float(schedule_cfg.get('beta_start', 1e-4))
    beta_end = float(schedule_cfg.get('beta_end', 0.02))
    num_harmonics = int(training_cfg.get('num_harmonics', 5))
    noise_loss_weight = float(training_cfg.get('noise_loss_weight', 0.5))
    prototype_loss_weight = float(training_cfg.get('prototype_loss_weight', 0.0))
    spectral_loss_weight = float(training_cfg.get('spectral_loss_weight', 0.0))
    epochs = int(training_cfg.get('epochs', default_training_cfg.get('epochs', 300)))
    lr = float(training_cfg.get('lr', 0.001))
    weight_decay = float(training_cfg.get('weight_decay', 0.0001))
    batch_size = int(training_cfg.get('batch_size', default_training_cfg.get('batch_size', 36)))
    eval_interval = int(training_cfg.get('eval_interval', 10))
    dropout = float(model_cfg.get('dropout', 0.5))
    use_input_norm = bool(model_cfg.get('use_input_norm', True))
    use_proto_fusion = bool(model_cfg.get('use_proto_fusion', False))
    classifier_mode = str(model_cfg.get('classifier_mode', 'attn_proto' if use_proto_fusion else 'attn'))
    logit_mix = float(model_cfg.get('logit_mix', 0.5))
    score_temperature = float(model_cfg.get('score_temperature', 8.0))
    timestep_sampling = str(model_cfg.get('timestep_sampling', 'uniform'))
    timestep_power = float(model_cfg.get('timestep_power', 2.0))
    spectral_bandwidth_hz = float(model_cfg.get('spectral_bandwidth_hz', 0.4))
    spectral_harmonic_boost = float(model_cfg.get('spectral_harmonic_boost', 4.0))

    classifier = DiffusionSSVEPClassifier(
        C_eeg=flat_channels,
        Nt=num_timesteps,
        Fs=sample_rate,
        freqs=freqs,
        Nf=num_classes,
        T_diff=num_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        num_harmonics=num_harmonics,
        dropout=dropout,
        use_input_norm=use_input_norm,
        use_proto_fusion=use_proto_fusion,
        logit_mix=logit_mix,
        score_temperature=score_temperature,
        timestep_sampling=timestep_sampling,
        timestep_power=timestep_power,
        classifier_mode=classifier_mode,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_noise = nn.MSELoss()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_steps = max(1, epochs * max(1, len(train_loader)))
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)

    best_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch_index in range(epochs):
        classifier.train()
        epoch_loss_cls = 0.0
        epoch_loss_noise = 0.0
        epoch_loss_proto = 0.0
        epoch_loss_spec = 0.0
        epoch_correct = 0
        epoch_total = 0

        for eeg, labels in train_loader:
            eeg = eeg.float().to(device)
            labels = labels.long().to(device)

            outputs = classifier.training_step(eeg, labels)
            logits = outputs['logits']
            loss_cls = criterion_cls(logits, labels)
            loss_noise = criterion_noise(outputs['noise_pred'], outputs['noise'])

            loss_proto = torch.zeros((), device=device)
            if prototype_loss_weight > 0:
                loss_proto = criterion_cls(outputs['proto_logits'], labels)

            loss_spec = torch.zeros((), device=device)
            if spectral_loss_weight > 0:
                x0_pred = classifier.predict_x0(outputs['x_t'], outputs['t'], outputs['noise_pred'])
                loss_spec = compute_harmonic_spectral_loss(
                    x0_pred,
                    outputs['x0'],
                    labels,
                    freqs,
                    sample_rate,
                    num_harmonics,
                    bandwidth_hz=spectral_bandwidth_hz,
                    harmonic_boost=spectral_harmonic_boost,
                )

            loss = (
                loss_cls
                + noise_loss_weight * loss_noise
                + prototype_loss_weight * loss_proto
                + spectral_loss_weight * loss_spec
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler_lr.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_noise += loss_noise.item()
            epoch_loss_proto += loss_proto.item()
            epoch_loss_spec += loss_spec.item()
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.shape[0]

        should_eval = ((epoch_index + 1) % eval_interval == 0) or (epoch_index + 1 == epochs)
        if should_eval:
            acc = evaluate_diffusion_classifier(classifier, test_loader, device)
            train_acc = epoch_correct / max(epoch_total, 1)
            avg_cls = epoch_loss_cls / max(len(train_loader), 1)
            avg_noise = epoch_loss_noise / max(len(train_loader), 1)
            avg_proto = epoch_loss_proto / max(len(train_loader), 1)
            avg_spec = epoch_loss_spec / max(len(train_loader), 1)

            if acc >= best_acc:
                best_acc = acc
                best_epoch = epoch_index + 1
                best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}

            print_progress_line(
                'diff-cls',
                epoch_index + 1,
                epochs,
                ce=avg_cls,
                mse=avg_noise,
                proto=avg_proto,
                spec=avg_spec,
                train=train_acc,
                test=acc,
                best=best_acc,
            )

    if best_state is not None:
        classifier.load_state_dict(best_state)

    finish_progress_line(
        'Diffusion classifier finished at '
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
        f'best_acc={best_acc:.4f} best_epoch={best_epoch}'
    )

    return best_acc, classifier


@torch.no_grad()
def evaluate_diffusion_classifier(classifier, test_loader, device):
    classifier.eval()
    total_correct = 0
    total_samples = 0

    for eeg, labels in test_loader:
        eeg = eeg.float().to(device)
        labels = labels.long().to(device)
        predictions, _ = classifier.classify(eeg)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.shape[0]

    return total_correct / total_samples if total_samples > 0 else 0.0


def compute_itr(num_classes, accuracy, decision_window_sec):
    if accuracy <= 0 or decision_window_sec <= 0 or num_classes <= 1:
        return 0.0
    if accuracy >= 1.0:
        accuracy = 0.9999
    if accuracy <= 1.0 / num_classes:
        return 0.0

    bits = (
        np.log2(num_classes)
        + accuracy * np.log2(accuracy)
        + (1 - accuracy) * np.log2((1 - accuracy) / (num_classes - 1))
    )
    return bits * (60.0 / decision_window_sec)


# Backward-compatible aliases.
harmonic_spectral_loss = compute_harmonic_spectral_loss
train_cross_subject = train_diffusion_classifier
evaluate = evaluate_diffusion_classifier
