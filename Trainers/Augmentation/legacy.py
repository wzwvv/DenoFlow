import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Augmentation import discriminators as Discriminator
from Models.Augmentation.GCTNet.models.GCTNet import Discriminator as GCTNetDiscriminator


def initialize_conv_weights(module):
    if isinstance(module, nn.Conv1d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def _infer_signal_shape(data_loader):
    clean, _, _ = next(iter(data_loader))
    if clean.ndim != 4:
        raise ValueError(f'Expected batched EEG to have shape (B, 1, C, T), got {tuple(clean.shape)}')
    return int(clean.shape[-2]), int(clean.shape[-1])


def train_gctnet(eeg_train_dataloader, eeg_test_dataloader, model, epochs, num_classes, device,
                 classification_weight=0.05, adversarial_weight=0.05, feature_weight=0.05):
    model_d = GCTNetDiscriminator(Nf=num_classes).to(device)
    model_d.apply(initialize_conv_weights)
    model.apply(initialize_conv_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9), eps=1e-8)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.0001)
    criterion_ce = torch.nn.CrossEntropyLoss()

    for epoch_index in range(epochs):
        model.train()
        model_d.train()
        losses = []

        for clean, source, labels in eeg_train_dataloader:
            source = source.float().squeeze().to(device)
            clean = clean.float().squeeze().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

            if source.dim() == 3:
                batch_size, channel_count, signal_length = source.shape
                source = source.view(batch_size * channel_count, signal_length).unsqueeze(dim=1)
                clean = clean.view(batch_size * channel_count, signal_length)
                labels = labels.repeat_interleave(channel_count)

            prediction = model(source).view(source.shape[0], -1)
            fake_adv, fake_cls, _, _, _ = model_d(prediction.unsqueeze(1))
            real_adv, real_cls, _, _, _ = model_d(clean.unsqueeze(1))

            loss_d = 0.5 * torch.mean(fake_adv ** 2) + 0.5 * torch.mean((real_adv - 1) ** 2)
            loss_d_cls = criterion_ce(real_cls, labels) + criterion_ce(fake_cls, labels)
            loss_d = loss_d + adversarial_weight * loss_d_cls

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            prediction = model(source).view(source.shape[0], -1)
            fake_adv, fake_cls, _, fake_feature, _ = model_d(prediction.unsqueeze(1))
            _, _, _, true_feature, _ = model_d(clean.unsqueeze(1))

            loss_mse = denoise_loss_mse(prediction, clean)
            loss_feat = denoise_loss_mse(fake_feature, true_feature)
            loss_cls = criterion_ce(fake_cls, labels)
            loss_g = loss_mse + feature_weight * loss_feat + adversarial_weight * torch.mean((fake_adv - 1) ** 2) + classification_weight * loss_cls

            optimizer.zero_grad()
            optimizer_d.zero_grad()
            loss_g.backward()
            optimizer.step()
            losses.append(loss_g.detach())

        print(f"\rEpoch [{epoch_index + 1}/{epochs}] Train loss: {torch.stack(losses).mean().item():.4f}", end='')

    model.eval()
    model_d.eval()
    correct_fake = 0
    correct_real = 0
    total = 0

    with torch.no_grad():
        for clean, source, labels in eeg_test_dataloader:
            source = source.float().squeeze().to(device)
            clean = clean.float().squeeze().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

            if source.dim() == 3:
                batch_size, channel_count, signal_length = source.shape
                source = source.view(batch_size * channel_count, signal_length).unsqueeze(dim=1)
                clean = clean.view(batch_size * channel_count, signal_length)
                labels = labels.repeat_interleave(channel_count)

            prediction = model(source).view(source.shape[0], -1)
            _, fake_cls, _, _, _ = model_d(prediction.unsqueeze(1))
            _, real_cls, _, _, _ = model_d(clean.unsqueeze(1))

            correct_fake += (fake_cls.argmax(dim=1) == labels).sum().item()
            correct_real += (real_cls.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    fake_acc = correct_fake / max(total, 1)
    real_acc = correct_real / max(total, 1)
    print(f'Test fake-cls acc: {fake_acc:.4f}, real-cls acc: {real_acc:.4f}')
    return model, model_d, fake_acc, real_acc


def calculate_consensus_column(prediction_list, prediction_index):
    values = []
    target_shape = prediction_list[prediction_index].shape
    for other_index, prediction in enumerate(prediction_list):
        if other_index == prediction_index:
            continue
        if prediction.shape == target_shape:
            values.append(prediction.detach())
    if not values:
        return prediction_list[prediction_index].detach()
    return torch.mean(torch.stack(values), dim=0)


def collaborative_loss(prediction_index, prediction_list, target, distance_fn, beta_height=0.05):
    prediction = prediction_list[prediction_index]
    target_loss = distance_fn(prediction, target)
    consensus = calculate_consensus_column(prediction_list, prediction_index)
    correlation_weight = random.uniform(0, beta_height)
    correlation_loss = distance_fn(prediction, consensus)
    return target_loss + correlation_weight * correlation_loss


def train_ct_dcenet_with_cls(
    eeg_train_dataloader,
    eeg_test_dataloader,
    device,
    model,
    num_classes,
    epochs_stage1=50,
    epochs_stage2=50,
    beta_height=0.05,
    lr=1e-3,
    lr_d=1e-4,
    weight_decay_d=0.0,
    betas=(0.9, 0.99),
    eps=1e-8,
    classification_weight=1.0,
):
    flat_channels, num_timesteps = _infer_signal_shape(eeg_train_dataloader)
    model = model.to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(flat_channels, num_timesteps, num_classes)
    ).to(device)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, weight_decay=weight_decay_d)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d,
        T_max=max(1, (epochs_stage1 + epochs_stage2) * len(eeg_train_dataloader)),
        eta_min=5e-6,
    )
    criterion_ce = nn.CrossEntropyLoss()

    print('\n==== Stage 1: Training Individual Learners + CLS ====')
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith(('sl', 'supplement', 'beta'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=betas, eps=eps)

    for epoch_index in range(epochs_stage1):
        model.train()
        discriminator.train()
        epoch_loss = 0.0

        for clean, source, labels in eeg_train_dataloader:
            source = source.float().squeeze().to(device)
            clean = clean.float().squeeze().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

            if source.dim() == 3:
                batch_size, channel_count, signal_length = source.shape
                source = source.view(batch_size * channel_count, signal_length)
                clean = clean.view(batch_size * channel_count, signal_length)
            else:
                batch_size = source.shape[0]
                channel_count = 1
                signal_length = source.shape[1]

            std = source.std(dim=1, keepdim=True) + 1e-6
            normalized_source = source / std
            normalized_clean = clean / std
            prediction_list = model(normalized_source)

            loss_unsupervised = 0.0
            for prediction_index in range(len(prediction_list)):
                distance_fn = F.l1_loss if prediction_index % 2 == 0 else F.mse_loss
                loss_unsupervised += collaborative_loss(prediction_index, prediction_list, normalized_clean, distance_fn, beta_height=beta_height)

            generated = torch.mean(torch.stack(prediction_list), dim=0)
            d_real = discriminator(normalized_clean.view(batch_size, channel_count, signal_length).unsqueeze(1))
            d_fake = discriminator(generated.view(batch_size, channel_count, signal_length).detach().unsqueeze(1))
            loss_d_cls = criterion_ce(d_real['cls'], labels) + criterion_ce(d_fake['cls'], labels)

            optimizer_d.zero_grad()
            loss_d_cls.backward()
            optimizer_d.step()
            scheduler_d.step()

            d_fake_for_g = discriminator(generated.view(batch_size, channel_count, signal_length).unsqueeze(1))
            loss_g_cls = criterion_ce(d_fake_for_g['cls'], labels)
            loss_g = loss_unsupervised + classification_weight * loss_g_cls

            optimizer.zero_grad()
            loss_g.backward()
            optimizer.step()
            epoch_loss += loss_g.item()

        print(f"\r[Stage1][Epoch {epoch_index + 1}/{epochs_stage1}] Loss: {epoch_loss / len(eeg_train_dataloader):.4f}", end='')

    print('\n==== Stage 2: Training Ensemble & Supplement + CLS ====')
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(('sl', 'supplement', 'beta'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=betas, eps=eps)

    for epoch_index in range(epochs_stage2):
        model.train()
        discriminator.train()
        epoch_loss = 0.0

        for clean, source, labels in eeg_train_dataloader:
            source = source.float().squeeze().to(device)
            clean = clean.float().squeeze().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

            if source.dim() == 3:
                batch_size, channel_count, signal_length = source.shape
                source = source.view(batch_size * channel_count, signal_length)
                clean = clean.view(batch_size * channel_count, signal_length)
            else:
                batch_size = source.shape[0]
                channel_count = 1
                signal_length = source.shape[1]

            std = source.std(dim=1, keepdim=True) + 1e-6
            normalized_source = source / std
            normalized_clean = clean / std
            prediction = model.denoise(normalized_source)
            loss_mse = F.mse_loss(prediction, normalized_clean)
            d_fake = discriminator(prediction.view(batch_size, channel_count, signal_length).unsqueeze(1))
            loss_cls = criterion_ce(d_fake['cls'], labels)
            loss = loss_mse + classification_weight * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"\r[Stage2][Epoch {epoch_index + 1}/{epochs_stage2}] Loss: {epoch_loss / len(eeg_train_dataloader):.4f}", end='')
    print()
    return model, discriminator


def denoise_loss_mse(denoise, clean):
    return F.mse_loss(denoise, clean)


def train_eegdnet(
    eeg_train_dataloader,
    eeg_test_dataloader,
    device,
    model,
    num_classes,
    epochs=10000,
    lr=5e-4,
    lr_d=1e-4,
    weight_decay_d=0.0,
    betas=(0.5, 0.9),
    eps=1e-8,
    classification_weight=1.0,
):
    flat_channels, num_timesteps = _infer_signal_shape(eeg_train_dataloader)
    model = model.to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(flat_channels, num_timesteps, num_classes)
    ).to(device)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, weight_decay=weight_decay_d)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d,
        T_max=max(1, epochs * len(eeg_train_dataloader)),
        eta_min=5e-6,
    )
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    for epoch_index in range(epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0.0

        for clean, source, labels in eeg_train_dataloader:
            clean = clean.float().to(device)
            source = source.float().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.long).to(device)

            if source.dim() == 4:
                source = source.squeeze(1)
                clean = clean.squeeze(1)

            if source.dim() == 3:
                batch_size, channel_count, signal_length = source.shape
                source = source.view(batch_size * channel_count, signal_length)
                clean = clean.view(batch_size * channel_count, signal_length)
            else:
                batch_size = source.shape[0]
                channel_count = 1
                signal_length = source.shape[1]

            prediction = model(source)
            d_real = discriminator(clean.view(batch_size, channel_count, signal_length).unsqueeze(1))
            d_fake = discriminator(prediction.detach().view(batch_size, channel_count, signal_length).unsqueeze(1))
            loss_d_cls = criterion_ce(d_real['cls'], labels) + criterion_ce(d_fake['cls'], labels)

            optimizer_d.zero_grad()
            loss_d_cls.backward()
            optimizer_d.step()
            scheduler_d.step()

            loss_mse = F.mse_loss(prediction, clean)
            d_fake_for_g = discriminator(prediction.view(batch_size, channel_count, signal_length).unsqueeze(1))
            loss_g_cls = criterion_ce(d_fake_for_g['cls'], labels)
            loss_g = loss_mse + classification_weight * loss_g_cls

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            epoch_loss += loss_g.item()

        print(f"\r[EEGDNet+CLS][Epoch {epoch_index + 1}/{epochs}] Loss: {epoch_loss / len(eeg_train_dataloader):.6f}", end='')

    print('\nTraining finished.')
    return model


# Backward-compatible aliases.
weights_init = initialize_conv_weights
train_GCTNet = train_gctnet
Train_EEGDNet = train_eegdnet
