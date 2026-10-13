import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Augmentation import discriminators as Discriminator
from Models.Augmentation.diffusion_generator import DiffusionGeneratorWrapper, EEGDiffusionDenoiser, NoiseScheduler
from Support.utils import finish_progress_line, print_progress_line


def _build_diffusion_kwargs(meta, method_cfg):
    return {
        'num_signal_channels': meta.num_filter_banks * meta.num_channels,
        'num_timesteps': meta.num_timesteps,
        'num_classes': meta.num_classes,
        'sample_rate': meta.sample_rate,
        'freqs': meta.freqs,
        'num_harmonics': int(method_cfg.get('num_harmonics', 5)),
        'dropout': float(method_cfg.get('dropout', 0.5)),
        'use_reference_condition': bool(method_cfg.get('use_reference_condition', True)),
        'use_class_mean_condition': bool(method_cfg.get('use_class_mean_condition', True)),
    }


def _compute_class_mean_bank(train_loader, num_classes, flat_channels, num_timesteps):
    sum_bank = torch.zeros(num_classes, flat_channels, num_timesteps, dtype=torch.float32)
    count_bank = torch.zeros(num_classes, dtype=torch.float32)

    if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'tensors'):
        clean_tensor, _source_tensor, labels_tensor = train_loader.dataset.tensors
        clean_batches = [(clean_tensor.float().cpu(), torch.as_tensor(labels_tensor.reshape(labels_tensor.shape[0]), dtype=torch.long))]
    else:
        clean_batches = []
        for clean, _source, labels in train_loader:
            clean_batches.append((clean.float().cpu(), torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.long)))

    for clean, labels in clean_batches:
        for class_index in range(num_classes):
            mask = labels == class_index
            if not mask.any():
                continue
            sum_bank[class_index] += clean[mask].sum(dim=0).squeeze(0)
            count_bank[class_index] += float(mask.sum().item())

    total_count = count_bank.sum().clamp_min(1.0)
    global_mean = sum_bank.sum(dim=0) / total_count
    class_mean_bank = global_mean.unsqueeze(0).repeat(num_classes, 1, 1)
    for class_index in range(num_classes):
        if count_bank[class_index] > 0:
            class_mean_bank[class_index] = sum_bank[class_index] / count_bank[class_index]
    return class_mean_bank


def _build_wrapper(meta, method_cfg, denoiser, class_mean_bank, device):
    num_diffusion_steps = int(method_cfg.get('num_diffusion_steps', 200))
    beta_start = float(method_cfg.get('beta_start', 1e-4))
    beta_end = float(method_cfg.get('beta_end', 0.02))
    inference_steps = int(method_cfg.get('inference_steps', 50))
    wrapper = DiffusionGeneratorWrapper(
        **_build_diffusion_kwargs(meta, method_cfg),
        T=num_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        inference_steps=inference_steps,
        use_ddim=bool(method_cfg.get('use_ddim', True)),
    ).to(device)
    wrapper.denoiser = denoiser
    wrapper.set_class_mean_bank(class_mean_bank)
    return wrapper


def train_diffusion_generator(eeg_train_dataloader, eeg_test_dataloader, meta, method_cfg, device):
    flat_channels = meta.num_filter_banks * meta.num_channels
    num_timesteps = meta.num_timesteps
    num_classes = meta.num_classes
    epochs = int(method_cfg.get('epochs', 300))
    num_diffusion_steps = int(method_cfg.get('num_diffusion_steps', 200))
    beta_start = float(method_cfg.get('beta_start', 1e-4))
    beta_end = float(method_cfg.get('beta_end', 0.02))
    lr = float(method_cfg.get('lr', 0.001))
    weight_decay = float(method_cfg.get('weight_decay', 0.0001))

    denoiser = EEGDiffusionDenoiser(**_build_diffusion_kwargs(meta, method_cfg)).to(device)
    class_mean_bank = _compute_class_mean_bank(eeg_train_dataloader, num_classes, flat_channels, num_timesteps).to(device)
    denoiser.set_class_mean_bank(class_mean_bank)
    scheduler = NoiseScheduler(T=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(flat_channels, num_timesteps, num_classes)
    ).to(device)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, weight_decay=weight_decay)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * max(1, len(eeg_train_dataloader))),
        eta_min=5e-6,
    )
    criterion_ce = nn.CrossEntropyLoss()

    val_acc_fake_cls = torch.tensor(0.0)
    val_acc_real_cls = torch.tensor(0.0)

    for epoch_index in range(epochs):
        denoiser.train()
        discriminator.train()
        epoch_loss = 0.0

        for clean, _source, labels in eeg_train_dataloader:
            clean = clean.float().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64, device=device)
            batch_size = clean.shape[0]

            t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device, dtype=torch.long)
            x_t, noise = scheduler.q_sample(clean, t)
            predicted_noise = denoiser(x_t, t, labels)
            loss_diff = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss_diff.backward()
            optimizer.step()
            scheduler_lr.step()
            epoch_loss += loss_diff.item()

            with torch.no_grad():
                x0_pred = scheduler.predict_x0(x_t, t, predicted_noise.detach())

            d_real = discriminator(clean.detach())
            d_fake = discriminator(x0_pred.detach())
            loss_d = criterion_ce(d_real['cls'], labels) + criterion_ce(d_fake['cls'], labels)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        if (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs:
            denoiser.eval()
            discriminator.eval()
            wrapper = _build_wrapper(meta, method_cfg, denoiser, class_mean_bank, device)

            sum_acc_fake_cls = 0.0
            sum_acc_real_cls = 0.0
            dataset_length = 0

            with torch.no_grad():
                for clean, _source, labels in eeg_test_dataloader:
                    clean = clean.float().to(device)
                    labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64, device=device)
                    dataset_length += len(labels)

                    _, _, _, _, output = wrapper(labels)
                    d_fake = discriminator(output)
                    d_real = discriminator(clean)
                    sum_acc_fake_cls += (d_fake['cls'].argmax(dim=1) == labels).float().sum()
                    sum_acc_real_cls += (d_real['cls'].argmax(dim=1) == labels).float().sum()

            val_acc_fake_cls = sum_acc_fake_cls / max(dataset_length, 1)
            val_acc_real_cls = sum_acc_real_cls / max(dataset_length, 1)
            print_progress_line(
                'aug-diff',
                epoch_index + 1,
                epochs,
                diff=epoch_loss / max(1, len(eeg_train_dataloader)),
                fake=val_acc_fake_cls,
                real=val_acc_real_cls,
            )

    finish_progress_line(
        f"[aug-diff] final_fake={float(val_acc_fake_cls):.4f} final_real={float(val_acc_real_cls):.4f} "
        f"time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    wrapper = _build_wrapper(meta, method_cfg, denoiser, class_mean_bank, device)
    return val_acc_fake_cls, val_acc_real_cls, wrapper, discriminator


def train_diffusion_gan_generator(eeg_train_dataloader, eeg_test_dataloader, meta, method_cfg, device):
    flat_channels = meta.num_filter_banks * meta.num_channels
    num_timesteps = meta.num_timesteps
    num_classes = meta.num_classes
    epochs = int(method_cfg.get('epochs', 300))
    num_diffusion_steps = int(method_cfg.get('num_diffusion_steps', 200))
    beta_start = float(method_cfg.get('beta_start', 1e-4))
    beta_end = float(method_cfg.get('beta_end', 0.02))
    inference_steps = int(method_cfg.get('inference_steps', 50))
    lr = float(method_cfg.get('lr', 0.001))
    weight_decay = float(method_cfg.get('weight_decay', 0.0001))
    classification_weight = float(method_cfg.get('classification_weight', 1.0))
    discriminator_weight = float(method_cfg.get('discriminator_weight', 1.0))
    discriminator_lr = float(method_cfg.get('discriminator_lr', 0.001))
    discriminator_weight_decay = float(method_cfg.get('discriminator_weight_decay', 0.0001))
    gan_interval = int(method_cfg.get('gan_interval', 5))
    train_ddim_steps = min(inference_steps, 20)

    denoiser = EEGDiffusionDenoiser(**_build_diffusion_kwargs(meta, method_cfg)).to(device)
    class_mean_bank = _compute_class_mean_bank(eeg_train_dataloader, num_classes, flat_channels, num_timesteps).to(device)
    denoiser.set_class_mean_bank(class_mean_bank)
    scheduler = NoiseScheduler(T=num_diffusion_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    discriminator = Discriminator.Spectral_Normalization(
        Discriminator.ESNet(flat_channels, num_timesteps, num_classes)
    ).to(device)

    optimizer_g = torch.optim.Adam(denoiser.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, weight_decay=discriminator_weight_decay)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=max(1, epochs * max(1, len(eeg_train_dataloader))),
        eta_min=5e-6,
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d,
        T_max=max(1, epochs * max(1, len(eeg_train_dataloader))),
        eta_min=5e-6,
    )
    criterion_ce = nn.CrossEntropyLoss()
    safe_log = lambda tensor: torch.log(tensor + 1e-8)

    global_step = 0
    val_acc_adv = torch.tensor(0.0)
    val_acc_fake_cls = torch.tensor(0.0)
    val_acc_real_cls = torch.tensor(0.0)

    for epoch_index in range(epochs):
        denoiser.train()
        discriminator.train()
        epoch_loss_diff = 0.0
        epoch_loss_cls = 0.0

        for clean, _source, labels in eeg_train_dataloader:
            clean = clean.float().to(device)
            labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64, device=device)
            batch_size = clean.shape[0]

            t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device, dtype=torch.long)
            x_t, noise = scheduler.q_sample(clean, t)
            predicted_noise = denoiser(x_t, t, labels)
            loss_diff = F.mse_loss(predicted_noise, noise)
            loss_cls_g = torch.zeros((), device=device)

            if global_step % gan_interval == 0:
                denoiser.eval()
                with torch.no_grad():
                    generated = scheduler.ddim_sample(denoiser, clean.shape, labels, inference_steps=train_ddim_steps)
                denoiser.train()

                optimizer_d.zero_grad()
                d_real = discriminator(clean.detach())
                d_fake = discriminator(generated.detach())
                loss_d_adv = -safe_log(d_real['adv']).mean() - safe_log(1 - d_fake['adv']).mean()
                loss_d_feat = -safe_log(d_real['cls'][range(len(labels)), labels]).mean() - safe_log(d_fake['cls'][range(len(labels)), labels]).mean()
                loss_d = loss_d_adv + discriminator_weight * loss_d_feat
                loss_d.backward()
                optimizer_d.step()
                scheduler_d.step()

                t_near_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                x_t_0, _ = scheduler.q_sample(clean, t_near_zero)
                pred_noise_0 = denoiser(x_t_0, t_near_zero, labels)
                x0_pred_for_g = scheduler.predict_x0(x_t_0, t_near_zero, pred_noise_0)
                d_gen = discriminator(x0_pred_for_g)
                loss_g_adv = -safe_log(d_gen['adv']).mean()
                loss_g_cls = criterion_ce(d_gen['cls'], labels)
                loss_cls_g = loss_g_adv + classification_weight * loss_g_cls

            loss_g = loss_diff + loss_cls_g
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            scheduler_g.step()
            epoch_loss_diff += loss_diff.item()
            epoch_loss_cls += loss_cls_g.item()
            global_step += 1

        if (epoch_index + 1) % 10 == 0 or epoch_index + 1 == epochs:
            denoiser.eval()
            discriminator.eval()
            wrapper = _build_wrapper(meta, method_cfg, denoiser, class_mean_bank, device)

            sum_acc_fake_cls = 0.0
            sum_acc_real_cls = 0.0
            sum_acc_adv = 0.0
            dataset_length = 0

            with torch.no_grad():
                for clean, _source, labels in eeg_test_dataloader:
                    clean = clean.float().to(device)
                    labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64, device=device)
                    dataset_length += len(labels)

                    _, _, _, _, output = wrapper(labels)
                    d_fake = discriminator(output)
                    d_real = discriminator(clean)
                    sum_acc_adv += (d_real['adv'] >= 0.5).float().sum() + (d_fake['adv'] < 0.5).float().sum()
                    sum_acc_fake_cls += (d_fake['cls'].argmax(dim=1) == labels).float().sum()
                    sum_acc_real_cls += (d_real['cls'].argmax(dim=1) == labels).float().sum()

            val_acc_adv = sum_acc_adv / max(dataset_length, 1) / 2
            val_acc_fake_cls = sum_acc_fake_cls / max(dataset_length, 1)
            val_acc_real_cls = sum_acc_real_cls / max(dataset_length, 1)
            print_progress_line(
                'aug-diff-gan',
                epoch_index + 1,
                epochs,
                diff=epoch_loss_diff / max(1, len(eeg_train_dataloader)),
                cls=epoch_loss_cls / max(1, len(eeg_train_dataloader)),
                adv=val_acc_adv,
                fake=val_acc_fake_cls,
                real=val_acc_real_cls,
            )

    finish_progress_line(
        f"[aug-diff-gan] final_adv={float(val_acc_adv):.4f} final_fake={float(val_acc_fake_cls):.4f} "
        f"final_real={float(val_acc_real_cls):.4f} time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    wrapper = _build_wrapper(meta, method_cfg, denoiser, class_mean_bank, device)
    return val_acc_fake_cls, val_acc_real_cls, wrapper, discriminator


train_diffusion = train_diffusion_generator
train_diffusion_gan = train_diffusion_gan_generator

