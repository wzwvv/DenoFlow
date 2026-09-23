"""Training loop for DenoFlow.

Each mini-batch draws a fresh path position and a fresh artifact segment, so neither the
position nor the contamination is fixed across epochs. Three terms make up the objective:

    L = w_flow * ||f(x_t, X~', t) - (X~ - X~')||^2 / (Nc T)     velocity regression
      + w_rec  * ||X^ - X~||^2 / (Nc T)                         reconstruction
      + w_cls  * CE(C(X^), y)                                   class

with X^ the output of S_tr Euler steps. The classifier is updated first, on the reference and
on a detached copy of the integrated output, so the class term reaches the field network only
through L.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Classifier import ESNet, Spectral_Normalization


def build_classifier(flat_channels, num_timesteps, num_classes, device):
    return Spectral_Normalization(
        ESNet(flat_channels, num_timesteps, num_classes)).to(device)


def train(model, train_loader, test_loader, device, flat_channels, num_timesteps,
          num_classes, cfg):
    epochs = int(cfg.get('epochs', 300))
    train_steps = int(cfg.get('train_sample_steps', 4))
    w_flow = float(cfg.get('flow_weight', 1.0))
    w_rec = float(cfg.get('sample_weight', 1.0))
    w_cls = float(cfg.get('classification_weight', 1.0))
    grad_clip = float(cfg.get('grad_clip', 1.0))
    eval_interval = int(cfg.get('eval_interval', 10))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(cfg.get('generator_lr', 1e-3)),
                                 weight_decay=float(cfg.get('generator_weight_decay', 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * max(1, len(train_loader))), eta_min=5e-6)

    classifier = optimizer_c = None
    if w_cls > 0:
        classifier = build_classifier(flat_channels, num_timesteps, num_classes, device)
        optimizer_c = torch.optim.Adam(
            classifier.parameters(), lr=float(cfg.get('discriminator_lr', 1e-3)),
            weight_decay=float(cfg.get('discriminator_weight_decay', 1e-4)))

    for epoch in range(epochs):
        model.train()
        if classifier is not None:
            # the classifier holds an LSTM, and cudnn refuses a backward pass on one left
            # in eval mode by the periodic evaluation below
            classifier.train()

        for clean, noisy, labels in train_loader:
            clean = clean.float().to(device)
            noisy = noisy.float().to(device)
            labels = labels.reshape(-1).to(device)

            # the field network is never conditioned on the label: at test time the label is
            # what the decoders are asked for, so conditioning would put the answer in the signal
            predicted, target = model.velocity_target(clean, noisy, None)
            loss = w_flow * F.mse_loss(predicted, target)

            generated = model.rollout(noisy, None, train_steps)
            if w_rec > 0:
                loss = loss + w_rec * F.mse_loss(generated, clean)

            if classifier is not None:
                loss_c = (F.cross_entropy(classifier(clean.detach())['cls_logits'], labels)
                          + F.cross_entropy(classifier(generated.detach())['cls_logits'], labels))
                optimizer_c.zero_grad()
                (w_cls * loss_c).backward()
                optimizer_c.step()
                loss = loss + w_cls * F.cross_entropy(
                    classifier(generated)['cls_logits'], labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

        if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
            mse = evaluate(model, test_loader, device)
            print(f'  epoch {epoch + 1:3d}/{epochs}  test mse {mse:.5f}', flush=True)

    model.classifier_state = classifier.state_dict() if classifier is not None else None
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for clean, noisy, _ in loader:
        clean, noisy = clean.float().to(device), noisy.float().to(device)
        generated = model.sample(noisy, None)
        total += F.mse_loss(generated, clean, reduction='sum').item()
        count += clean.numel()
    return total / max(count, 1)


@torch.no_grad()
def denoise(model, noisy, device, steps=None, batch_size=64):
    """Integrate the field forward from a set of contaminated trials."""
    model.eval()
    flat = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)
    out = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start:start + batch_size].to(device)
        out.append(model.sample(chunk, None, steps).squeeze(1).cpu().numpy())
    return torch.cat([torch.tensor(o) for o in out]).numpy()
