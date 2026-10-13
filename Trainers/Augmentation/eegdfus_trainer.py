"""Train EEGDfus on the filter-bank rows of one split and denoise the training and test trials.

Every row (one electrode in one sub-band) is an independent single-channel segment. Rows are scaled
by the standard deviation of the contaminated row and the output is scaled back. Training follows
the official utils.train: Adam, batches of 512 segments with drop_last, gradient clipping at 1, and
a step decay by 0.1, placed here at the original 1500/4000 fraction of the epoch budget. No
classification loss is added: the objective regresses the diffusion noise of individual rows, so
no denoised trial exists during training to be classified.
"""
import time

import numpy as np
import torch

from Models.Augmentation.EEGDfus.eegdfus import EEGDfus


def _rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1, x.shape[-1])


@torch.no_grad()
def _denoise_rows(model, noisy, device, batch):
    out = []
    for start in range(0, noisy.shape[0], batch):
        chunk = torch.from_numpy(noisy[start:start + batch]).unsqueeze(1).to(device)
        out.append(model.denoise(chunk).squeeze(1).cpu().numpy())
    return np.concatenate(out, axis=0)


def eegdfus_denoise(signals, method_cfg, device):
    epochs = int(method_cfg.get("epochs", 2000))
    batch = int(method_cfg.get("batch_size", 512))
    lr = float(method_cfg.get("lr", 1e-3))
    infer_batch = int(method_cfg.get("inference_batch_size", 2048))
    shape_tr, shape_te = signals.source.train.shape, signals.source.test.shape
    seq_len = shape_tr[-1]

    noisy_tr, clean_tr, noisy_te = _rows(signals.source.train), _rows(signals.clean.train), _rows(signals.source.test)
    std_tr = noisy_tr.std(axis=1, keepdims=True) + 1e-6
    std_te = noisy_te.std(axis=1, keepdims=True) + 1e-6
    x_noisy = torch.from_numpy(noisy_tr / std_tr).unsqueeze(1).to(device)
    x_clean = torch.from_numpy(clean_tr / std_tr).unsqueeze(1).to(device)

    model = EEGDfus(seq_len, feats=int(method_cfg.get("feats", 64)),
                    num_steps=int(method_cfg.get("num_steps", 500)),
                    beta_start=float(method_cfg.get("beta_start", 1e-4)),
                    beta_end=float(method_cfg.get("beta_end", 0.02))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, round(epochs * 1500 / 4000)), gamma=0.1)

    n = x_noisy.shape[0]
    t0, last = time.time(), float("nan")
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for start in range(0, n - batch + 1, batch):          # drop_last, as in the official loader
            idx = order[start:start + batch]
            optimizer.zero_grad()
            loss = model.p_losses(x_clean[idx], x_noisy[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        scheduler.step()
        last = total / max(steps, 1)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs:
            print(f"[EEGDfus] epoch {epoch + 1}/{epochs} loss/segment {last / batch:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    t1 = time.time()
    den_tr = _denoise_rows(model, (noisy_tr / std_tr).astype(np.float32), device, infer_batch) * std_tr
    den_te = _denoise_rows(model, (noisy_te / std_te).astype(np.float32), device, infer_batch) * std_te
    print(f"[EEGDfus] sampling {n + noisy_te.shape[0]} rows x {model.num_steps} steps took "
          f"{time.time() - t1:.0f}s", flush=True)
    metrics = {"eegdfus_final_loss_per_segment": float(last / batch), "eegdfus_epochs": float(epochs),
               "eegdfus_train_seconds": float(t1 - t0)}
    return den_tr.reshape(shape_tr).astype(np.float32), den_te.reshape(shape_te).astype(np.float32), metrics
