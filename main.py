"""DenoFlow: train the velocity field on one subject and score the denoised trials.

    python main.py --dataset benchmark --subject 1 --fold 0 --artifact both

Everything not given on the command line comes from etc/config.yaml.
"""
import argparse
import os

import numpy as np
import torch
import yaml

from Models.DenoFlow import FlowDenoiser
from Utils import decode, trainer
from Utils.dataset import META, build_views, fold_indices, load_subject, make_loaders
from Utils.inject import set_random_seed
from Utils.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='etc/config.yaml')
    p.add_argument('--dataset', choices=sorted(META), default=None)
    p.add_argument('--subject', type=int, default=None)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--artifact', choices=['emg', 'eog', 'both'], default=None)
    p.add_argument('--ratio', type=float, default=None, help='mask fraction')
    p.add_argument('--snr', type=float, default=None, help='target ratio gamma, in dB')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--decode', action='store_true',
                   help='also train downstream decoders on the denoised trials')
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)

    for key in ('dataset', 'subject', 'fold', 'artifact', 'ratio', 'snr', 'epochs'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    device = torch.device(args.device or cfg.get('device')
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(int(cfg.get('seed', 42)))

    meta = META[cfg['dataset']]
    meta.num_banks = int(cfg.get('num_banks', meta.num_banks))
    root = cfg['dataset_root'][cfg['dataset']]

    eeg, labels = load_subject(meta.name, root, int(cfg['subject']))
    clean, noisy, _ = build_views(eeg, meta, cfg['artifact'], cfg.get('mode', 'time'),
                                  float(cfg['ratio']), float(cfg['snr']))
    train_idx, test_idx = fold_indices(meta, int(cfg['fold']),
                                       float(cfg.get('train_ratio', 0.8)))
    train_loader, test_loader = make_loaders(clean, noisy, labels, train_idx, test_idx,
                                             int(cfg.get('batch_size', 36)))

    print(f"{meta.name} S{cfg['subject']} fold {cfg['fold']} | artifact {cfg['artifact']} "
          f"| mask {cfg['ratio']} | gamma {cfg['snr']} dB | Nc {meta.flat_channels} "
          f"| T {meta.num_timesteps} | device {device}", flush=True)

    model = FlowDenoiser(meta.flat_channels, meta.num_timesteps,
                         dropout=float(cfg.get('dropout', 0.5)), num_classes=0,
                         steps=int(cfg.get('flow_steps', 4)),
                         arch={'conv_order': 'alt', 'bands': max(meta.num_banks, 1),
                               **cfg.get('arch', {})})
    model = trainer.train(model, train_loader, test_loader, device, meta.flat_channels,
                          meta.num_timesteps, meta.num_classes, cfg)

    denoised = trainer.denoise(model, noisy[test_idx], device,
                               steps=int(cfg.get('flow_steps', 4)))
    scores = compute_metrics(clean[test_idx], denoised, noisy[test_idx])
    print(f"  fidelity  CC {scores['cc']:.4f}  RRMSE {scores['rrmse']:.4f}  "
          f"SNRI {scores['snri']:.2f} dB", flush=True)

    if args.decode:
        train_denoised = trainer.denoise(model, noisy[train_idx], device,
                                         steps=int(cfg.get('flow_steps', 4)))
        for name in cfg.get('decoders', ['EEGNet', 'SSVEPNet']):
            acc = decode.run(name, train_denoised, labels[train_idx], denoised,
                             labels[test_idx], meta.num_classes, device,
                             epochs=int(cfg.get('decoder_epochs', 100)))
            print(f"  {name:<10} accuracy {100 * acc:.2f}%", flush=True)

    out_dir = cfg.get('output_dir', 'result')
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{meta.name}_S{cfg['subject']}_fold{cfg['fold']}_{cfg['artifact']}"
    np.save(os.path.join(out_dir, f'{tag}_denoised.npy'), denoised)
    torch.save(model.state_dict(), os.path.join(out_dir, f'{tag}_field.pt'))
    print(f'  saved to {out_dir}/{tag}_*', flush=True)


if __name__ == '__main__':
    main()
