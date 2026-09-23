"""Downstream decoding of the denoised trials.

The point of denoising here is not a smaller waveform error but a trial that still decodes,
so the output is handed to an ordinary SSVEP decoder and scored by classification accuracy.
This module wires the two decoders that read the filter-bank tensor directly. The remaining
three used in the paper are under Models/Decoders and need their own front ends: FBCCA and
TRCA build sinusoidal references and spatial filters, CCNN a complex-spectrum feature map.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Models.Decoders.EEGNet import EEGNet
from Models.Decoders.SSVEPNet import ESNet as SSVEPNet

DECODERS = {'EEGNet': EEGNet, 'SSVEPNet': SSVEPNet}


def _loader(x, y, batch_size, shuffle):
    tensors = (torch.tensor(x, dtype=torch.float32).unsqueeze(1),
               torch.tensor(y, dtype=torch.int64))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def run(name, train_x, train_y, test_x, test_y, num_classes, device,
        epochs=100, batch_size=64, lr=1e-3):
    """Train one decoder on `train_x` and return its accuracy on `test_x`."""
    if name not in DECODERS:
        raise KeyError(f'{name} is not wired here; see the module docstring')
    channels, timesteps = train_x.shape[1], train_x.shape[2]
    model = DECODERS[name](channels, timesteps, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    loader = _loader(train_x, train_y, batch_size, True)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out['cls_logits'] if isinstance(out, dict) else out
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in _loader(test_x, test_y, batch_size, False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out['cls_logits'] if isinstance(out, dict) else out
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
