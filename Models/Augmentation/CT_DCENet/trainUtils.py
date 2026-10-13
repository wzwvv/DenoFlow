import os.path

import torch
from matplotlib import pyplot as plt
from torch import nn
from collections import defaultdict
from evaluateUtils import evaluate, Cal_SNR
import pytorch_warmup as warmup
from torch.nn import functional as F

from lossUtils import collaborative_loss
from modelUtils import load_models, model_structure

import numpy as np

def train_individual(
    train_iter,
    val_iter,
    save_dir,
    configs,
    signal_len=512,
    model=None
):
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = configs.num_individual_epochs
    device = configs.device
    lr = configs.lr
    betas = configs.betas
    eps = configs.eps
    milestones = configs.milestones
    head_type_list = configs.head_type_list
    loss_list = configs.loss_list
    beta_height = configs.beta_height
    block_num = configs.block_num
    ablation = configs.ablation

    if model is None:
        model = load_models(
            save_dir,
            head_type_list,
            ablation=ablation,
            signal_len=signal_len,
            block_num=block_num
        )
        model.to(device)

    # 冻结 sl / supplement / beta
    for name, param in model.named_parameters():
        param.requires_grad = not (
            name.startswith('sl')
            or name.startswith('supplement')
            or name.startswith('beta')
        )

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=milestones, gamma=0.1
    )
    warmup_scheduler = warmup.LinearWarmup(opt, warmup_period=10)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_individual_epoch(
            model,
            train_iter,
            loss_list,
            beta_height,
            opt,
            scheduler,
            warmup_scheduler,
            device
        )

        val_loss, snr_list = val_individual_epoch(
            model,
            val_iter,
            loss_list,
            beta_height,
            device
        )

        print(
            f"[Individual] Epoch {epoch:03d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            + " ".join([f"SNR{i}:{s:.2f}" for i, s in enumerate(snr_list)])
        )

    # ✅ 只保存最终模型
    torch.save(model.state_dict(), f"{save_dir}/individual_final.pth")

    return model


def train_individual_epoch(
    model,
    train_iter,
    loss_list,
    beta_height,
    opt,
    scheduler,
    warmup_scheduler,
    device
):
    model.train()
    losses = []

    for eegc, eegp in train_iter:
        eegc, eegp = eegc.to(device), eegp.to(device)

        std = eegc.std(dim=1, keepdims=True)
        eegc = eegc / std
        eegp = eegp / std

        fl_out_list = model(eegc)

        loss = collaborative_loss(
            0, fl_out_list, eegp, loss_list[0], beta_height
        )
        for i in range(1, len(fl_out_list)):
            loss += collaborative_loss(
                i, fl_out_list, eegp, loss_list[i], beta_height
            )

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.detach())

    with warmup_scheduler.dampening():
        scheduler.step()

    return torch.stack(losses).mean().item()
def val_individual_epoch(
    model,
    val_iter,
    loss_list,
    beta_height,
    device
):
    model.eval()
    losses = []
    snr_dict = defaultdict(list)

    with torch.no_grad():
        for eegc, eegp in val_iter:
            eegc, eegp = eegc.to(device), eegp.to(device)

            std = eegc.std(dim=1, keepdims=True)
            eegc = eegc / std
            eegp = eegp / std

            fl_out_list = model(eegc)

            loss = collaborative_loss(
                0, fl_out_list, eegp, loss_list[0], beta_height
            )
            snr_dict[0].append(
                Cal_SNR(fl_out_list[0], eegp, reduction='none')
            )

            for i in range(1, len(fl_out_list)):
                loss += collaborative_loss(
                    i, fl_out_list, eegp, loss_list[i], beta_height
                )
                snr_dict[i].append(
                    Cal_SNR(fl_out_list[i], eegp, reduction='none')
                )

            losses.append(loss.detach())

    snr_list = [
        torch.cat(v, dim=0).mean()
        for v in snr_dict.values()
    ]

    return torch.stack(losses).mean().item(), snr_list
def train_ensemble(
    train_iter,
    val_iter,
    save_dir,
    configs,
    model=None
):
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = configs.num_ensemble_epochs
    device = configs.device
    lr = configs.lr
    betas = configs.betas
    eps = configs.eps
    milestones = configs.milestones
    block_num = configs.block_num
    ablation = configs.ablation

    if model is None:
        model = load_models(
            save_dir,
            configs.head_type_list,
            block_num=block_num
        )
        model.to(device)

    model.feature_learner.eval()
    model.heads.eval()

    for name, param in model.named_parameters():
        param.requires_grad = (
            name.startswith('sl')
            or name.startswith('supplement')
            or name.startswith('beta')
        )

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=milestones, gamma=0.1
    )
    warmup_scheduler = warmup.LinearWarmup(opt, warmup_period=10)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_ensemble_epoch(
            model,
            train_iter,
            opt,
            scheduler,
            warmup_scheduler,
            device,
            ablation
        )

        val_loss = val_ensemble_epoch(
            model,
            val_iter,
            device,
            ablation
        )

        print(
            f"[Ensemble] Epoch {epoch:03d} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f}"
        )

    # ✅ 只保存最终 ensemble 模型
    torch.save(model.state_dict(), f"{save_dir}/ensemble_final.pth")

def train_ensemble_epoch(model, train_iter,opt,scheduler,warmup_scheduler,device,ablation):
    model.sl.train()
    if not ablation:
        model.supplement.train()
    train_l_list = []
    for eegc,eegp in train_iter:#(N,L)污染EEG 干净EEG
        eegc,eegp = eegc.to(device),eegp.to(device) #(N,L),(N,L)
        # 数据增强
        # EEG时域信号内部单独做标准化(既不考虑其他信号，等价于LayerNorm)，并且均值使用经验常数0
        std = eegc.std(dim=1, keepdims=True)
        eegc_view = eegc / std
        eegp_view = eegp / std
        # 前向传播计算损失
        pred_eegp = model.denoise(eegc_view)
        l = F.mse_loss(pred_eegp,eegp_view)
        # 反向传播更新参数
        opt.zero_grad()  # 梯度置为0
        l.backward() #反向传播更新参数
        opt.step()
        train_l_list.append(l.detach())
    with warmup_scheduler.dampening():
        scheduler.step()
    train_loss = torch.stack(train_l_list).mean().item()
    return train_loss

def val_ensemble_epoch(model, val_iter,device,ablation):
    model.eval()
    val_l_list = []
    with torch.no_grad():
        for eegc,eegp in val_iter:#(N,L)污染EEG 干净EEG
            eegc,eegp = eegc.to(device),eegp.to(device) #(N,L),(N,L)
            # EEG时域信号内部单独做标准化(既不考虑其他信号，等价于LayerNorm)，并且均值使用经验常数0
            std = eegc.std(dim=1, keepdims=True)
            eegc = eegc / std
            eegp = eegp / std
            # 前向传播计算损失
            pred_eegp = model.denoise(eegc)
            l = F.mse_loss(pred_eegp,eegp)
            # 反向传播更新参数
            val_l_list.append(l.detach())
        val_loss = torch.stack(val_l_list).mean().item()
    return val_loss