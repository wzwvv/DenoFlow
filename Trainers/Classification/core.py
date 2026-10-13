from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from Data.core import flatten_structured_eeg
from Models.Classification.DeepL import (
    CCNN, ConvCA, DDGCNN, DualBranchNet, EEGNet, FBtCNN, SSVEPNet, SSVEPformer,
)
from Models.Classification.HZTKD.Hztlayer import TDCAlayer
from Models.Classification.KNoW.CCA import CCA_Base
from Models.Classification.KNoW.FBCCA import FBCCA
from Models.Classification.KNoW.MSI import MSI_Base
from Models.Classification.KNoW.TRCA import TRCA
from Support.config import (
    get_classifier_model_config,
    get_experiment_config,
    get_knowledge_method_config,
    get_signal_config,
    use_marginal_smoothing,
)
from Support.utils import CELoss_Marginal_Smooth, Spectral_Normalization, get_Template_Signal, initialize_weights
from Trainers.Classification.diffusion import compute_itr, train_diffusion_classifier
from Trainers.Classification.hybrid import is_hybrid_method, run_single_hybrid_method


REFERENCE_METHODS = {'CCA': CCA_Base, 'MSI': MSI_Base}
TRADITIONAL_METHODS = set(REFERENCE_METHODS) | {'FBCCA', 'TRCA', 'TDCA'}
DEEP_METHODS = {'EEGNet', 'CCNN', 'FBtCNN', 'ConvCA', 'SSVEPNet', 'SSVEPformer',
                'DDGCNN', 'DualBranchNet'}
DIFFUSION_METHODS = {'DiffusionClassifier', 'DiffusionSSVEPClassifier'}


def get_downstream_method_family(method: str) -> str:
    if method in TRADITIONAL_METHODS:
        return 'traditional'
    if method in DEEP_METHODS:
        return 'deep'
    if is_hybrid_method(method):
        return 'hybrid'
    if method in DIFFUSION_METHODS:
        return 'diffusion'
    return 'extra'


class ArrayClassifierDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray):
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.labels = torch.tensor(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.eeg[index].unsqueeze(0), self.labels[index]


@dataclass
class DownstreamResult:
    acc: float
    itr: float
    extra_metrics: Dict[str, float]


def _compute_itr(meta, accuracy: float, config) -> float:
    decision_overhead = get_experiment_config(config).get('decision_overhead_sec', 0.5)
    return float(compute_itr(meta.num_classes, accuracy, meta.window_sec + decision_overhead))


def _build_reference_opt(meta, num_channels: int, num_timesteps: int):
    return type('ReferenceOpt', (), {
        'Fs': meta.sample_rate,
        'ws': num_timesteps / meta.sample_rate,
        'Nm': meta.num_filter_banks,
        'Nc': num_channels,
        'Nf': meta.num_classes,
        'Nh': meta.num_trials,
        'dataset': 'experiment',
        'is_ensemble': False,
    })()


def _reshape_for_spatial_filtering(data: np.ndarray, meta, labels: np.ndarray = None):
    """Reshape trials to class-first layout for the spatial-filtering methods.

    The reshape below assumes samples arrive in contiguous per-class blocks. That holds
    for plain splits, but NOT for sample-axis augmentation (double_batch_*), which yields
    [all classes clean][all classes generated]. Without the sort the templates are built
    from spliced classes and TRCA silently collapses to chance. Pass labels whenever they
    are available.
    """
    structured = data.astype(np.float64)
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        counts = np.bincount(labels, minlength=meta.num_classes)
        if len(counts) != meta.num_classes or counts.min() != counts.max():
            raise ValueError(
                f'Spatial-filtering methods need the same number of trials per class; '
                f'got counts {counts.tolist()}. Reshaping these would silently mix classes.'
            )
        order = np.argsort(labels, kind='stable')
        structured = structured[order]
        labels = labels[order]
    trial_first = structured
    class_first = np.transpose(
        structured.reshape(meta.num_classes, -1, meta.num_filter_banks, meta.num_channels, structured.shape[-1]),
        (0, 2, 3, 4, 1),
    )
    return trial_first, class_first, labels


def _evaluate_reference(method: str, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config) -> DownstreamResult:
    train_flat = flatten_structured_eeg(train_data).astype(np.float64)
    test_flat = flatten_structured_eeg(test_data).astype(np.float64)
    opt = _build_reference_opt(meta, train_flat.shape[1], train_flat.shape[-1])
    targets = meta.freqs
    classifier = REFERENCE_METHODS[method](opt)
    if method == 'CCA':
        _, pred = classifier.cca_classify(targets, test_flat, num_harmonics=meta.num_filter_banks)
        pred = np.array(pred).reshape(-1) - 1
    else:
        _, pred = classifier.msi_classify(targets, test_flat, num_harmonics=meta.num_filter_banks)
        pred = np.array(pred).reshape(-1) - 1
    acc = float(np.mean(pred == test_labels))
    return DownstreamResult(acc=acc, itr=_compute_itr(meta, acc, config), extra_metrics={})


def _evaluate_fbcca(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config) -> DownstreamResult:
    train_flat = flatten_structured_eeg(train_data).astype(np.float64)
    test_flat = flatten_structured_eeg(test_data).astype(np.float64)
    opt = _build_reference_opt(meta, meta.num_channels, train_flat.shape[-1])
    fbcca = FBCCA(opt)

    def dummy_filter_bank(eeg):
        return eeg.reshape(eeg.shape[0], meta.num_filter_banks, meta.num_channels, eeg.shape[-1])

    fbcca.filter_bank = dummy_filter_bank
    pred = np.array(
        fbcca.fbcca_classify(
            meta.freqs,
            test_flat,
            num_harmonics=meta.num_filter_banks,
            train_labels=train_labels,
            train_data=train_flat,
            template=True,
        )
    ).reshape(-1)
    acc = float(np.mean(pred == test_labels))
    return DownstreamResult(acc=acc, itr=_compute_itr(meta, acc, config), extra_metrics={})


def _evaluate_trca(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config) -> DownstreamResult:
    train_trials, train_class_first, train_labels_sorted = _reshape_for_spatial_filtering(train_data, meta, train_labels)
    _, test_class_first, test_labels_sorted = _reshape_for_spatial_filtering(test_data, meta, test_labels)
    classifier = TRCA(train_class_first)
    classifier.trca(train_trials, train_labels_sorted)
    acc = float(classifier.test_trca(test_class_first, test_labels_sorted.reshape(meta.num_classes, -1)))
    return DownstreamResult(acc=acc, itr=_compute_itr(meta, acc, config), extra_metrics={})


def _evaluate_tdca(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config) -> DownstreamResult:
    n_components = int(get_knowledge_method_config(config, 'TDCA').get('num_components', 3))
    train_trials, train_class_first, train_labels_sorted = _reshape_for_spatial_filtering(train_data, meta, train_labels)
    _, test_class_first, test_labels_sorted = _reshape_for_spatial_filtering(test_data, meta, test_labels)
    classifier = TDCAlayer(
        KLG_FIX=True,
        EEGDATA=train_class_first,
        channels=list(range(meta.num_channels)),
        DL=False,
        n_components=n_components,
        output=np.zeros((meta.num_classes, meta.num_classes, test_labels_sorted.reshape(meta.num_classes, -1).shape[1])),
    )
    classifier.tdca(train_trials, train_labels_sorted)
    acc = float(classifier.test_tdca(test_class_first, test_labels_sorted.reshape(meta.num_classes, -1)))
    return DownstreamResult(acc=acc, itr=_compute_itr(meta, acc, config), extra_metrics={})


def _build_deep_model(method: str, meta, config, device: str, num_timesteps: int, batch_size: int):
    flat_channels = meta.num_filter_banks * meta.num_channels
    model_cfg = get_classifier_model_config(config, method)
    if method == 'EEGNet':
        return EEGNet.EEGNet(flat_channels, num_timesteps, meta.num_classes)
    if method == 'CCNN':
        return CCNN.CNN(flat_channels, 220, meta.num_classes)
    if method == 'FBtCNN':
        return FBtCNN.tCNN(flat_channels, num_timesteps, meta.num_classes, meta.sample_rate)
    if method == 'ConvCA':
        return ConvCA.convca(flat_channels, num_timesteps, meta.num_classes)
    if method == 'SSVEPformer':
        model = SSVEPformer.SSVEPformer(
            depth=2,
            attention_kernal_length=31,
            FFT_PARAMS=[meta.sample_rate, num_timesteps / meta.sample_rate],
            chs_num=flat_channels,
            class_num=meta.num_classes,
            dropout=0.5,
            resolution=meta.frequency_step,
            start_freq=meta.start_frequency,
            end_freq=64,
        )
        model.apply(initialize_weights)
        return model
    if method == 'SSVEPNet':
        return Spectral_Normalization(SSVEPNet.ESNet(flat_channels, num_timesteps, meta.num_classes))
    if method == 'DualBranchNet':
        # Same module the adversarial framework uses as its discriminator, with the
        # domain head omitted, so the decoder reported here is the discriminator trained
        # there rather than a separate model that merely resembles it.
        return Spectral_Normalization(DualBranchNet.DualBranchNet(
            flat_channels, num_timesteps, meta.num_classes,
            sample_rate=float(meta.sample_rate)))
    if method == 'DDGCNN':
        return DDGCNN.DenseDDGCNN(
            [batch_size, num_timesteps, flat_channels],
            k_adj=3,
            num_out=int(model_cfg['num_filters']),
            dropout=0.5,
            n_blocks=3,
            nclass=meta.num_classes,
            bias=False,
            norm=model_cfg['normalization'],
            act=model_cfg['activation'],
            trans_class=model_cfg['transform_class'],
            device=device,
        )
    raise ValueError(f'Unsupported deep method: {method}')


def _prepare_deep_dataloaders(method: str, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, batch_size: int):
    train_flat = flatten_structured_eeg(train_data)
    test_flat = flatten_structured_eeg(test_data)
    train_labels_tensor = torch.tensor(train_labels.astype(np.int64))
    test_labels_tensor = torch.tensor(test_labels.astype(np.int64))

    if method == 'ConvCA':
        train_sig = torch.tensor(train_flat, dtype=torch.float32).unsqueeze(1)
        test_sig = torch.tensor(test_flat, dtype=torch.float32).unsqueeze(1)
        train_sig = torch.swapaxes(train_sig, 2, 3)
        test_sig = torch.swapaxes(test_sig, 2, 3)

        train_temp = get_Template_Signal(train_sig, meta.num_classes)
        train_temp = torch.swapaxes(train_temp, 0, 1).repeat(train_sig.shape[0], 1, 1, 1)
        train_temp = torch.swapaxes(train_temp, 1, 3)

        test_temp = get_Template_Signal(test_sig, meta.num_classes)
        test_temp = torch.swapaxes(test_temp, 0, 1).repeat(test_sig.shape[0], 1, 1, 1)
        test_temp = torch.swapaxes(test_temp, 1, 3)

        train_dataset = TensorDataset(train_sig, train_temp, train_labels_tensor)
        test_dataset = TensorDataset(test_sig, test_temp, test_labels_tensor)
    else:
        train_tensor = torch.tensor(train_flat, dtype=torch.float32).unsqueeze(1)
        test_tensor = torch.tensor(test_flat, dtype=torch.float32).unsqueeze(1)
        if method == 'CCNN':
            train_tensor = torch.from_numpy(CCNN.complex_spectrum_features(train_tensor.numpy(), FFT_PARAMS=[meta.sample_rate, meta.window_sec]))
            test_tensor = torch.from_numpy(CCNN.complex_spectrum_features(test_tensor.numpy(), FFT_PARAMS=[meta.sample_rate, meta.window_sec]))
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        test_dataset = TensorDataset(test_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def _evaluate_deep(method: str, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config, device: str) -> DownstreamResult:
    num_timesteps = train_data.shape[-1]
    model_cfg = get_classifier_model_config(config, method)
    batch_size = int(model_cfg.get('batch_size', 30))
    num_epochs = int(model_cfg['epochs'])
    ensemble_size = int(get_signal_config(config).get('ensemble_size', 1))
    lr = float(model_cfg['lr'])
    weight_decay = float(model_cfg['weight_decay'])
    use_scheduler = bool(model_cfg.get('use_scheduler', False))
    train_loader, test_loader = _prepare_deep_dataloaders(method, train_data, train_labels, test_data, test_labels, meta, batch_size)

    model_packs = []
    for _ in range(ensemble_size):
        model = _build_deep_model(method, meta, config, device, num_timesteps, batch_size).to(device)
        # DualBranchNet shares SSVEPNet's recurrent temporal path, so it gets the same
        # marginal label smoothing; comparing them under different losses would confound the
        # architecture with the training objective.
        if use_marginal_smoothing(config) and method in ('SSVEPNet', 'DualBranchNet'):
            criterion = CELoss_Marginal_Smooth(meta.num_classes, stimulus_type=40 if meta.dataset_name != 'benchmark12' else 12)
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
        if method == 'SSVEPformer':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        if method == 'DDGCNN':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(model_cfg['lr_decay_factor']),
                patience=int(model_cfg['scheduler_patience']),
                verbose=False,
                eps=1e-8,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs * max(1, len(train_loader))),
                eta_min=5e-6,
            )
        model_packs.append({'model': model, 'criterion': criterion, 'optimizer': optimizer, 'scheduler': scheduler})

    # 'best' (old protocol) keeps the highest test accuracy over the periodic evaluations;
    # 'final' evaluates once, after the last epoch, so no test trial selects the model.
    report = str(get_experiment_config(config).get('decoder_report', 'best'))
    best_acc = 0.0
    for epoch_index in range(num_epochs):
        epoch_loss = torch.tensor(0.0)
        for pack in model_packs:
            model = pack['model']
            optimizer = pack['optimizer']
            criterion = pack['criterion']
            scheduler = pack['scheduler']
            model.train()
            for batch in train_loader:
                if method == 'ConvCA':
                    x, temp, y = batch
                    x = x.float().to(device)
                    temp = temp.float().to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    logits = model(x, temp)
                else:
                    x, y = batch
                    x = x.float().to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    logits = model(x)
                loss = criterion(logits, y).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if method != 'DDGCNN' and use_scheduler:
                    scheduler.step()
                epoch_loss = loss.detach()
            if method == 'DDGCNN' and use_scheduler:
                scheduler.step(epoch_loss.detach().cpu().item())

        final_epoch = epoch_index + 1 == num_epochs
        if final_epoch if report == 'final' else ((epoch_index + 1) % 10 == 0 or final_epoch):
            total_correct = 0
            total_samples = 0
            for pack in model_packs:
                pack['model'].eval()
            with torch.no_grad():
                for batch in test_loader:
                    if method == 'ConvCA':
                        x, temp, y = batch
                        x = x.float().to(device)
                        temp = temp.float().to(device)
                        y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    else:
                        x, y = batch
                        x = x.float().to(device)
                        y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    total_logits = 0
                    for pack in model_packs:
                        if method == 'ConvCA':
                            total_logits = total_logits + pack['model'](x, temp)
                        else:
                            total_logits = total_logits + pack['model'](x)
                    predictions = (total_logits / ensemble_size).argmax(dim=-1)
                    total_correct += (predictions == y).sum().item()
                    total_samples += y.shape[0]
            current_acc = total_correct / max(total_samples, 1)
            best_acc = current_acc if report == 'final' else max(best_acc, current_acc)

    return DownstreamResult(acc=float(best_acc), itr=_compute_itr(meta, best_acc, config), extra_metrics={})


def _evaluate_hybrid(
    method: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    meta,
    config,
    device: str,
    knowledge_train_data: Optional[np.ndarray] = None,
    knowledge_train_labels: Optional[np.ndarray] = None,
    knowledge_test_data: Optional[np.ndarray] = None,
    knowledge_test_labels: Optional[np.ndarray] = None,
) -> DownstreamResult:
    train_flat = flatten_structured_eeg(train_data)
    test_flat = flatten_structured_eeg(test_data)
    knowledge_train_flat = None if knowledge_train_data is None else flatten_structured_eeg(knowledge_train_data)
    knowledge_test_flat = None if knowledge_test_data is None else flatten_structured_eeg(knowledge_test_data)
    acc = float(
        run_single_hybrid_method(
            config,
            device,
            *method.split('+', 1),
            train_flat,
            train_labels,
            test_flat,
            test_labels,
            knowledge_train_data=knowledge_train_flat,
            knowledge_train_labels=knowledge_train_labels,
            knowledge_test_data=knowledge_test_flat,
            knowledge_test_labels=knowledge_test_labels,
        )
    )
    return DownstreamResult(acc=acc, itr=_compute_itr(meta, acc, config), extra_metrics={})


def _evaluate_diffusion_classifier(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, meta, config, device: str) -> DownstreamResult:
    train_flat = flatten_structured_eeg(train_data)
    test_flat = flatten_structured_eeg(test_data)
    train_dataset = ArrayClassifierDataset(train_flat, train_labels)
    test_dataset = ArrayClassifierDataset(test_flat, test_labels)
    acc, _ = train_diffusion_classifier(train_dataset, test_dataset, meta, config, device)
    return DownstreamResult(acc=float(acc), itr=_compute_itr(meta, float(acc), config), extra_metrics={})


def evaluate_downstream(
    method: str,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    meta,
    config,
    device: str,
    knowledge_train_data: Optional[np.ndarray] = None,
    knowledge_train_labels: Optional[np.ndarray] = None,
    knowledge_test_data: Optional[np.ndarray] = None,
    knowledge_test_labels: Optional[np.ndarray] = None,
) -> DownstreamResult:
    if method in REFERENCE_METHODS:
        return _evaluate_reference(method, train_data, train_labels, test_data, test_labels, meta, config)
    if method == 'FBCCA':
        return _evaluate_fbcca(train_data, train_labels, test_data, test_labels, meta, config)
    if method == 'TRCA':
        return _evaluate_trca(train_data, train_labels, test_data, test_labels, meta, config)
    if method == 'TDCA':
        return _evaluate_tdca(train_data, train_labels, test_data, test_labels, meta, config)
    if method in DEEP_METHODS:
        return _evaluate_deep(method, train_data, train_labels, test_data, test_labels, meta, config, device)
    if is_hybrid_method(method):
        return _evaluate_hybrid(
            method,
            train_data,
            train_labels,
            test_data,
            test_labels,
            meta,
            config,
            device,
            knowledge_train_data=knowledge_train_data,
            knowledge_train_labels=knowledge_train_labels,
            knowledge_test_data=knowledge_test_data,
            knowledge_test_labels=knowledge_test_labels,
        )
    if method in DIFFUSION_METHODS:
        return _evaluate_diffusion_classifier(train_data, train_labels, test_data, test_labels, meta, config, device)
    raise ValueError(f'Unsupported downstream method: {method}')
