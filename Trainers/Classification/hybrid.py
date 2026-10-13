import time

import numpy as np
import torch
import torch.nn as nn

from Models.Classification.DeepL import CCNN, DDGCNN, EEGNet, FBtCNN, SSVEPNet, SSVEPformer
from Models.Classification.DeepL import EnhancedSSVEPformer as EnhSSVEPformerModule
from Models.Classification.HZTKD.Hztlayer import KLGlayer
from Support.config import (
    get_active_dataset_config,
    get_classifier_model_config,
    get_knowledge_method_config,
    get_signal_config,
    use_marginal_smoothing,
)
from Support.utils import (
    CELoss_Marginal_Smooth,
    Spectral_Normalization,
    finish_progress_line,
    initialize_weights,
    print_progress_line,
)


SUPPORTED_KNOWLEDGE_METHODS = {'TRCA', 'TDCA', 'TD-RCA'}
SUPPORTED_DEEP_METHODS = {'EEGNet', 'CCNN', 'FBtCNN', 'SSVEPNet', 'SSVEPformer', 'DDGCNN', 'EnhSSVEPformer'}


def is_hybrid_method(method_name):
    if '+' not in method_name:
        return False
    knowledge_method, deep_method = method_name.split('+', 1)
    return knowledge_method in SUPPORTED_KNOWLEDGE_METHODS and deep_method in SUPPORTED_DEEP_METHODS


def parse_hybrid_method(method_name):
    if not is_hybrid_method(method_name):
        raise ValueError(f'Unsupported hybrid method: {method_name}')
    return method_name.split('+', 1)


class HybridKnowledgeBackbone(nn.Module):
    def __init__(self, knowledge_layer, backbone, num_filter_banks, num_channels):
        super().__init__()
        self.knowledge_layer = knowledge_layer
        self.backbone = backbone
        self.num_filter_banks = num_filter_banks
        self.num_channels = num_channels

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            knowledge_x = x[0]
        else:
            knowledge_x = x
        if knowledge_x.ndim != 4 or knowledge_x.shape[1] != 1:
            raise ValueError(f'Expected knowledge_x to have shape (B, 1, C, T), got {tuple(knowledge_x.shape)}')
        batch_size, _, _, num_timesteps = knowledge_x.shape
        knowledge_x = knowledge_x[:, 0].reshape(batch_size, self.num_filter_banks, self.num_channels, num_timesteps)
        features = self.knowledge_layer(knowledge_x)
        return self.backbone(features)


class DualInputHybridBackbone(nn.Module):
    def __init__(self, knowledge_layer, backbone, num_filter_banks, num_channels):
        super().__init__()
        self.knowledge_layer = knowledge_layer
        self.backbone = backbone
        self.num_filter_banks = num_filter_banks
        self.num_channels = num_channels

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            knowledge_x, raw_x = x
        else:
            knowledge_x = x
            raw_x = x

        if knowledge_x.ndim != 4 or knowledge_x.shape[1] != 1:
            raise ValueError(f'Expected knowledge_x to have shape (B, 1, C, T), got {tuple(knowledge_x.shape)}')
        if raw_x.ndim != 4 or raw_x.shape[1] != 1:
            raise ValueError(f'Expected raw_x to have shape (B, 1, C, T), got {tuple(raw_x.shape)}')

        knowledge_batch_size, _, _, knowledge_timesteps = knowledge_x.shape
        raw_batch_size, _, _, raw_timesteps = raw_x.shape
        if knowledge_batch_size != raw_batch_size:
            raise ValueError(
                f'Expected knowledge_x and raw_x to share the same batch size, got {knowledge_batch_size} and {raw_batch_size}'
            )

        knowledge_eeg = knowledge_x[:, 0].reshape(knowledge_batch_size, self.num_filter_banks, self.num_channels, knowledge_timesteps)
        raw_eeg = raw_x[:, 0].reshape(raw_batch_size, self.num_filter_banks, self.num_channels, raw_timesteps)
        raw_flat = raw_eeg.reshape(raw_batch_size, -1, raw_timesteps)

        knowledge_features = self.knowledge_layer(knowledge_eeg)
        if knowledge_features.ndim == 4:
            if knowledge_features.shape[1] != 1:
                raise ValueError(
                    f'Expected knowledge features to have a singleton channel dimension, got {tuple(knowledge_features.shape)}'
                )
            knowledge_features = knowledge_features[:, 0]
        elif knowledge_features.ndim != 3:
            raise ValueError(
                f'Expected knowledge features to have shape (B, C, T) or (B, 1, C, T), got {tuple(knowledge_features.shape)}'
            )
        return self.backbone(knowledge_features, raw_flat)


def _reshape_flat_to_filter_bank(data, num_filter_banks, num_channels):
    if data.ndim != 3:
        raise ValueError(f'Expected data to have shape (B, C, T), got {tuple(data.shape)}')
    expected_channels = num_filter_banks * num_channels
    if data.shape[1] != expected_channels:
        raise ValueError(
            f'Expected {expected_channels} flattened channels for {num_filter_banks} filter banks and '
            f'{num_channels} channels, got {data.shape[1]}'
        )
    return data.reshape(data.shape[0], num_filter_banks, num_channels, data.shape[-1])


def _build_backbone(config, deep_method, device, dl_input, num_timesteps):
    _, dataset_spec = get_active_dataset_config(config)
    num_classes = int(dataset_spec['num_classes'])
    sample_rate = int(dataset_spec['sample_rate'])
    window_sec = num_timesteps / sample_rate
    start_frequency = float(dataset_spec['start_frequency'])
    frequency_step = float(dataset_spec['frequency_step'])
    model_cfg = get_classifier_model_config(config, deep_method)

    if deep_method == 'EEGNet':
        return EEGNet.EEGNet(dl_input, num_timesteps, num_classes)
    if deep_method == 'CCNN':
        return CCNN.CNN(dl_input, 220, num_classes)
    if deep_method == 'FBtCNN':
        return FBtCNN.tCNN(dl_input, num_timesteps, num_classes, sample_rate)
    if deep_method == 'SSVEPformer':
        model = SSVEPformer.SSVEPformer(
            depth=2,
            attention_kernal_length=31,
            FFT_PARAMS=[sample_rate, window_sec],
            chs_num=dl_input,
            class_num=num_classes,
            dropout=0.5,
            resolution=frequency_step,
            start_freq=start_frequency,
            end_freq=64,
        )
        model.apply(initialize_weights)
        return model
    if deep_method == 'SSVEPNet':
        return Spectral_Normalization(SSVEPNet.ESNet(dl_input, num_timesteps, num_classes))
    if deep_method == 'DDGCNN':
        return DDGCNN.DenseDDGCNN(
            [int(model_cfg.get('batch_size', 30)), num_timesteps, dl_input],
            k_adj=3,
            num_out=int(model_cfg['num_filters']),
            dropout=0.5,
            n_blocks=3,
            nclass=num_classes,
            bias=False,
            norm=model_cfg['normalization'],
            act=model_cfg['activation'],
            trans_class=model_cfg['transform_class'],
            device=device,
        )
    raise ValueError(f'Unsupported hybrid deep model: {deep_method}')


def _build_enh_ssvepformer(config, device, dl_input, num_timesteps, knowledge_layer, num_filter_banks, num_channels):
    _, dataset_spec = get_active_dataset_config(config)
    num_classes = int(dataset_spec['num_classes'])
    sample_rate = int(dataset_spec['sample_rate'])
    start_frequency = float(dataset_spec['start_frequency'])
    frequency_step = float(dataset_spec['frequency_step'])
    model_cfg = get_classifier_model_config(config, 'EnhSSVEPformer')

    ref_sources = model_cfg.get('ref_sources', ['sincos'])
    if isinstance(ref_sources, str):
        ref_sources = [ref_sources]
    num_harmonics = int(model_cfg.get('num_harmonics', 5))
    raw_channels = num_filter_banks * num_channels

    raw_template_dim = 0
    trca_template_dim = 0
    tdca_template_dim = 0

    if 'raw_template' in ref_sources:
        raw_template_dim = num_filter_banks * num_channels * num_timesteps
    if 'trca' in ref_sources:
        if hasattr(knowledge_layer, 'know2layer'):
            n_comp = knowledge_layer.know2layer.n_components
        elif hasattr(knowledge_layer, 'knowlayer') and hasattr(knowledge_layer.knowlayer, 'n_components'):
            n_comp = knowledge_layer.knowlayer.n_components
        else:
            n_comp = 1
        trca_template_dim = num_filter_banks * n_comp * num_timesteps
    if 'tdca' in ref_sources:
        if hasattr(knowledge_layer, 'know1layer'):
            tdca_n_comp = knowledge_layer.know1layer.n_components
        elif hasattr(knowledge_layer, 'knowlayer') and hasattr(knowledge_layer.knowlayer, 'n_components'):
            tdca_n_comp = knowledge_layer.knowlayer.n_components
        else:
            tdca_n_comp = 1
        tdca_template_dim = num_filter_banks * tdca_n_comp * num_timesteps

    model = EnhSSVEPformerModule.EnhancedSSVEPformer(
        knowledge_channels=dl_input,
        num_timesteps=num_timesteps,
        num_classes=num_classes,
        sample_rate=sample_rate,
        start_frequency=start_frequency,
        frequency_step=frequency_step,
        raw_channels=raw_channels,
        d_model=int(model_cfg.get('d_model', 128)),
        depth=int(model_cfg.get('depth', 2)),
        num_heads=int(model_cfg.get('num_heads', 4)),
        dropout=float(model_cfg.get('dropout', 0.5)),
        num_harmonics=num_harmonics,
        resolution=frequency_step,
        end_freq=64,
        ref_sources=ref_sources,
        raw_template_dim=raw_template_dim,
        trca_template_dim=trca_template_dim,
        tdca_template_dim=tdca_template_dim,
    )

    _set_reference_templates(model, knowledge_layer, ref_sources, num_filter_banks, num_timesteps)
    return model


def _set_reference_templates(model, knowledge_layer, ref_sources, num_filter_banks, num_timesteps):
    Nf = model.num_classes

    if 'raw_template' in ref_sources and 'raw_template' in model.ref_cross_attns:
        if hasattr(knowledge_layer, 'eeg_data_train'):
            raw_trains = knowledge_layer.eeg_data_train
            templates = np.mean(raw_trains, axis=-1)
            templates_flat = templates.reshape(Nf, -1)
            model.set_reference_templates('raw_template', templates_flat)

    if 'trca' in ref_sources and 'trca' in model.ref_cross_attns:
        layer = knowledge_layer.know2layer if hasattr(knowledge_layer, 'know2layer') else getattr(knowledge_layer, 'knowlayer', None)
        if layer is not None and hasattr(layer, 'trains') and hasattr(layer, 'W'):
            trains = layer.trains.detach().cpu().numpy() if isinstance(layer.trains, torch.Tensor) else layer.trains
            W = layer.W.detach().cpu().numpy()
            n_comp = layer.n_components
            Nm = W.shape[0]
            templates = np.zeros((Nf, Nm * n_comp * num_timesteps), dtype=np.float32)
            for fi in range(Nf):
                parts = []
                for fb_i in range(Nm):
                    train_data = trains[fi, fb_i]
                    w = W[fb_i, fi]
                    projected = w @ train_data
                    parts.append(projected.flatten())
                templates[fi] = np.concatenate(parts)
            model.set_reference_templates('trca', templates)

    if 'tdca' in ref_sources and 'tdca' in model.ref_cross_attns:
        layer = knowledge_layer.know1layer if hasattr(knowledge_layer, 'know1layer') else getattr(knowledge_layer, 'knowlayer', None)
        if layer is not None and hasattr(layer, 'trains') and hasattr(layer, 'W') and hasattr(layer, 'M'):
            trains = layer.trains.detach().cpu().numpy() if isinstance(layer.trains, torch.Tensor) else layer.trains
            W = layer.W.detach().cpu().numpy()
            M = layer.M
            n_comp = layer.n_components
            Nm = W.shape[0]
            templates = np.zeros((Nf, Nm * n_comp * num_timesteps), dtype=np.float32)
            for fi in range(Nf):
                parts = []
                for fb_i in range(Nm):
                    train_data = trains[fi, fb_i] - M[fb_i]
                    w = W[fb_i]
                    projected = w @ train_data
                    parts.append(projected.flatten())
                templates[fi] = np.concatenate(parts)
            model.set_reference_templates('tdca', templates)


def _build_hybrid_models(
    config,
    device,
    knowledge_method,
    deep_method,
    train_data,
    train_labels,
    test_data,
    test_labels,
    knowledge_train_data=None,
    knowledge_train_labels=None,
    knowledge_test_data=None,
    knowledge_test_labels=None,
):
    signal_cfg = get_signal_config(config)
    _, dataset_spec = get_active_dataset_config(config)

    num_filter_banks = max(1, int(signal_cfg.get('num_filter_banks', 1)))
    num_channels = int(dataset_spec['num_channels'])
    num_classes = int(dataset_spec['num_classes'])
    ensemble_size = int(signal_cfg.get('ensemble_size', 1))
    num_timesteps = train_data.shape[-1]

    knowledge_train_data = train_data if knowledge_train_data is None else knowledge_train_data
    knowledge_test_data = test_data if knowledge_test_data is None else knowledge_test_data
    knowledge_train_labels = train_labels if knowledge_train_labels is None else knowledge_train_labels
    knowledge_test_labels = test_labels if knowledge_test_labels is None else knowledge_test_labels

    train_structured = _reshape_flat_to_filter_bank(knowledge_train_data, num_filter_banks, num_channels)
    test_structured = _reshape_flat_to_filter_bank(knowledge_test_data, num_filter_banks, num_channels)

    if knowledge_method == 'TDCA':
        feature_channels = int(get_knowledge_method_config(config, knowledge_method).get('num_components', 1))
    elif knowledge_method == 'TD-RCA':
        knowledge_cfg = get_knowledge_method_config(config, knowledge_method)
        trca_mode = str(knowledge_cfg.get('trca_init_mode', ''))
        tdca_mode = str(knowledge_cfg.get('tdca_init_mode', ''))
        if trca_mode == 'disabled' and tdca_mode == 'disabled':
            feature_channels = num_channels
        elif trca_mode == 'disabled':
            feature_channels = int(knowledge_cfg.get('tdca_n_components', 3))
        else:
            feature_channels = num_classes
    else:
        feature_channels = num_classes
    dl_input = feature_channels * num_filter_banks

    output_shape = (num_classes, num_classes, max(1, knowledge_test_labels.shape[0] // num_classes))
    model_cfg = get_classifier_model_config(config, deep_method)
    models = []

    for _ in range(ensemble_size):
        knowledge_layer = KLGlayer(
            eeg_data_train=train_structured,
            label_data_train=knowledge_train_labels,
            eeg_data_test=test_structured,
            label_data_test=knowledge_test_labels,
            config=config,
            knowledge_method=knowledge_method,
            output=np.zeros(output_shape, dtype=np.float64),
        )

        if deep_method == 'EnhSSVEPformer':
            backbone = _build_enh_ssvepformer(config, device, dl_input, num_timesteps, knowledge_layer, num_filter_banks, num_channels)
            model = DualInputHybridBackbone(knowledge_layer, backbone, num_filter_banks, num_channels)
        else:
            backbone = _build_backbone(config, deep_method, device, dl_input, num_timesteps)
            model = HybridKnowledgeBackbone(knowledge_layer, backbone, num_filter_banks, num_channels)

        if use_marginal_smoothing(config) and deep_method == 'SSVEPNet':
            stimulus_type = 40 if num_classes == 40 else 12
            criterion = CELoss_Marginal_Smooth(num_classes, stimulus_type=stimulus_type)
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')

        lr = float(model_cfg['lr'])
        weight_decay = float(model_cfg['weight_decay'])
        params = filter(lambda p: p.requires_grad, model.parameters())
        if deep_method in ('SSVEPformer', 'EnhSSVEPformer'):
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

        models.append({'model': model.to(device), 'criterion': criterion, 'optimizer': optimizer, 'acc_1': 0})
    return models


def _align_hybrid_branch_inputs(branch_data, branch_labels, target_labels):
    if branch_labels.shape[0] == target_labels.shape[0]:
        return branch_data
    if target_labels.shape[0] % branch_labels.shape[0] != 0:
        raise ValueError(
            f'Cannot align knowledge branch with {branch_labels.shape[0]} samples to downstream branch with {target_labels.shape[0]} samples.'
        )

    repeats = target_labels.shape[0] // branch_labels.shape[0]
    expanded_data = np.concatenate([branch_data] * repeats, axis=0)
    expanded_labels = np.concatenate([branch_labels] * repeats, axis=0)
    if not np.array_equal(expanded_labels, target_labels):
        raise ValueError('Knowledge-branch labels do not align with downstream labels after expansion.')
    return expanded_data


def _train_hybrid_models(
    config,
    models,
    device,
    deep_method,
    train_data,
    train_labels,
    test_data,
    test_labels,
    knowledge_train_data=None,
    knowledge_train_labels=None,
    knowledge_test_data=None,
    knowledge_test_labels=None,
):
    model_cfg = get_classifier_model_config(config, deep_method)
    batch_size = int(model_cfg.get('batch_size', 30))
    num_epochs = int(model_cfg['epochs'])
    ensemble_size = int(get_signal_config(config).get('ensemble_size', 1))
    use_scheduler = bool(model_cfg.get('use_scheduler', False))

    knowledge_train_data = train_data if knowledge_train_data is None else knowledge_train_data
    knowledge_test_data = test_data if knowledge_test_data is None else knowledge_test_data
    knowledge_train_labels = train_labels if knowledge_train_labels is None else knowledge_train_labels
    knowledge_test_labels = test_labels if knowledge_test_labels is None else knowledge_test_labels

    paired_knowledge_train_data = _align_hybrid_branch_inputs(knowledge_train_data, knowledge_train_labels, train_labels)
    paired_knowledge_test_data = _align_hybrid_branch_inputs(knowledge_test_data, knowledge_test_labels, test_labels)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    knowledge_train_tensor = torch.tensor(paired_knowledge_train_data, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_labels)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    knowledge_test_tensor = torch.tensor(paired_knowledge_test_data, dtype=torch.float32)
    test_label_tensor = torch.tensor(test_labels)

    train_set = torch.utils.data.TensorDataset(knowledge_train_tensor.unsqueeze(1), train_tensor.unsqueeze(1), train_label_tensor.unsqueeze(1))
    test_set = torch.utils.data.TensorDataset(knowledge_test_tensor.unsqueeze(1), test_tensor.unsqueeze(1), test_label_tensor.unsqueeze(1))
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    schedulers = []
    for model_pack in models:
        optimizer = model_pack['optimizer']
        if deep_method == 'DDGCNN':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=float(model_cfg['lr_decay_factor']),
                patience=int(model_cfg['scheduler_patience']),
                verbose=True,
                eps=1e-8,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs * len(train_iter)),
                eta_min=5e-6,
            )
        schedulers.append(scheduler)

    final_valid_acc = torch.tensor(0.0)
    for epoch_index in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for ensemble_index in range(ensemble_size):
            net = models[ensemble_index]['model']
            optimizer = models[ensemble_index]['optimizer']
            criterion = models[ensemble_index]['criterion']
            scheduler = schedulers[ensemble_index]
            net.train()
            sum_loss = 0.0
            sum_acc = 0.0
            for knowledge_data, data, labels in train_iter:
                knowledge_data = knowledge_data.float().to(device)
                data = data.float().to(device)
                labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)
                logits = net((knowledge_data, data))
                loss = criterion(logits, labels).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if use_scheduler and deep_method != 'DDGCNN':
                    scheduler.step()
                sum_loss += loss.item() / labels.shape[0]
                sum_acc += (labels == logits.argmax(dim=-1)).float().mean()

            train_loss = sum_loss / len(train_iter)
            train_acc = sum_acc / len(train_iter)
            if use_scheduler and deep_method == 'DDGCNN':
                scheduler.step(train_loss)

        if (epoch_index + 1) % 10 == 0 or epoch_index + 1 == num_epochs:
            sum_acc = 0.0
            total_batches = 0
            for knowledge_data, data, labels in test_iter:
                total_batches += 1
                knowledge_data = knowledge_data.float().to(device)
                data = data.float().to(device)
                labels = torch.as_tensor(labels.reshape(labels.shape[0]), dtype=torch.int64).to(device)

                total_logits = 0
                for ensemble_index in range(ensemble_size):
                    net = models[ensemble_index]['model']
                    net.eval()
                    with torch.no_grad():
                        total_logits += net((knowledge_data, data))

                avg_logits = total_logits / ensemble_size
                predictions = avg_logits.argmax(dim=-1)
                sum_acc += (labels == predictions).float().mean()

            final_valid_acc = sum_acc / total_batches
            print_progress_line(
                f'hybrid/{deep_method}',
                epoch_index + 1,
                num_epochs,
                loss=train_loss,
                train=train_acc,
                valid=final_valid_acc,
            )

    finish_progress_line(
        f"[hybrid/{deep_method}] final_valid={float(final_valid_acc):.4f} "
        f"time={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    torch.cuda.empty_cache()
    return models[-1]['acc_1'], final_valid_acc.cpu().data


def run_single_hybrid_method(
    config,
    device,
    knowledge_method,
    deep_method,
    train_data,
    train_labels,
    test_data,
    test_labels,
    knowledge_train_data=None,
    knowledge_train_labels=None,
    knowledge_test_data=None,
    knowledge_test_labels=None,
):
    models = _build_hybrid_models(
        config,
        device,
        knowledge_method,
        deep_method,
        train_data,
        train_labels,
        test_data,
        test_labels,
        knowledge_train_data=knowledge_train_data,
        knowledge_train_labels=knowledge_train_labels,
        knowledge_test_data=knowledge_test_data,
        knowledge_test_labels=knowledge_test_labels,
    )
    _, acc = _train_hybrid_models(
        config,
        models,
        device,
        deep_method,
        train_data,
        train_labels,
        test_data,
        test_labels,
        knowledge_train_data=knowledge_train_data,
        knowledge_train_labels=knowledge_train_labels,
        knowledge_test_data=knowledge_test_data,
        knowledge_test_labels=knowledge_test_labels,
    )
    del models
    return acc


def evaluate_hybrid_method(config, device, method_name, train_variants, train_labels, test_variants, test_labels):
    knowledge_method, deep_method = parse_hybrid_method(method_name)

    acc_clean = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['clean'], train_labels, test_variants['clean'], test_labels)
    acc_noisy = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['noisy'], train_labels, test_variants['noisy'], test_labels)
    acc_generator = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['generator'], train_labels, test_variants['generator'], test_labels)
    acc_ng = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['ng'], train_labels, test_variants['ng'], test_labels)
    acc_cn = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['cn'], train_labels, test_variants['cn'], test_labels)
    acc_cg = run_single_hybrid_method(config, device, knowledge_method, deep_method, train_variants['cg'], train_labels, test_variants['cg'], test_labels)
    return (
        float(acc_clean),
        float(acc_noisy),
        float(acc_generator),
        float(acc_ng),
        float(acc_cn),
        float(acc_cg),
    )


is_hybrid_testmethod = is_hybrid_method
parse_hybrid_testmethod = parse_hybrid_method
_run_single_hybrid = run_single_hybrid_method


