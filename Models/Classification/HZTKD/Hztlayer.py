from dataclasses import dataclass
from pathlib import Path
from typing import List

import math

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.signal import firwin

from Support.config import (
    get_active_dataset_config,
    get_dataset_root,
    get_knowledge_method_config,
    get_signal_config,
    resolve_dataset_name,
)


TARGET_OCCIPITAL_CHANNELS = {'PZ', 'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'OZ', 'O1', 'O2'}

VALID_INIT_MODES = {'analytical_frozen', 'analytical_trainable', 'random_trainable', 'disabled'}


@dataclass
class HybridLayerSettings:
    dataset_name: str
    dataset_root: str
    sample_rate: int
    num_classes: int
    num_channels: int
    num_filter_banks: int
    num_timesteps: int
    start_frequency: float
    frequency_step: float
    use_bandpass: bool
    freeze_bandpass: bool
    use_all_channels: bool
    num_components: int
    freeze_knowledge_layer: bool
    use_mix_layer: bool
    use_deep_layout: bool
    use_filter_bank_output: bool
    freeze_filter_bank: bool
    mix_layer_fixed: bool
    use_temporal_filter_bank: bool = False
    tdca_n_components: int = 3
    trca_n_components: int = 1
    tdca_freeze: bool = True
    trca_freeze: bool = True
    tdca_init_mode: str = 'analytical_frozen'
    trca_init_mode: str = 'analytical_frozen'
    fb_init_mode: str = 'analytical_frozen'


def isPD(matrix):
    try:
        _ = np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def nearestPD(matrix):
    sym = (matrix + matrix.T) / 2
    _, singular_values, vh = np.linalg.svd(sym)
    h_mat = np.dot(vh.T, np.dot(np.diag(singular_values), vh))
    candidate = (sym + h_mat) / 2
    candidate = (candidate + candidate.T) / 2

    if isPD(candidate):
        return candidate

    spacing = np.spacing(np.linalg.norm(matrix))
    eye = np.eye(matrix.shape[0])
    k = 1
    while not isPD(candidate):
        min_eig = np.min(np.real(np.linalg.eigvals(candidate)))
        candidate += eye * (-min_eig * k ** 2 + spacing)
        k += 1
    return candidate


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper()


def _load_beta_channel_indices(dataset_root: str) -> List[int]:
    first_subject = Path(dataset_root) / 'S1.mat'
    if not first_subject.exists():
        return []

    data = loadmat(first_subject)
    raw_channels = data['data'][0, 0][1][0, 0][3][:, -1]
    channel_names = [_normalize_channel_name(channel[0]) for channel in raw_channels]
    return [
        index
        for index, channel_name in enumerate(channel_names)
        if channel_name in TARGET_OCCIPITAL_CHANNELS
    ]


def _resolve_channel_indices(config, dataset_name: str, dataset_root: str, num_channels: int, use_all_channels: bool) -> List[int]:
    if use_all_channels:
        return list(range(num_channels))

    _, dataset_spec = get_active_dataset_config(config)
    selected_channels = dataset_spec.get('selected_channels')
    if isinstance(selected_channels, list) and selected_channels:
        if len(selected_channels) == num_channels:
            return list(range(num_channels))
        if max(selected_channels) < num_channels:
            return [int(index) for index in selected_channels]

    if dataset_name == 'beta':
        beta_indices = _load_beta_channel_indices(dataset_root)
        if beta_indices:
            if len(beta_indices) == num_channels:
                return list(range(num_channels))
            if max(beta_indices) < num_channels:
                return beta_indices

    return list(range(num_channels))


def _resolve_init_mode(cfg: dict, mode_key: str, freeze_key: str, default_freeze: bool = True) -> str:
    explicit = str(cfg.get(mode_key, ''))
    if explicit in VALID_INIT_MODES:
        return explicit
    frozen = bool(cfg.get(freeze_key, default_freeze))
    return 'analytical_frozen' if frozen else 'analytical_trainable'


def _build_hybrid_layer_settings(config, knowledge_method: str, num_timesteps: int) -> HybridLayerSettings:
    dataset_name = resolve_dataset_name(config)
    dataset_root = get_dataset_root(config, dataset_name)
    _, dataset_spec = get_active_dataset_config(config)
    signal_cfg = get_signal_config(config)
    knowledge_cfg = get_knowledge_method_config(config, knowledge_method)
    pipeline_cfg = signal_cfg.get('knowledge_pipeline', {}) if isinstance(signal_cfg.get('knowledge_pipeline'), dict) else {}

    return HybridLayerSettings(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        sample_rate=int(dataset_spec['sample_rate']),
        num_classes=int(dataset_spec['num_classes']),
        num_channels=int(dataset_spec['num_channels']),
        num_filter_banks=max(1, int(signal_cfg.get('num_filter_banks', 1))),
        num_timesteps=int(num_timesteps),
        start_frequency=float(dataset_spec['start_frequency']),
        frequency_step=float(dataset_spec['frequency_step']),
        use_bandpass=bool(pipeline_cfg.get('use_bandpass', False)),
        freeze_bandpass=bool(knowledge_cfg.get('freeze_bandpass', False)),
        use_all_channels=bool(knowledge_cfg.get('use_all_channels', True)),
        num_components=int(knowledge_cfg.get('num_components', 1)),
        freeze_knowledge_layer=bool(knowledge_cfg.get('freeze_knowledge_layer', False)),
        use_mix_layer=bool(pipeline_cfg.get('use_mix_layer', False)),
        use_deep_layout=True,
        use_filter_bank_output=bool(pipeline_cfg.get('use_filter_bank_output', True)),
        freeze_filter_bank=bool(knowledge_cfg.get('freeze_filter_bank', False)),
        mix_layer_fixed=bool(pipeline_cfg.get('mix_layer_fixed', False)),
        use_temporal_filter_bank=bool(knowledge_cfg.get('use_temporal_filter_bank', False)),
        tdca_n_components=int(knowledge_cfg.get('tdca_n_components', knowledge_cfg.get('num_components', 3))),
        trca_n_components=int(knowledge_cfg.get('trca_n_components', 1)),
        tdca_freeze=bool(knowledge_cfg.get('tdca_freeze', knowledge_cfg.get('freeze_knowledge_layer', True))),
        trca_freeze=bool(knowledge_cfg.get('trca_freeze', knowledge_cfg.get('freeze_knowledge_layer', True))),
        tdca_init_mode=_resolve_init_mode(knowledge_cfg, 'tdca_init_mode', 'tdca_freeze',
                                          default_freeze=bool(knowledge_cfg.get('freeze_knowledge_layer', True))),
        trca_init_mode=_resolve_init_mode(knowledge_cfg, 'trca_init_mode', 'trca_freeze',
                                          default_freeze=bool(knowledge_cfg.get('freeze_knowledge_layer', True))),
        fb_init_mode=_resolve_init_mode(knowledge_cfg, 'fb_init_mode', 'freeze_filter_bank',
                                        default_freeze=False),
    )


class KLGlayer(nn.Module):
    def __init__(
        self,
        kernel_size=33,
        eeg_data_train=None,
        label_data_train=None,
        eeg_data_test=None,
        label_data_test=None,
        config=None,
        knowledge_method=None,
        output=0,
    ):
        super().__init__()
        if config is None:
            raise ValueError('config is required for KLGlayer.')
        if knowledge_method is None:
            raise ValueError('knowledge_method is required for KLGlayer.')
        if eeg_data_train is None or eeg_data_test is None:
            raise ValueError('eeg_data_train and eeg_data_test are required for KLGlayer.')

        self.kernel_size = kernel_size
        self.output = output
        self.KLGalgorithm = knowledge_method
        self.label_data_train = np.asarray(label_data_train).reshape(-1)
        self.Nt = eeg_data_train.shape[-1]
        self.settings = _build_hybrid_layer_settings(config, knowledge_method, self.Nt)

        self.deltafreq = self.settings.frequency_step
        self.initfreq = self.settings.start_frequency
        self.Fs = self.settings.sample_rate
        self.Nf = self.settings.num_classes
        self.Nm = self.settings.num_filter_banks
        self.Nc = self.settings.num_channels
        self.Bandpass = self.settings.use_bandpass
        self.Bandpass_FIX = self.settings.freeze_bandpass
        self.slc_all = self.settings.use_all_channels
        self.n_components = self.settings.num_components
        self.KLG_FIX = self.settings.freeze_knowledge_layer
        self.MIX = self.settings.use_mix_layer
        self.DL = self.settings.use_deep_layout
        self.FB = self.settings.use_filter_bank_output
        self.mix_FIX = self.settings.mix_layer_fixed
        self.FB_FIX = self.settings.freeze_filter_bank

        self.channels = _resolve_channel_indices(
            config,
            self.settings.dataset_name,
            self.settings.dataset_root,
            self.Nc,
            self.slc_all,
        )

        self.eeg_data_train_raw = np.asarray(eeg_data_train, dtype=np.float32)
        self.eeg_data_test_raw = np.asarray(eeg_data_test, dtype=np.float32)
        self.eeg_data_train = self._reshape_trials_to_class_first(self.eeg_data_train_raw, self.Nc)
        self.eeg_data_test = self._reshape_trials_to_class_first(self.eeg_data_test_raw, self.Nc)
        self.label_data_test = np.asarray(label_data_test).reshape(self.Nf, -1)

        self._build_pipeline()

    def _reshape_trials_to_class_first(self, eeg_data: np.ndarray, num_channels: int) -> np.ndarray:
        reshaped = eeg_data.reshape(self.Nf, -1, self.Nm, num_channels, self.Nt)
        return np.transpose(reshaped, (0, 2, 3, 4, 1))

    def _build_pipeline(self):
        if self.Bandpass:
            if self.settings.use_temporal_filter_bank:
                self.fb_layer = TemporalFilterBank(
                    Fs=self.Fs,
                    Nc=self.Nc,
                    Nm=self.Nm,
                    kernel_size=129,
                    dataset_name=self.settings.dataset_name,
                    FIX=self.Bandpass_FIX,
                )
                train_raw_single = torch.tensor(self.eeg_data_train_raw[:, 0], dtype=torch.float32)
                test_raw_single = torch.tensor(self.eeg_data_test_raw[:, 0], dtype=torch.float32)
                train_tensor = self.fb_layer(train_raw_single)
                test_tensor = self.fb_layer(test_raw_single)
            else:
                self.Bandpasslayer = BandpassConvLayer(
                    Fs=self.Fs,
                    initfreq=self.initfreq,
                    deltafreq=self.deltafreq,
                    Nf=self.Nf,
                    kernel_size=self.kernel_size,
                    Nm=self.Nm,
                    Bandpass_FIX=self.Bandpass_FIX,
                )
                train_tensor = self.Bandpasslayer(torch.tensor(self.eeg_data_train_raw, dtype=torch.float32))
                test_tensor = self.Bandpasslayer(torch.tensor(self.eeg_data_test_raw, dtype=torch.float32))
            train_structured = self._reshape_trials_to_class_first(train_tensor.detach().cpu().numpy(), self.Nc)
            test_structured = self._reshape_trials_to_class_first(test_tensor.detach().cpu().numpy(), self.Nc)
            self._init_knowledge_layers(train_tensor, test_tensor, train_structured, test_structured, bandpassed=True)
            return

        self._init_knowledge_layers(
            self.eeg_data_train_raw,
            self.eeg_data_test_raw,
            self.eeg_data_train,
            self.eeg_data_test,
            bandpassed=False,
        )

    def _init_knowledge_layers(self, train_trials, test_trials, train_structured, test_structured, bandpassed: bool):
        if self.KLGalgorithm == 'TRCA':
            self._init_trca_layer(train_trials, test_structured, train_structured)
            feature_count = self.Nf
        elif self.KLGalgorithm == 'TDCA':
            self._init_tdca_layer(train_trials, test_structured, train_structured)
            feature_count = self.n_components
        elif self.KLGalgorithm == 'TD-RCA':
            feature_count = self._init_td_rca_layer(train_trials, test_trials, test_structured, train_structured, bandpassed)
        else:
            raise ValueError(f'Unsupported knowledge method: {self.KLGalgorithm}')

        self.mixlayer = MIXlayer(mix_FIX=self.mix_FIX, Nf=feature_count, Nm=self.Nm, DL=self.DL)

        fb_mode = self.settings.fb_init_mode
        if fb_mode == 'disabled':
            self.FB = False
            self.fblayer = None
        else:
            fb_frozen = (fb_mode == 'analytical_frozen')
            self.fblayer = FBlayer(FB_FIX=fb_frozen, Nf=feature_count, Nm=self.Nm, DL=self.DL)
            if fb_mode == 'random_trainable':
                with torch.no_grad():
                    self.fblayer.weights.data.uniform_(0.1, 1.0)
                self.fblayer.weights.requires_grad = True

    def _init_trca_layer(self, train_trials, test_structured, train_structured):
        self.knowlayer = TRCAlayer(self.KLG_FIX, train_structured, self.channels, self.DL)
        self.knowlayer.trca(self._as_numpy(train_trials), self.label_data_train)
        self.init1_acc = self.knowlayer.test_trca(test_structured, self.label_data_test)
        self.output = self.knowlayer.output

    def _init_tdca_layer(self, train_trials, test_structured, train_structured):
        self.knowlayer = TDCAlayer(
            self.KLG_FIX,
            train_structured,
            self.channels,
            self.DL,
            n_components=self.n_components,
        )
        self.knowlayer.tdca(self._as_numpy(train_trials), self.label_data_train)
        self.init1_acc = self.knowlayer.test_tdca(test_structured, self.label_data_test)
        self.output = self.knowlayer.output

    def _init_td_rca_layer(self, train_trials, test_trials, test_structured, train_structured, bandpassed: bool):
        tdca_mode = self.settings.tdca_init_mode
        trca_mode = self.settings.trca_init_mode
        tdca_n_comp = self.settings.tdca_n_components
        trca_n_comp = self.settings.trca_n_components

        # --- Stage 1: TDCA ---
        if tdca_mode != 'disabled':
            tdca_frozen = (tdca_mode == 'analytical_frozen')
            first_stage_kwargs = {}
            if not bandpassed:
                first_stage_kwargs['output'] = self.output

            self.know1layer = TDCAlayer(
                tdca_frozen,
                train_structured,
                self.channels,
                self.DL,
                n_components=tdca_n_comp,
                **first_stage_kwargs,
            )
            train_trials_np = self._as_numpy(train_trials)
            self.know1layer.tdca(train_trials_np, self.label_data_train)

            if tdca_mode == 'random_trainable':
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.know1layer.W)
                self.know1layer.W.requires_grad = True

            self.init1_acc = self.know1layer.test_tdca(test_structured, self.label_data_test)
            self.output = self.know1layer.output

            tdca_train = self.know1layer.forward2(self._as_tensor(train_trials))
            tdca_train_np = tdca_train.detach().cpu().numpy()
            tdca_train_structured = self._reshape_trials_to_class_first(tdca_train_np, tdca_n_comp)
            trca_input_channels = list(range(tdca_n_comp))

            tdca_test = self.know1layer.forward2(self._as_tensor(test_trials))
            tdca_test_structured = self._reshape_trials_to_class_first(tdca_test.detach().cpu().numpy(), tdca_n_comp)
            trca_train_data = tdca_train_np
            trca_train_structured = tdca_train_structured
            trca_test_structured_local = tdca_test_structured
        else:
            self.know1layer = None
            self.init1_acc = 0.0
            trca_input_channels = self.channels
            trca_train_data = self._as_numpy(train_trials)
            trca_train_structured = train_structured
            trca_test_structured_local = test_structured

        # --- Stage 2: TRCA ---
        if trca_mode != 'disabled':
            trca_frozen = (trca_mode == 'analytical_frozen')
            self.know2layer = TRCAlayer(
                trca_frozen, trca_train_structured, trca_input_channels, self.DL, n_components=trca_n_comp,
            )
            self.know2layer.trca(trca_train_data, self.label_data_train)

            if trca_mode == 'random_trainable':
                with torch.no_grad():
                    nn.init.xavier_uniform_(self.know2layer.W)
                self.know2layer.W.requires_grad = True

            self.init2_acc = self.know2layer.test_trca(trca_test_structured_local, self.label_data_test)
            feature_count = self.Nf
        else:
            self.know2layer = None
            self.init2_acc = 0.0
            feature_count = tdca_n_comp if tdca_mode != 'disabled' else len(self.channels)

        # Build composite knowledge layer
        if self.know1layer is not None and self.know2layer is not None:
            self.knowlayer = nn.Sequential(self.know1layer, self.know2layer)
        elif self.know1layer is not None:
            self.knowlayer = self.know1layer
        elif self.know2layer is not None:
            self.knowlayer = self.know2layer
        else:
            self.knowlayer = nn.Identity()

        return feature_count

    @staticmethod
    def _as_numpy(array_like):
        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    @staticmethod
    def _as_tensor(array_like):
        if isinstance(array_like, torch.Tensor):
            return array_like.float()
        return torch.tensor(array_like, dtype=torch.float32)

    def forward(self, x):
        if self.Bandpass:
            if self.settings.use_temporal_filter_bank:
                x = self.fb_layer(x[:, 0] if x.ndim == 4 else x)
            else:
                x = self.Bandpasslayer(x)
        x = self.knowlayer(x)
        if self.MIX:
            x = self.mixlayer(x)
        else:
            x = x.flatten(start_dim=1, end_dim=2).unsqueeze(1)
        if self.FB:
            x = self.fblayer(x)
        else:
            x = x.flatten(start_dim=1, end_dim=2).unsqueeze(1)
        return x


class TemporalFilterBank(nn.Module):
    def __init__(self, Fs, Nc, Nm, kernel_size=None, dataset_name='benchmark12', FIX=True):
        super().__init__()
        self.Fs = Fs
        self.Nc = Nc
        self.Nm = Nm

        if kernel_size is None:
            self.kernel_size = int(0.5 * Fs)
        else:
            self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        if dataset_name in ('benchmark40', 'beta'):
            passband = [8, 16, 24, 32, 40, 48]
            stopband = [6, 12, 18, 26, 34, 42]
        else:
            passband = [9, 18, 27, 36, 45, 54]
            stopband = [7, 15, 19, 28, 37, 46]

        self.freq_bands = [(stopband[i], passband[i]) for i in range(Nm)]

        self.conv = nn.Conv1d(
            in_channels=Nc,
            out_channels=Nc * Nm,
            kernel_size=self.kernel_size,
            groups=Nc,
            padding=self.kernel_size // 2,
            bias=False,
        )
        self._init_fir_weights()

        if FIX:
            for p in self.parameters():
                p.requires_grad = False

    def _init_fir_weights(self):
        with torch.no_grad():
            for m, (low, high) in enumerate(self.freq_bands):
                h = firwin(
                    numtaps=self.kernel_size,
                    cutoff=[low, high],
                    pass_zero=False,
                    fs=self.Fs,
                    window='hann',
                )
                h = h / np.sqrt(np.sum(h ** 2))
                h_tensor = torch.tensor(h, dtype=torch.float32)
                for c in range(self.Nc):
                    self.conv.weight[m * self.Nc + c, 0, :] = h_tensor

    def forward(self, x):
        y_fwd = self.conv(x)
        x_rev = torch.flip(x, dims=[-1])
        y_bwd = torch.flip(self.conv(x_rev), dims=[-1])
        y = 0.5 * (y_fwd + y_bwd)
        B, _, T = y.shape
        return y.view(B, self.Nm, self.Nc, T)


class BandpassConvLayer(nn.Module):
    def __init__(self, Fs=250, initfreq=8, deltafreq=0.2, Nf=40, Nm=1, kernel_size=63, Bandpass_FIX=False):
        super().__init__()
        nyquist = Fs // 2
        self.Nm = Nm
        self.conv = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2) for _ in range(self.Nm)]
        )
        for harmonic_index in range(self.Nm):
            nn.init.zeros_(self.conv[harmonic_index].bias)
            for param in self.conv[harmonic_index].parameters():
                if Bandpass_FIX:
                    param.requires_grad = False
            wp = [
                initfreq * (harmonic_index + 1) / nyquist,
                (initfreq * (harmonic_index + 1) + deltafreq * (Nf - 1)) / nyquist,
            ]
            ws = [
                (initfreq * (harmonic_index + 1) - 2) / nyquist,
                (initfreq * (harmonic_index + 1) + deltafreq * Nf + 2) / nyquist,
            ]
            order, wn = signal.cheb1ord(wp, ws, 1, 40)
            b_coef, a_coef = signal.cheby1(order, 1, wn, btype='bandpass', analog=False)[:]
            impulse_response = signal.lfilter(b_coef, a_coef, np.append(1, np.zeros(kernel_size - 1)))
            with torch.no_grad():
                self.conv[harmonic_index].weight.copy_(torch.tensor(impulse_response).reshape(1, 1, -1))

    def forward(self, x):
        num_channels = x.shape[1]
        if x.ndim != 3:
            raise ValueError(f'X.ndim must be 3 but got {x.shape}')
        return torch.cat(
            [
                torch.cat([self.conv[harmonic_index](x[:, channel_index, :].unsqueeze(1)) for channel_index in range(num_channels)], dim=1)
                .unsqueeze(1)
                for harmonic_index in range(self.Nm)
            ],
            dim=1,
        )


class FBlayer(nn.Module):
    def __init__(self, FB_FIX=True, fb_coefs_a=-1.25, fb_coefs_b=0.25, Nf=40, Nm=1, DL=True):
        super().__init__()
        self.Nm = Nm
        self.Nc = Nf
        self.DL = DL
        weights = torch.tensor(
            [math.pow(i, fb_coefs_a) + fb_coefs_b for i in range(1, self.Nm + 1) for _ in range(self.Nc)]
        ).reshape(-1, 1)
        self.weights = nn.Parameter(weights, requires_grad=not FB_FIX)

    def forward(self, x):
        if self.DL:
            if x.ndim != 4:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            x = x.flatten(start_dim=1, end_dim=2)
            return (x * self.weights).unsqueeze(1)

        if x.ndim != 3:
            raise ValueError(f'X.ndim must be 3 but got {x.shape}')
        output = torch.cat(
            [(x[:, i, :] * self.weights[i * self.Nc:(i + 1) * self.Nc].T).unsqueeze(1) for i in range(self.Nm)],
            dim=1,
        ).sum(dim=1)
        return output


class MIXlayer(nn.Module):
    def __init__(self, mix_FIX=True, fb_coefs_a=-1.25, fb_coefs_b=0.25, Nf=40, Nm=1, DL=True):
        super().__init__()
        self.Nm = Nm
        self.Nc = Nf
        self.DL = DL
        weights = torch.tensor(
            [math.pow(i, fb_coefs_a) + fb_coefs_b for i in range(1, self.Nm + 1) for _ in range(self.Nc)]
        ).reshape(-1, 1)
        self.weights = nn.Parameter(weights, requires_grad=not mix_FIX)

    def forward(self, x):
        if self.DL:
            if x.ndim != 4:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            batch_size, num_filters, num_features, num_timesteps = x.shape
            flat = x.flatten(start_dim=1, end_dim=2)
            output = (flat * self.weights).unsqueeze(1)
            output = output.reshape(batch_size, num_filters, num_features, num_timesteps)
            return output.sum(dim=1, keepdim=True)

        if x.ndim != 3:
            raise ValueError(f'X.ndim must be 3 but got {x.shape}')
        output = torch.cat(
            [(x[:, i, :] * self.weights[i * self.Nc:(i + 1) * self.Nc].T).unsqueeze(1) for i in range(self.Nm)],
            dim=1,
        ).sum(dim=1)
        return output


class Weightlayer(nn.Module):
    def __init__(self, FB_FIX=True, fb_coefs_a=-1.25, fb_coefs_b=0.25, Nf=40, Nm=1, DL=True):
        super().__init__()
        self.Nm = Nm
        self.Nc = Nf
        self.DL = DL

        fb_init = torch.tensor(
            [math.pow(i, fb_coefs_a) + fb_coefs_b for i in range(1, Nm + 1)],
            dtype=torch.float32,
        )
        fb_init = fb_init[:, None].repeat(1, self.Nc)

        self.gate = nn.Conv2d(
            in_channels=Nm * self.Nc,
            out_channels=Nm * self.Nc,
            kernel_size=1,
            groups=Nm * self.Nc,
            bias=False,
        )
        with torch.no_grad():
            self.gate.weight.copy_(fb_init.reshape(Nm * self.Nc, 1, 1, 1))

        if FB_FIX:
            for p in self.gate.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.DL:
            if x.ndim != 4:
                raise ValueError(f'x must be (B, Nm, Nc, T), got {x.shape}')
            B, Nm, Nc, T = x.shape
            x = x.reshape(B, Nm * Nc, 1, T)
            x = self.gate(x)
            return x.view(B, Nm, Nc, T)
        else:
            if x.ndim != 3:
                raise ValueError(f'x must be (B, Nm, Nc), got {x.shape}')
            B, Nm, Nc = x.shape
            x = x.reshape(B, Nm * Nc, 1, 1)
            x = self.gate(x)
            x = x.view(B, Nm, Nc)
            return x.sum(dim=1)


class TRCAlayer(nn.Module):
    def __init__(self, KLG_FIX=True, EEGDATA=None, channels=None, DL=True, n_components=1, dropout_p=0):
        super().__init__()
        if EEGDATA is None:
            raise ValueError('EEGDATA is required for TRCAlayer.')
        self.EEGDATA = EEGDATA
        self.DL = DL
        if EEGDATA.ndim != 5:
            raise ValueError(f'X.ndim must be 5 but got {EEGDATA.shape}')

        self.Nf, self.Nm, self.Nc, self.T, _ = self.EEGDATA.shape
        self.channels = list(range(self.Nc)) if channels is None else channels
        self.n_components = n_components
        self.trains = np.mean(self.EEGDATA, -1)
        self.W = torch.zeros((self.Nm, self.Nf, self.n_components, self.Nc))
        self.is_ensemble = False
        self.output = 0

        if KLG_FIX:
            self.W = nn.Parameter(self.W, requires_grad=False)
            self.trains = nn.Parameter(torch.tensor(self.trains, dtype=torch.float32), requires_grad=False)
        else:
            self.W = nn.Parameter(self.W, requires_grad=True)
            self.trains = nn.Parameter(torch.tensor(self.trains, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        if self.DL:
            if x.ndim != 4:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            outputs = []
            for fb_i in range(self.Nm):
                class_outs = []
                for cls in range(self.Nf):
                    proj = torch.einsum(
                        'kc,bct->bkt',
                        self.W[fb_i, cls],
                        x[:, fb_i],
                    )
                    class_outs.append(proj.unsqueeze(2))
                class_outs = torch.cat(class_outs, dim=1)
                outputs.append(class_outs.unsqueeze(1))
            x = torch.cat(outputs, dim=1)
            x = x.squeeze()
            return x

        if x.ndim != self.trains.ndim:
            raise ValueError('X.ndim != trains.ndim')
        device = x.device if isinstance(x, torch.Tensor) else self.W.device
        r = torch.zeros((len(x), self.Nm, self.Nf), device=device)
        for trial in range(len(x)):
            for class_i in range(self.Nf):
                for fb_i in range(self.Nm):
                    w = self.W[fb_i, class_i]
                    if self.trains.ndim == 5:
                        test = w @ x[trial, fb_i, :, class_i, :]
                        train = w @ self.trains[class_i, fb_i, :, class_i, :]
                    else:
                        test = w @ x[trial, fb_i, :, :]
                        train = w @ self.trains[class_i, fb_i, :, :]
                    r_tmp = torch.corrcoef(torch.stack((test.flatten(), train.flatten())))[0, 1]
                    r[trial, fb_i, class_i] = r_tmp
        return r

    def trca(self, X=None, labels=None):
        for fb_i in range(self.Nm):
            for class_index in range(self.Nf):
                if X.ndim != 4:
                    raise ValueError(f'X.ndim must be 4 but got {X.shape}')
                X_c = X[labels == class_index]
                X_c = X_c[:, fb_i, self.channels, :]
                n_trials, n_channels, _ = X_c.shape
                s_mat = np.zeros((n_channels, n_channels))
                q_mat = np.zeros((n_channels, n_channels))
                for trial_i in range(n_trials):
                    x_i = X_c[trial_i].copy()
                    x_i -= x_i.mean(axis=1, keepdims=True)
                    q_mat += x_i @ x_i.T
                    for trial_j in range(trial_i + 1, n_trials):
                        x_j = X_c[trial_j].copy()
                        x_j -= x_j.mean(axis=1, keepdims=True)
                        s_mat += x_i @ x_j.T + x_j @ x_i.T
                q_mat += 1e-8 * np.eye(n_channels)
                eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(q_mat) @ s_mat)
                sorted_indices = np.argsort(eigvals)[::-1]
                w_mat = eigvecs[:, sorted_indices[:self.n_components]].T
                with torch.no_grad():
                    self.W[fb_i, class_index, :, self.channels] = torch.tensor(w_mat, dtype=torch.float32)
        return self

    def test_trca(self, eeg, test_label):
        if isinstance(eeg, torch.Tensor):
            eeg = eeg.detach().cpu().numpy()
        num_trials = eeg.shape[-1]
        fb_coefs = np.array([math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)])
        results = np.zeros((self.Nf, num_trials))
        for targ_i in range(self.Nf):
            if eeg.ndim != 5:
                raise ValueError(f'X.ndim must be 5 but got {eeg.shape}')
            test_tmp = eeg[targ_i, :, :, :, :]
            r_mat = np.zeros((self.Nm, self.Nf, num_trials))
            for fb_i in range(self.Nm):
                testdata = test_tmp[fb_i, :, :, :]
                for class_i in range(self.Nf):
                    if self.trains.ndim == 4:
                        traindata = self.trains[class_i, fb_i, :, :]
                    else:
                        traindata = self.trains[class_i, fb_i, :, class_i, :]
                    w_mat = self.W[fb_i, class_i].detach().cpu().numpy()
                    train_np = traindata.detach().cpu().numpy() if isinstance(traindata, torch.Tensor) else traindata
                    train_w = w_mat @ train_np
                    for trial_i in range(num_trials):
                        test_w = w_mat @ testdata[:, :, trial_i]
                        r_tmp = np.corrcoef(test_w.flatten(), train_w.flatten())
                        r_mat[fb_i, class_i, trial_i] = r_tmp[0, 1]
            rho = np.einsum('j,jkl->kl', fb_coefs, r_mat)
            results[targ_i, :] = np.argmax(rho, axis=0)
        is_correct = np.array(results == test_label).astype(int)
        test_acc = np.mean(is_correct)
        return test_acc


class TDCAlayer(nn.Module):
    def __init__(self, KLG_FIX=True, EEGDATA=None, channels=None, DL=True, n_components=1, output=0, dropout_p=0):
        super().__init__()
        if EEGDATA is None:
            raise ValueError('EEGDATA is required for TDCAlayer.')
        self.EEGDATA = EEGDATA
        self.DL = DL
        self.output = output
        if EEGDATA.ndim == 5:
            self.Nf, self.Nm, self.Nc, self.T, _ = self.EEGDATA.shape
        else:
            self.Nf, self.Nm, self.Nc, self.Nf, self.T, _ = self.EEGDATA.shape
        self.channels = list(range(self.Nc)) if channels is None else channels
        self.n_components = n_components
        self.trains = np.mean(self.EEGDATA, -1)
        self.W = torch.zeros((self.Nm, self.n_components, self.Nc))
        self.M = np.zeros((self.Nm, self.Nc, self.T))
        self.is_ensemble = False

        if self.DL:
            self.bn = nn.BatchNorm2d(self.n_components)
            self.act = nn.PReLU(num_parameters=self.n_components)
            self.dropout = nn.Dropout(p=dropout_p)

        if KLG_FIX:
            self.W = nn.Parameter(self.W, requires_grad=False)
            self.trains = nn.Parameter(torch.tensor(self.trains, dtype=torch.float32), requires_grad=False)
        else:
            self.W = nn.Parameter(self.W, requires_grad=True)
            self.trains = nn.Parameter(torch.tensor(self.trains, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        if self.DL:
            if x.ndim != 4:
                raise ValueError(f'X.ndim must be 4 but got {x.shape}')
            outputs = []
            for fb_i in range(self.Nm):
                comp_outs = []
                for comp_i in range(self.n_components):
                    proj = torch.matmul(
                        x[:, fb_i].transpose(1, 2),
                        self.W[fb_i, comp_i].view(-1, 1),
                    ).transpose(1, 2)
                    comp_outs.append(proj)
                fb_out = torch.cat(comp_outs, dim=1).unsqueeze(2)
                fb_out = self.bn(fb_out)
                fb_out = self.act(fb_out)
                fb_out = self.dropout(fb_out)
                outputs.append(fb_out)
            return torch.cat(outputs, dim=2).transpose(1, 2)

        if x.ndim != self.trains.ndim:
            raise ValueError('X.ndim != trains.ndim')
        device = x.device if isinstance(x, torch.Tensor) else self.W.device
        r = torch.zeros((len(x), self.Nm, self.Nf), device=device)
        for trial in range(len(x)):
            for class_i in range(self.Nf):
                for fb_i in range(self.Nm):
                    if self.trains.ndim == 5:
                        test = self.W[fb_i] @ x[trial, fb_i, :, class_i, :]
                        train = self.W[fb_i] @ self.trains[class_i, fb_i, :, class_i, :]
                    else:
                        test = self.W[fb_i] @ x[trial, fb_i, :, :]
                        train = self.W[fb_i] @ self.trains[class_i, fb_i, :, :]
                    r_tmp = torch.corrcoef(torch.stack((test.flatten(), train.flatten())))[0, 1]
                    r[trial, fb_i, class_i] = r_tmp
        return r

    def forward2(self, x):
        if not self.DL:
            raise RuntimeError('forward2 requires DL=True')
        if x.ndim != 4:
            raise ValueError(f'X.ndim must be 4 but got {x.shape}')
        outputs = []
        for fb_i in range(self.Nm):
            comp_outs = []
            for comp_i in range(self.n_components):
                proj = torch.matmul(
                    x[:, fb_i].transpose(1, 2),
                    self.W[fb_i, comp_i].view(-1, 1),
                ).transpose(1, 2)
                comp_outs.append(proj)
            fb_out = torch.cat(comp_outs, dim=1).unsqueeze(2)
            fb_out = self.bn(fb_out)
            fb_out = self.act(fb_out)
            outputs.append(fb_out)
        return torch.cat(outputs, dim=2).transpose(1, 2)

    def tdca(self, X=None, labels=None):
        for fb_i in range(self.Nm):
            x_fb = X[:, fb_i, ...]
            x_fb = x_fb - np.mean(x_fb, axis=-1, keepdims=True)
            y_labels = np.unique(labels)
            if x_fb.ndim == 3:
                self.M[fb_i] = np.mean(x_fb, axis=0)
                x_fb = x_fb[:, self.channels, :]
            else:
                raise ValueError(f'X.ndim must be 3 but got {x_fb.shape}')

            n_labels = np.array([np.sum(labels == label) for label in y_labels])
            class_means, class_scatter = zip(
                *[
                    (
                        np.mean(x_fb[labels == label], axis=0),
                        np.sum(
                            np.matmul(x_fb[labels == label], np.swapaxes(x_fb[labels == label], -1, -2)),
                            axis=0,
                        ),
                    )
                    for label in y_labels
                ]
            )
            class_means = np.stack(class_means)
            class_scatter = np.stack(class_scatter)
            sw = np.sum(
                class_scatter - n_labels[:, np.newaxis, np.newaxis] * np.matmul(class_means, np.swapaxes(class_means, -1, -2)),
                axis=0,
            )
            centered_means = class_means - self.M[fb_i, self.channels]
            sb = np.sum(
                n_labels[:, np.newaxis, np.newaxis] * np.matmul(centered_means, np.swapaxes(centered_means, -1, -2)),
                axis=0,
            )
            eps = 1e-6
            sw_reg = sw + eps * np.eye(sw.shape[0])
            eigvals, eigvecs = eigh(sb, sw_reg)
            sorted_indices = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sorted_indices]
            with torch.no_grad():
                self.W[fb_i, :, self.channels] = torch.tensor(eigvecs.T[:self.n_components], dtype=torch.float32)
        return self

    def test_tdca(self, eeg, test_label):
        num_trials = eeg.shape[-1]
        fb_coefs = np.array([math.pow(i, -1.25) + 0.25 for i in range(1, self.Nm + 1)])
        results = np.zeros((self.Nf, num_trials))
        for targ_i in range(self.Nf):
            if eeg.ndim != 5:
                raise ValueError(f'X.ndim must be 5 but got {eeg.shape}')
            test_tmp = eeg[targ_i, :, :, :, :]
            r_mat = np.zeros((self.Nm, self.Nf, num_trials))
            for fb_i in range(self.Nm):
                testdata = test_tmp[fb_i, :, :, :]
                for class_i in range(self.Nf):
                    traindata = self.trains[class_i, fb_i, :, :]
                    w_mat = self.W[fb_i, :, :]
                    for trial_i in range(num_trials):
                        if eeg.ndim == 5:
                            testdata_w = np.matmul(w_mat.detach().cpu().numpy(), testdata[:, :, trial_i] - self.M[fb_i])
                        else:
                            testdata_w = np.matmul(
                                w_mat.detach().cpu().numpy(),
                                testdata[:, class_i, :, trial_i] - self.M[fb_i],
                            )
                        traindata_w = np.matmul(w_mat.detach().cpu().numpy(), traindata.detach().cpu().numpy() - self.M[fb_i])
                        r_tmp = np.corrcoef(testdata_w.flatten(), traindata_w.flatten())
                        r_mat[fb_i, class_i, trial_i] = r_tmp[0, 1]
            rho = np.einsum('j,jkl->kl', fb_coefs, r_mat)
            self.output[targ_i] += rho
            results[targ_i, :] = np.argmax(self.output[targ_i], axis=0)
        is_correct = np.array(results == test_label).astype(int)
        test_acc = np.mean(is_correct)
        return test_acc

