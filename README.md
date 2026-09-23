<div align="center">
<h1>DenoFlow</h1>
<h3>Flow Matching for SSVEP Denoising under Real Physiological Artifacts</h3>

Zhentao He<sup>1,†</sup>, [Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1,†</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>†</sup> Z. He and Z. Wang contributed equally to this work.

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

<sup>:email:</sup> Corresponding Author

</div>

> This repository contains the implementation of our paper: **"DenoFlow: Flow Matching for SSVEP Denoising under Real Physiological Artifacts"**. Although recent denoising approaches have shown promise, they are fitted without paired ground truth, can settle on reproducing their input, and are optimized on waveform distance alone, which says nothing about whether the output stays decodable. To address these issues, we propose DenoFlow, which casts SSVEP denoising as transport: instead of learning a direct map from a contaminated trial to a clean one, a field network regresses the velocity of the straight path between them, following the rectified-flow formulation, and denoising integrates that field forward from the observation. The field network is an encoder-decoder that sees the contaminated trial at every layer and the path position at its bottleneck, and a classifier trained alongside it supervises the integrated output. Because the observation itself is both the conditioning input and the starting point of the integration, the model never generates a trial from noise, and training reduces to regression, removing the adversarial min-max game. To obtain paired data on datasets with no ground truth, artifacts are injected from recorded EMG and EOG epochs under a controlled signal-to-noise target. Experiments on two public SSVEP datasets show that DenoFlow improves both signal fidelity and downstream decoding accuracy over competing denoisers.

## Overview

**DenoFlow** learns the *path* from a contaminated trial to its clean counterpart rather than a
direct mapping between them.

A trial $X$ becomes a contaminated trial $X'$ once an artifact is injected. Both pass through the
same filter bank, giving the reference $\tilde{X}$ and the input $\tilde{X}'$. The straight path
between them is $x_t = (1-t)\tilde{X}' + t\tilde{X}$, whose velocity $\tilde{X} - \tilde{X}'$ is
constant, and the field network $f_\theta$ regresses that velocity at a position drawn uniformly
at random. Denoising integrates the field forward from the observation with $S$ Euler steps,

$$\hat{X}^{(s+1)} = \hat{X}^{(s)} + \tfrac{1}{S} f_\theta\big(\hat{X}^{(s)}, \tilde{X}', \tfrac{s}{S}\big), \qquad \hat{X}^{(0)} = \tilde{X}'.$$

Three terms make up the objective — velocity regression, reconstruction on the integrated output,
and a class term from a jointly trained classifier:

$$\mathcal{L} = \lambda_{\text{flow}}\mathcal{L}_{\text{flow}} + \frac{\lambda_{\text{rec}}}{N_cT}\lVert\hat{X}-\tilde{X}\rVert_F^2 + \lambda_{\text{cls}}\\,\mathrm{CE}\big(C(\hat{X}), y\big).$$

Two properties follow from taking the observation as the origin of the path. The model never has
to generate a trial from noise — it only has to move a trial it is given — and training is plain
regression, with no adversarial min-max game. The step count $S$ enters only through the
integrator, so it can be changed after training without touching the field.

## Artifact Injection

Public SSVEP datasets record the contaminated signal only, so the pairs have to be built. Segments
are drawn from the EMG and EOG libraries of
[EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) and spread over the electrodes by a
log-normal gain, a coarse stand-in for volume conduction. A binary mask $M$ selects the
contaminated region, and each segment is scaled so that inside the mask it meets a target ratio
$\gamma$:

$$\alpha_k=\sqrt{\frac{\mathbb{E}[(X\odot M)^2]}{\mathbb{E}[(A_k\odot M)^2]\cdot 10^{\gamma/10}}},\qquad X'=X+\Big(\sum_k \alpha_k A_k\Big)\odot M.$$

Three conditions are used: EMG only, EOG only, and mixed. Severity is set by the mask fraction and
by $\gamma$; the reference condition throughout the paper is a mask fraction of $0.5$ and
$\gamma = 0$ dB.

## Baselines

Seven published denoising models were retrained under the same protocol and compared with
DenoFlow — ResCNN, FCNN, SimpleCNN, EEGDNet, GCTNet, TOED-GAN and CT-DCENet — of which TOED-GAN is
the only one designed for SSVEP. Denoised trials were decoded by five SSVEP decoders: FBCCA, eTRCA,
EEGNet, CCNN and SSVEPNet.

DenoFlow ranked first on all three fidelity metrics under all three artifacts on both datasets, and
was the only method to improve on leaving the trial untouched under every artifact. On Benchmark
under mixed contamination it lifts the average accuracy over the five decoders from $62.29\\%$
(contaminated) to $71.91\\%$, against $79.71\\%$ on the uncontaminated reference.

## Evaluation Tasks

Denoising is assessed at two levels:

- **Signal fidelity** — the correlation coefficient (CC), the relative root-mean-square error
  (RRMSE), and the improvement in signal-to-noise ratio (SNRI) of $\hat{X}$ against $\tilde{X}$.
- **Downstream decoding** — the denoised trial is handed to an ordinary SSVEP decoder and scored by
  classification accuracy, since the point is a trial that still decodes rather than a smaller
  waveform error.

All experiments are within subject, with a chronological fold split: five-fold on Nakanishi2015 and
six-fold on Benchmark.

## Getting Started

### Installation

```bash
git clone https://github.com/wzwvv/DenoFlow.git
cd DenoFlow
pip install -r requirements.txt
```

### Data

Place the datasets and the artifact libraries under `Data/`:

```
Data/
├── EMG_all_epochs.npy          # from EEGdenoiseNet
├── EOG_all_epochs.npy          # from EEGdenoiseNet
├── Nakanishi2015/S1.mat ...    # 10 subjects, 8 channels, 256 Hz, 12 classes
└── Benchmark/S1.mat ...        # 35 subjects, 9 posterior channels, 250 Hz, 40 classes
```

- [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) — the EMG and EOG epochs.
- [Nakanishi2015](https://github.com/mnakanishi/12JFPM_SSVEP) — 12-class SSVEP.
- [Benchmark](http://bci.med.tsinghua.edu.cn/download.html) — 40-class SSVEP; the nine posterior
  channels (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2) are selected in `Utils/dataset.py`.

### Usage

```bash
# train the field network on one subject and report signal fidelity
python main.py --dataset benchmark --subject 1 --fold 0 --artifact both

# also train downstream decoders on the denoised trials
python main.py --dataset benchmark --subject 1 --artifact both --decode

# vary the contamination: mask fraction and target ratio
python main.py --dataset nakanishi2015 --subject 1 --artifact eog --ratio 0.75 --snr -5
```

Defaults live in `etc/config.yaml` and every one of them can be overridden on the command line.
The settings used in the paper are the file's defaults: $\lambda_{\text{flow}} =
\lambda_{\text{rec}} = \lambda_{\text{cls}} = 1$, $S_{\text{tr}} = S = 4$, Adam at $10^{-3}$ with
weight decay $10^{-4}$, batch size 36, 300 epochs, and a filter bank of three Chebyshev type-I
sub-bands applied with zero-phase forward-backward filtering.

Setting `classification_weight: 0` drops the class term, `sample_weight: 0` the reconstruction
term, and `flow_weight: 0` the velocity term — the three rows of the ablation.

## Code Structure

```
DenoFlow/
│
├── etc/
│   └── config.yaml              # datasets, contamination, architecture, objective, training
│
├── Models/
│   ├── DenoFlow.py              # FlowDenoiser: velocity field + Euler integrator
│   │                            #   AltFieldNet — interleaved spatial/temporal encoder-decoder
│   │                            #   around a Transformer bottleneck, conditioned on the
│   │                            #   contaminated trial at every layer
│   ├── Classifier.py            # auxiliary classifier that supervises the integrated output
│   │
│   └── Decoders/                # downstream SSVEP decoders used to score the denoised trials
│       ├── FBCCA.py
│       ├── TRCA.py
│       ├── EEGNet.py
│       ├── CCNN.py
│       └── SSVEPNet.py
│
├── Utils/
│   ├── inject.py                # artifact injection from recorded EMG/EOG at a target SNR
│   ├── filterbank.py            # Chebyshev type-I sub-bands, zero-phase
│   ├── dataset.py               # subject loading, chronological folds, paired views
│   ├── trainer.py               # the DenoFlow training loop and the Euler integrator
│   ├── decode.py                # downstream decoding of the denoised trials
│   └── metrics.py               # CC / RRMSE / SNR / SNRI
│
└── main.py                      # entry point
```

---

## 📄 Citation

If you find this work helpful, please stay tuned for our full paper, which will be available on
arXiv soon. We appreciate your interest and patience. Feel free to raise issues or pull requests
for questions or improvements.
