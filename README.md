<div align="center">
<h1>DenoFlow</h1>
<h3>Flow Matching for SSVEP Denoising under Recorded Physiological Artifacts</h3>

Zhentao He<sup>1,†</sup>, [Ziwei Wang](https://scholar.google.com/citations?user=fjlXqvQAAAAJ&hl=en)<sup>1,†</sup>, and [Dongrui Wu](https://scholar.google.com/citations?user=UYGzCPEAAAAJ&hl=en)<sup>1 :email:</sup>

<sup>†</sup> Z. He and Z. Wang contributed equally to this work.

<sup>1</sup> School of Artificial Intelligence and Automation, Huazhong University of Science and Technology

<sup>:email:</sup> Corresponding Author

</div>

This repository contains the code of DenoFlow and the scripts that reproduce the main comparison of the
paper (Tables 4 to 7): DenoFlow and nine baseline denoisers on two public SSVEP datasets under three
artifact types, evaluated by signal fidelity and by the accuracy of five downstream decoders, together
with the statistical tests.

DenoFlow casts SSVEP denoising as transport from a contaminated trial to its reference. A field network
regresses the velocity of the straight path between the filtered contaminated trial and the filtered
reference, following the rectified-flow formulation, and the denoised trial is obtained by integrating
this field forward from the observation with a few Euler steps. Training combines velocity regression
along the path, reconstruction of the integrated output, and a classification loss from a jointly trained
classifier, without an adversarial game.

## Contents

| | |
|---|---|
| Denoisers | DenoFlow and the nine baselines ICA, ResCNN, FCNN, SimpleCNN, EEGDNet, GCTNet, TOED-GAN, CT-DCENet, and EEGDfus |
| Datasets | Nakanishi2015 (12 classes, 10 subjects) and Benchmark (40 classes, 35 subjects) |
| Contamination | recorded EMG, EOG, and mixed EMG-EOG artifact segments injected into the recorded trials |
| Decoders | FBCCA (individual templates), eTRCA, EEGNet, CCNN, SSVEPNet |
| Metrics | CC, RRMSE, and SNRI against the filtered reference, and decoding accuracy |

## Installation

The experiments of the paper were run with Python 3.10, PyTorch 2.3.1 with CUDA 12.1, scikit-learn
1.5.2, and SciPy 1.9.3 on NVIDIA GeForce RTX 3090 GPUs.

```bash
git clone https://github.com/wzwvv/DenoFlow.git
cd DenoFlow
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default one does not.

## Data

The datasets and the artifact libraries are not included. Download them and arrange them as follows:

```
DenoFlow/
├── datasets/
│   ├── Nakanishi2015/S1.mat ... S10.mat
│   └── Benchmark/S1.mat ... S35.mat
└── Data/                      # code folder of the repository
    ├── EMG_all_epochs.npy
    └── EOG_all_epochs.npy
```

- Nakanishi2015: https://github.com/mnakanishi/12JFPM_SSVEP (the `.mat` files of the 10 subjects).
- Benchmark: http://bci.med.tsinghua.edu.cn/download.html (the `.mat` files of the 35 subjects). The
  nine posterior channels Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, and O2 are selected in `etc/benchmark.yaml`.
- EMG and EOG libraries: `EMG_all_epochs.npy` and `EOG_all_epochs.npy` of
  [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet).

The dataset folders can be changed under `paths.datasets` in `etc/nakanishi2015.yaml` and
`etc/benchmark.yaml`. In the code, Nakanishi2015 is called `benchmark12` and Benchmark `benchmark40`.

## Running one experiment

`run.py` writes the configuration of a run from a base file and its arguments, and runs it:

```bash
python run.py --name flow_both_bm40_a_f0 --base etc/benchmark.yaml --gpu 0 \
    --subjects 1,2,3,4,5,6,7,8,9 --artifact both --noise-ratio 0.5 --noise-snr 0 --folds 0 \
    --augmenter FLOW_CLS
```

This trains DenoFlow on the training split of fold 0 for subjects 1 to 9 of Benchmark under mixed
contamination, denoises the training and test trials, and trains and tests the five decoders on the
uncontaminated (Ref.), contaminated (Art.), and denoised trials. `--augmenter` accepts `FLOW_CLS`
(DenoFlow), `ICA`, `EEGDfus`, `TOEDGAN`, `EEGDNet`, `GCTNet`, `CT_DCENet`, `FCNN`, `SimpleCNN`, and `ResCNN`,
or a comma-separated list of them. The configuration of the run is saved to `etc/generated/<name>.yaml`
and its log to `logs/<name>.log`. Results are written to
`results/<dataset>/intra/augmentation/denoise/<time>__...__run-<name>__.../results/`:

- `experiment_results.csv`: one row per subject, fold, view, and decoder, with the test accuracy
  (`train_view` is `generated` for the denoised trials, `noisy` for Art., and `clean` for Ref.);
- `denoise_results.csv`: one row per subject, fold, and denoiser, with the fidelity metrics of the test
  trials (`cc`, `rrmse`, `snri`).

## Reproducing Tables 4 to 7

```bash
bash scripts/run_main_comparison.sh            # both datasets; or: bm12 (Nakanishi2015), bm40 (Benchmark)
python scripts/aggregate.py                    # writes results/paper_tables/
```

`run_main_comparison.sh` launches 468 runs, one for each combination of dataset, artifact type, denoiser
group, fold, and group of subjects: 180 on Nakanishi2015 (3 artifact types x 4 denoiser groups x 5 folds x
3 subject groups) and 288 on Benchmark (3 x 4 x 6 x 4). The runs are executed one after another on the GPU
given by `GPU` (default 0). To spread them over several GPUs, print the commands with
`DRY=1 bash scripts/run_main_comparison.sh` and distribute the lines.

`aggregate.py` collects all runs and writes

- `summary.md`: Tables 4 to 7 with significance markers, and the numbers derived from them that the paper
  quotes (ranking over the 18 fidelity combinations, accuracy gains over the contaminated trials, margins
  over the strongest baseline, and gains per decoder);
- `tables.json`: every mean, standard deviation, and corrected p-value;
- `subjects.csv`: the fold-averaged value of every subject, method, and metric.

A subject enters the tables only when all of its folds are present. While runs are still in progress,
`--allow-partial` averages the folds that exist and lists the missing ones.

## Evaluation protocol

- **Data split.** All experiments are within subject, and the trials are split block-wise. Nakanishi2015
  (15 blocks) uses five-fold cross-validation with three consecutive blocks held out for testing in each
  fold, and Benchmark (6 blocks) uses six-fold cross-validation with one block held out. Each block is
  tested exactly once.
- **Contamination.** One recorded segment per library is injected over a temporal mask covering half of
  the trial, scaled so that the ratio of trial to artifact power within the mask is 0 dB, and spread over
  the electrodes with log-normal gains (sigma = 0.3). Training and test trials are contaminated
  independently, once per fold, with the seed 42 plus the position of the subject within its run, so all
  denoisers receive the same contaminated trials. The groups of subjects in
  `scripts/run_main_comparison.sh` must therefore be kept to reproduce the contaminated trials of the paper.
- **Denoisers.** Every denoiser is trained on the contaminated training trials and the corresponding
  references only, and is evaluated with the weights of its final training epoch.
- **Decoders.** Each decoder is trained on the training split and tested on the test split of the same
  view (reference, contaminated, or denoised trials). FBCCA uses individual templates, i.e., class-wise
  averages of the training trials, and the eTRCA spatial filters and templates are estimated from the
  training split. EEGNet, CCNN, and SSVEPNet are trained from scratch for every view, denoiser, and fold
  for a fixed number of epochs without early stopping, and the accuracy after the final epoch is reported.
  No test trial is used for training or model selection.
- **Statistics.** Values are averaged over the folds of each subject. Tables report the mean and standard
  deviation over subjects. DenoFlow is compared with each of the nine baselines by a two-sided Wilcoxon
  signed-rank test paired over subjects, Holm-corrected within each family of nine comparisons (one
  fidelity metric, or the average accuracy over the five decoders, under one artifact type on one dataset).

## Configuration

`etc/nakanishi2015.yaml` and `etc/benchmark.yaml` hold every setting of the experiments. The settings of
DenoFlow are under `augmentation.methods.flow_cls`, with `train_sample_steps` and `flow_steps` the numbers
of Euler steps used in training and at inference (4 and 4), and the weights `flow_weight`, `sample_weight`,
and `classification_weight` of the three loss terms left at their default of 1. The baselines read their
settings from `augmentation.methods.<name>` and, for TOED-GAN, from `augmentation.stateful.defaults`. The
decoders read theirs from `classification_models`. `run.py --mset KEY=VALUE` overrides
`augmentation.methods.<method>.KEY` for one run.

## Code structure

```
DenoFlow/
├── run.py                        # writes the configuration of a run and runs main.py on it
├── main.py                       # runs the experiment of one configuration file
├── etc/                          # base configurations of the two datasets
├── scripts/
│   ├── run_main_comparison.sh    # every run behind Tables 4 to 7
│   └── aggregate.py              # tables and statistical tests
├── Data/                         # loading, block-wise folds, filter bank, artifact injection
├── Support/                      # experiment loop, configuration access, fidelity metrics
├── Trainers/
│   ├── Augmentation/             # training of DenoFlow (generative.py) and of the baselines
│   └── Classification/           # training and testing of the decoders
└── Models/
    ├── Augmentation/             # DenoFlow (generative.py: FlowDenoiser, AltFieldNet) and the baselines
    └── Classification/           # decoders
```

Names in the code: `FLOW_CLS` is DenoFlow, `TOEDGAN` is TOED-GAN, `CT_DCENet` is CT-DCENet, `TRCA` is
eTRCA, `benchmark12` is Nakanishi2015, and `benchmark40` is Benchmark. The folders of the baselines under
`Models/Augmentation/` also keep files of their original implementations that the experiments do not use.

Small numerical differences from the paper can occur with other GPUs or library versions.

## Citation

The paper is under review. The citation will be added when it is published.
