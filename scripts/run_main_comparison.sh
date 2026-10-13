#!/usr/bin/env bash
# Runs behind Tables 4 to 7: DenoFlow and the nine baseline denoisers on both datasets under the three
# artifact types, each followed by the five downstream decoders, under the evaluation protocol of
# Section 4.3.1 (block-wise cross-validation, decoders evaluated after their final epoch).
#
#   scripts/run_main_comparison.sh               all datasets
#   scripts/run_main_comparison.sh bm12          Nakanishi2015 only (bm40 = Benchmark)
#   GPU=1 scripts/run_main_comparison.sh         CUDA device passed to run.py (default 0)
#   DRY=1 scripts/run_main_comparison.sh         print the commands instead of running them
#
# One run covers one fold of one group of subjects ("shard"). The contamination of a trial is drawn
# with the seed 42 + (position of its subject within the run), so the shards below must be kept to
# reproduce the contaminated trials of the paper. Every denoiser of a dataset, artifact type, and fold
# then receives the same contaminated trials. Runs are executed one after another; to spread them over
# several GPUs, print them with DRY=1 and distribute the lines.
#
# Run names, read by scripts/aggregate.py: <method>_<artifact>_<dataset>_<shard>_f<fold>.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
DRY=${DRY:-0}
PY=${PY:-python}
BASE_bm12=etc/nakanishi2015.yaml
BASE_bm40=etc/benchmark.yaml
declare -A SUBS=(
  [bm12_a]=1,2,3,4 [bm12_b]=5,6,7 [bm12_c]=8,9,10
  [bm40_a]=1,2,3,4,5,6,7,8,9 [bm40_b]=10,11,12,13,14,15,16,17,18
  [bm40_c]=19,20,21,22,23,24,25,26,27 [bm40_d]=28,29,30,31,32,33,34,35 )
SHARDS_bm12="a b c";   FOLDS_bm12="0 1 2 3 4"       # five-fold: 3 of the 15 blocks are tested per fold
SHARDS_bm40="a b c d"; FOLDS_bm40="0 1 2 3 4 5"     # six-fold: 1 of the 6 blocks is tested per fold

# run-name tag -> denoisers of the run
declare -A METHODS=(
  [flow]=FLOW_CLS                                                   # DenoFlow
  [base]=TOEDGAN,EEGDNet,GCTNet,CT_DCENet,FCNN,SimpleCNN,ResCNN      # seven published deep baselines
  [ica]=ICA
  [dfus]=EEGDfus )

datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(bm12 bm40)
for ds in "${datasets[@]}"; do
  case "$ds" in bm12|bm40) ;; *) echo "unknown dataset: $ds (use bm12 or bm40)" >&2; exit 1 ;; esac
  base_var="BASE_$ds"; shards_var="SHARDS_$ds"; folds_var="FOLDS_$ds"
  for art in emg eog both; do
    for tag in flow base ica dfus; do
      for f in ${!folds_var}; do
        for sh in ${!shards_var}; do
          cmd=("$PY" run.py --name "${tag}_${art}_${ds}_${sh}_f${f}" --gpu "$GPU" --base "${!base_var}"
               --subjects "${SUBS[${ds}_${sh}]}" --artifact "$art" --noise-ratio 0.5 --noise-snr 0 --seed 42
               --folds "$f" --augmenter "${METHODS[$tag]}")
          if [ "$DRY" = 1 ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
        done
      done
    done
  done
done
