#!/usr/bin/env python
"""Launch one run: copy a base configuration, apply the overrides, and run main.py on it.

    python run.py --name flow_both_bm40_a_f0 --base etc/benchmark.yaml --subjects 1,2,3,4,5,6,7,8,9 \
        --artifact both --folds 0 --augmenter FLOW_CLS

The configuration of the run is written to etc/generated/<name>.yaml, its log to logs/<name>.log, and
its results to <paths.outputs.root>/<dataset>/intra/augmentation/denoise/<time>__...__run-<name>__.../
(results/experiment_results.csv: decoding accuracy; results/denoise_results.csv: signal fidelity).
--mset KEY=VALUE sets augmentation.methods.<method>.KEY for every denoiser of the run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def coerce(value: str):
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name (at most 40 characters)")
    ap.add_argument("--base", default="etc/benchmark.yaml", help="base configuration")
    ap.add_argument("--gpu", default="0", help="CUDA device index, or cpu")
    ap.add_argument("--augmenter", default="FLOW_CLS",
                    help="comma-separated denoisers: FLOW_CLS (DenoFlow), ICA, EEGDfus, TOEDGAN, EEGDNet, "
                         "GCTNet, CT_DCENet, FCNN, SimpleCNN, ResCNN")
    ap.add_argument("--subjects", default=None, help="comma-separated subject ids (default: all)")
    ap.add_argument("--folds", default=None, help="comma-separated fold ids, or 'all'")
    ap.add_argument("--seed", type=int, default=42, help="experiment.random_seed")
    ap.add_argument("--artifact", default=None, help="emg, eog, or both (mixed)")
    ap.add_argument("--noise-ratio", type=float, default=None, help="mask fraction")
    ap.add_argument("--noise-snr", type=float, default=None, help="target ratio gamma in dB")
    ap.add_argument("--mset", action="append", default=[], metavar="KEY=VALUE",
                    help="augmentation.methods.<method>.KEY for every denoiser of the run")
    ap.add_argument("--background", action="store_true", help="detach and return immediately")
    ap.add_argument("--dry-run", action="store_true", help="write the configuration only")
    args = ap.parse_args()

    os.chdir(HERE)
    if len(args.name) > 40:
        sys.exit("--name must be at most 40 characters, since it is part of the results directory")
    with open(args.base, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ex = cfg["experiment"]
    ex["device"] = "cpu" if args.gpu == "cpu" else coerce(args.gpu)
    ex["methods"]["augmentation"] = [a.strip() for a in args.augmenter.split(",") if a.strip()]
    ex["run_name"] = args.name
    ex["random_seed"] = args.seed
    if args.subjects:
        ex["subject_ids"] = [int(s) for s in args.subjects.split(",") if s.strip()]
    if args.folds is not None:
        ex["fold_ids"] = [] if args.folds == "all" else [int(f) for f in args.folds.split(",") if f.strip()]

    noise = cfg["augmentation"]["noise"]
    if args.noise_ratio is not None:
        noise["ratio"] = args.noise_ratio
    if args.noise_snr is not None:
        noise["snr_db"] = args.noise_snr
    if args.artifact is not None:
        noise["artifact_type"] = args.artifact

    methods = cfg["augmentation"].setdefault("methods", {})
    for augmenter in ex["methods"]["augmentation"]:
        block = methods.setdefault(augmenter.lower(), {})
        for item in args.mset:
            key, sep, raw = item.partition("=")
            if not sep:
                sys.exit(f"bad --mset {item!r}, expected KEY=VALUE")
            block[key] = coerce(raw)

    os.makedirs("etc/generated", exist_ok=True)
    cfg_path = f"etc/generated/{args.name}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    print(f"[{args.name}] denoisers={args.augmenter} config={cfg_path}")
    if args.dry_run:
        return

    os.makedirs("logs", exist_ok=True)
    log = f"logs/{args.name}.log"
    cmd = [sys.executable, "-u", "main.py", "--f", cfg_path]
    with open(log, "wb") as out:
        if args.background:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            print(f"[{args.name}] started pid={proc.pid} log={log}")
            return
        code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
    print(f"[{args.name}] finished with exit code {code}, log={log}")
    sys.exit(code)


if __name__ == "__main__":
    main()
