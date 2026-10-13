#!/usr/bin/env python
"""Tables 4 to 7 and their statistical tests, from the runs of scripts/run_main_comparison.sh.

    python scripts/aggregate.py                                # results/ -> results/paper_tables/
    python scripts/aggregate.py --root results --out results/paper_tables --allow-partial

Aggregation (Sections 4.3.2 and 4.4.1):
  * each value is first obtained per subject and fold: the fidelity metrics are averaged over the test
    trials (and over the rows of the filter-bank trial), and the accuracy is the fraction of correctly
    decoded test trials of a decoder;
  * the values of a subject are averaged over its folds, and the tables report the mean and the sample
    standard deviation over subjects; the average accuracy (Avg.) is the mean over the five decoders;
  * two-sided Wilcoxon signed-rank tests compare DenoFlow with each of the nine baseline denoisers,
    paired over subjects, and the p-values are Holm-corrected within each family of nine comparisons,
    a family being one fidelity metric or the average accuracy under one artifact type on one dataset;
  * * p < 0.05, ** p < 0.005, *** p < 0.001 after correction.
Ref. and Art. decode the uncontaminated and the contaminated trials and are taken from the DenoFlow runs.
By default a subject enters the tables only when all of its folds are present; --allow-partial averages
the folds that exist (useful while runs are still in progress) and lists the missing ones.

Outputs: summary.md (the four tables and the numbers quoted in the text), tables.json (all means,
standard deviations, and corrected p-values), subjects.csv (the fold-averaged value of every subject).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

DATASETS = {"bm12": ("benchmark12", "Nakanishi2015", 5), "bm40": ("benchmark40", "Benchmark", 6)}
DECODERS = ["FBCCA", "TRCA", "EEGNet", "CCNN", "SSVEPNet"]
DECODER_NAME = {"TRCA": "eTRCA"}
METRICS = ["cc", "rrmse", "snri"]
METRIC_NAME = {"cc": "CC", "rrmse": "RRMSE", "snri": "SNRI (dB)"}
HIGHER = {"cc": True, "rrmse": False, "snri": True}
DIGITS = {"cc": 4, "rrmse": 4, "snri": 2}
ARTS = ["emg", "eog", "both"]
ART_NAME = {"emg": "EMG", "eog": "EOG", "both": "Mixed"}
BASELINES = ["ResCNN", "FCNN", "SimpleCNN", "EEGDNet", "GCTNet", "TOEDGAN", "CT_DCENet"]
COMPARATORS = ["ICA"] + BASELINES + ["EEGDfus"]
OURS = "FLOW_CLS"
NAME = {"TOEDGAN": "TOED-GAN", "CT_DCENet": "CT-DCENet", "FLOW_CLS": "DenoFlow"}
RUNS = {"flow": [OURS], "base": BASELINES, "ica": ["ICA"], "dfus": ["EEGDfus"]}   # run tag -> denoisers
RUN_RE = re.compile(r"__run-(?P<name>.+?)__")
MISSING = defaultdict(set)


# ---------------------------------------------------------------- loading
def load_runs(root, prefix, tag, art, ds):
    """Fold-level values of all runs <prefix><tag>_<art>_<ds>_<shard>_f<fold>.

    Returns gen[denoiser][key][(subject, fold)] with key a decoder or a fidelity metric, and the context
    views ctx['noisy' | 'clean'][decoder][(subject, fold)]. When a cell appears more than once (a run that
    was started again), the newest row is kept."""
    base = os.path.join(root, DATASETS[ds][0], "intra", "augmentation", "denoise")
    want = re.compile(rf"^{re.escape(prefix)}{tag}_{art}_{ds}_[a-z]_f\d+$")
    gen = defaultdict(lambda: defaultdict(dict))
    ctx = defaultdict(lambda: defaultdict(dict))
    stamp = {}

    def put(table, key, sf, value, ts):
        if key not in stamp or ts >= stamp[key]:
            stamp[key] = ts
            table[sf] = value

    for d in sorted(glob.glob(os.path.join(base, f"*__run-{prefix}{tag}_{art}_{ds}_*"))):
        m = RUN_RE.search(os.path.basename(d) + "__")
        if not m or not want.match(m.group("name")):
            continue
        path = os.path.join(d, "results", "experiment_results.csv")
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8")):
                sf, k, ts = (int(r["subject"]), int(r["fold"])), r["downstream_method"], r["timestamp"]
                if r["train_view"] == "generated" and r["augmenter"]:
                    put(gen[r["augmenter"]][k], ("g", r["augmenter"], k, sf), sf, float(r["acc"]), ts)
                elif not r["augmenter"] and r["train_view"] in ("noisy", "clean"):
                    put(ctx[r["train_view"]][k], ("c", r["train_view"], k, sf), sf, float(r["acc"]), ts)
        path = os.path.join(d, "results", "denoise_results.csv")
        if os.path.exists(path):
            for r in csv.DictReader(open(path, encoding="utf-8")):
                sf, ts = (int(r["subject"]), int(r["fold"])), r["timestamp"]
                for mt in METRICS:
                    put(gen[r["augmenter"]][mt], ("f", r["augmenter"], mt, sf), sf, float(r[mt]), ts)
    return gen, ctx


def per_subject(table, ds, label, allow_partial):
    """{key: {(subject, fold): v}} -> {key: {subject: mean over folds}}, accuracy in %, plus Avg."""
    nfolds = DATASETS[ds][2]
    out = {}
    for key, values in table.items():
        by_subject = defaultdict(dict)
        for (s, f), v in values.items():
            by_subject[s][f] = v
        out[key] = {}
        for s, folds in by_subject.items():
            if len(folds) < nfolds:
                MISSING[(ds, label)].add(s)
                if not allow_partial:
                    continue
            v = float(np.mean(list(folds.values())))
            out[key][s] = 100 * v if key in DECODERS else v
    if all(k in out for k in DECODERS):
        common = set.intersection(*[set(out[k]) for k in DECODERS])
        out["Avg"] = {s: float(np.mean([out[k][s] for k in DECODERS])) for s in common}
    return out


# ---------------------------------------------------------------- statistics
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_p(a, b):
    """Two-sided Wilcoxon signed-rank test (SciPy defaults) over the subjects present in both."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    if all(u == v for u, v in zip(x, y)):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def stars(p):
    if p is None or p != p:
        return ""
    return "***" if p < 1e-3 else "**" if p < 5e-3 else "*" if p < 0.05 else ""


def mean_sd(values):
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return [float("nan"), float("nan"), 0]
    return [float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan"), int(v.size)]


def fmt(ms, digits):
    m, s = ms[0], ms[1]
    if m != m:
        return "n/a"
    return f"{m:.{digits}f}±{s:.{digits}f}" if s == s else f"{m:.{digits}f}"


# ---------------------------------------------------------------- tables
def collect(args):
    cells, subjects = {}, {}
    for ds in DATASETS:
        for art in ARTS:
            sv = {}
            for tag, denoisers in RUNS.items():
                gen, ctx = load_runs(args.root, args.prefix, tag, art, ds)
                for m in denoisers:
                    sv[m] = per_subject(gen.get(m, {}), ds, f"{NAME.get(m, m)} {ART_NAME[art]}", args.allow_partial)
                if tag == "flow":
                    sv["Ref."] = per_subject(ctx.get("clean", {}), ds, f"Ref. {ART_NAME[art]}", args.allow_partial)
                    sv["Art."] = per_subject(ctx.get("noisy", {}), ds, f"Art. {ART_NAME[art]}", args.allow_partial)
            cell = {"mean": {}, "p": {}}
            for key in METRICS + DECODERS + ["Avg"]:
                raw = [paired_p(sv[OURS].get(key, {}), sv[c].get(key, {})) for c in COMPARATORS]
                ok = [i for i, p in enumerate(raw) if p == p]
                cell["p"][key] = dict(zip([COMPARATORS[i] for i in ok], holm([raw[i] for i in ok])))
                for m in sv:
                    if key in METRICS and m in ("Ref.", "Art."):
                        continue
                    cell["mean"].setdefault(m, {})[key] = mean_sd(sv[m].get(key, {}).values())
            cells[f"{ds}_{art}"] = cell
            subjects[f"{ds}_{art}"] = sv
    return cells, subjects


def report_tables(cells):
    out = []
    for number, ds in ((4, "bm12"), (5, "bm40")):
        out.append(f"\n## Table {number}: signal-level fidelity on {DATASETS[ds][1]}\n")
        out.append("| Artifact | Method | CC | RRMSE | SNRI (dB) |\n|---|---|---|---|---|")
        for art in ARTS:
            c = cells[f"{ds}_{art}"]
            for m in COMPARATORS + [OURS]:
                cols = [fmt(c["mean"].get(m, {}).get(mt, [float("nan")] * 3), DIGITS[mt])
                        + ("" if m == OURS else stars(c["p"][mt].get(m))) for mt in METRICS]
                out.append(f"| {ART_NAME[art]} | {NAME.get(m, m)} | " + " | ".join(cols) + " |")
    for number, ds in ((6, "bm12"), (7, "bm40")):
        out.append(f"\n## Table {number}: decoding accuracy (%) on {DATASETS[ds][1]}\n")
        head = ["Ref.", "Art."] + COMPARATORS + [OURS]
        out.append("| Artifact | Decoder | " + " | ".join(NAME.get(m, m) for m in head) + " |")
        out.append("|---|---|" + "---|" * len(head))
        for art in ARTS:
            c = cells[f"{ds}_{art}"]
            for k in DECODERS + ["Avg"]:
                cols = [fmt(c["mean"].get(m, {}).get(k, [float("nan")] * 3), 2)
                        + (stars(c["p"]["Avg"].get(m)) if k == "Avg" and m in COMPARATORS else "") for m in head]
                label = "Avg." if k == "Avg" else DECODER_NAME.get(k, k)
                out.append(f"| {ART_NAME[art]} | {label} | " + " | ".join(cols) + " |")
    return out


def report_numbers(cells):
    """Numbers derived from Tables 4 to 7 that the abstract and Sections 4.4.1 and 4.4.2 quote."""
    out = ["\n## Numbers quoted in the text\n"]
    get = lambda c, m, k: c["mean"].get(m, {}).get(k, [float("nan")])[0]
    first, total = 0, 0
    for ds in DATASETS:
        for art in ARTS:
            c = cells[f"{ds}_{art}"]
            for mt in METRICS:
                vals = {m: get(c, m, mt) for m in COMPARATORS + [OURS]}
                vals = {m: v for m, v in vals.items() if v == v}
                if OURS not in vals:
                    continue
                total += 1
                best = (max if HIGHER[mt] else min)(vals, key=vals.get)
                if best == OURS:
                    first += 1
                else:
                    out.append(f"- {DATASETS[ds][1]}, {ART_NAME[art]}, {METRIC_NAME[mt]}: best is {NAME.get(best, best)}")
    out.append(f"- DenoFlow ranks first in {first} of {total} combinations of fidelity metric, artifact type, and dataset.")
    for ds in DATASETS:
        for art in ARTS:
            c = cells[f"{ds}_{art}"]
            avg = {m: get(c, m, "Avg") for m in COMPARATORS + [OURS, "Ref.", "Art."]}
            strongest = max(COMPARATORS, key=lambda m: avg[m] if avg[m] == avg[m] else -1.0)
            out.append(f"- {DATASETS[ds][1]}, {ART_NAME[art]}: Avg. DenoFlow {avg[OURS]:.2f}%, Art. {avg['Art.']:.2f}% "
                       f"(gain {avg[OURS] - avg['Art.']:.2f}%), Ref. {avg['Ref.']:.2f}%, strongest baseline "
                       f"{NAME.get(strongest, strongest)} {avg[strongest]:.2f}% (margin {avg[OURS] - avg[strongest]:.2f}%), "
                       f"SNRI of DenoFlow {get(c, OURS, 'snri'):.2f} dB")
    for ds in DATASETS:
        out.append(f"- Gain of DenoFlow over Art. per decoder on {DATASETS[ds][1]} (EMG / EOG / mixed):")
        for k in DECODERS:
            g = [get(cells[f"{ds}_{a}"], OURS, k) - get(cells[f"{ds}_{a}"], "Art.", k) for a in ARTS]
            rng = f" (from {min(g):.2f}% to {max(g):.2f}%)" if all(x == x for x in g) else ""
            out.append(f"  - {DECODER_NAME.get(k, k)}: " + " / ".join(f"{x:.2f}%" for x in g) + rng)
        for art in ARTS:
            c = cells[f"{ds}_{art}"]
            for k in DECODERS:
                above = [NAME.get(m, m) for m in COMPARATORS if get(c, m, k) > get(c, OURS, k)]
                if above:
                    out.append(f"  - {ART_NAME[art]}, {DECODER_NAME.get(k, k)}: above DenoFlow: {', '.join(above)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results", help="paths.outputs.root of the runs")
    ap.add_argument("--prefix", default="", help="common prefix of the run names, if any")
    ap.add_argument("--out", default="results/paper_tables")
    ap.add_argument("--allow-partial", action="store_true", help="average the folds that exist")
    args = ap.parse_args()

    cells, subjects = collect(args)
    lines = ["# DenoFlow: Tables 4 to 7",
             f"\nResults root `{args.root}`. Mean ± standard deviation over subjects of the fold-averaged values; "
             "* p < 0.05, ** p < 0.005, *** p < 0.001 (Wilcoxon signed-rank test against DenoFlow, Holm-corrected "
             "over the nine baselines). Stars on the accuracy tables are given for Avg. only, as in the paper."]
    lines += report_tables(cells)
    lines += report_numbers(cells)
    if MISSING:
        lines.append("\n## Subjects with missing folds\n")
        for (ds, label), subs in sorted(MISSING.items()):
            lines.append(f"- {DATASETS[ds][1]}, {label}: subjects {sorted(subs)}"
                         + (" (averaged over the folds present)" if args.allow_partial else " (left out)"))
    os.makedirs(args.out, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out, "tables.json"), "w", encoding="utf-8") as fh:
        json.dump(cells, fh, indent=1)
    with open(os.path.join(args.out, "subjects.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "artifact", "method", "key", "subject", "value"])
        for cell, sv in subjects.items():
            ds, art = cell.split("_")
            for m, table in sv.items():
                for k, values in table.items():
                    for s, v in sorted(values.items()):
                        w.writerow([DATASETS[ds][1], ART_NAME[art], NAME.get(m, m), k, s, f"{v:.6f}"])
    print(text)


if __name__ == "__main__":
    main()
