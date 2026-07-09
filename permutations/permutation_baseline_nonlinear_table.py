"""
Prints tables summarising nonlinear permutation baseline results across all
model×dataset combos.  Reads permutation_baseline_nonlinear.json, which
stores per-layer R² statistics rather than the per-neuron |r| values in
permutation_baseline.json.

For each feature (pitch, bpm) a table shows per-combo:
  - % layers exceeding null p95 R²
  - Mean observed R² (nonlinear model)
  - Mean null p95 R²
  - Mean linear R²  (for comparison)
  - Mean nonlinear gain  (observed_r2 - linear_observed_r2)
"""

import csv
import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "6_cluster"

MODELS = ["strings", "drum_loops", "taylor_vocal", "encodec"]
DATASETS = ["strings", "drum_loops", "stimuli", "vocals"]


def load_records():
    records = []
    for model in MODELS:
        for dataset in DATASETS:
            path = RESULTS_DIR / model / dataset / "permutation_baseline_nonlinear.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for feature, layers in data.items():
                layer_vals = list(layers.values())
                if not layer_vals:
                    continue

                pct_exceeding = 100 * sum(v["exceeds_null_p95"] for v in layer_vals) / len(layer_vals)

                def mean(key):
                    vals = [v[key] for v in layer_vals if key in v]
                    return sum(vals) / len(vals) if vals else math.nan

                records.append({
                    "model":              model,
                    "dataset":            dataset,
                    "feature":            feature,
                    "n_layers":           len(layer_vals),
                    "pct_exceeding_p95":  pct_exceeding,
                    "mean_observed_r2":   mean("observed_r2"),
                    "mean_null_p95_r2":   mean("null_p95_r2"),
                    "mean_null_mean_r2":  mean("null_mean_r2"),
                    "mean_linear_r2":     mean("linear_observed_r2"),
                    "mean_nonlinear_gain": mean("nonlinear_gain"),
                })
    return records


def print_table(records, feature):
    rows = [r for r in records if r["feature"] == feature]
    if not rows:
        print(f"No data for {feature}\n")
        return

    header = (
        f"{'Model':<14} {'Dataset':<12} {'N layers':>8} "
        f"{'%>p95':>6} "
        f"{'Obs R²':>8} {'Null p95 R²':>11} {'Linear R²':>10} {'NL gain':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (per-layer R², nonlinear model)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['model']:<14} {r['dataset']:<12} {r['n_layers']:>8} "
            f"{r['pct_exceeding_p95']:>5.1f}% "
            f"{r['mean_observed_r2']:>8.4f} {r['mean_null_p95_r2']:>11.4f} "
            f"{r['mean_linear_r2']:>10.4f} {r['mean_nonlinear_gain']:>8.4f}"
        )

    def _mean_row(label, subset):
        if not subset:
            return
        def avg(key):
            return sum(r[key] for r in subset) / len(subset)
        print(
            f"{label:<14} {'':<12} {avg('n_layers'):>8.1f} "
            f"{avg('pct_exceeding_p95'):>5.1f}% "
            f"{avg('mean_observed_r2'):>8.4f} {avg('mean_null_p95_r2'):>11.4f} "
            f"{avg('mean_linear_r2'):>10.4f} {avg('mean_nonlinear_gain'):>8.4f}"
        )

    print(sep)
    _mean_row("MEAN", rows)
    if feature == "bpm":
        excl = [r for r in rows if r["dataset"] != "vocals"]
        if len(excl) < len(rows):
            _mean_row("MEAN (excl. vocals)", excl)
    print()


records = load_records()
print_table(records, "pitch")
print_table(records, "bpm")
print_table(records, "spectral_centroid")
print_table(records, "spectral_bandwidth")

csv_path = Path(__file__).parent / "permutation_baseline_nonlinear_table.csv"
if records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {csv_path}")
