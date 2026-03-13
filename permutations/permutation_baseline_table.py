"""
Prints tables summarising permutation baseline results across all model×dataset combos.
For each feature (pitch, bpm):
  - Null: mean|r|, p95, % exceeding threshold (across permutations)
  - Observed: mean|r|, max|r|, % exceeding threshold
  - Ratio: observed mean|r| / null mean|r|
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "6_cluster"
THRESHOLD = 0.15

MODELS = ["strings", "drum_loops", "taylor_vocal"]
DATASETS = ["strings", "drum_loops", "stimuli", "vocals"]

records = []
for model in MODELS:
    for dataset in DATASETS:
        path = RESULTS_DIR / model / dataset / "permutation_baseline.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for feature, vals in data.items():
            records.append({
                "model": model,
                "dataset": dataset,
                "feature": feature,
                **vals,
            })


def print_table(feature):
    rows = [r for r in records if r["feature"] == feature]
    if not rows:
        print(f"No data for {feature}\n")
        return

    header = (
        f"{'Model':<14} {'Dataset':<12} "
        f"{'Null mean|r|':>12} {'Null p95':>9} "
        f"{'Obs mean|r|':>12} {'Obs max|r|':>10} {'Obs %>p95':>10} "
        f"{'Ratio':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (null p95 threshold, per-neuron)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in rows:
        obs_pct = r.get("observed_pct_exceeding_p95", float("nan"))
        ratio = r["observed_mean_r"] / r["null_mean_r"] if r["null_mean_r"] > 0 else float("inf")
        print(
            f"{r['model']:<14} {r['dataset']:<12} "
            f"{r['null_mean_r']:>12.4f} {r['null_p95_r']:>9.4f} "
            f"{r['observed_mean_r']:>12.4f} {r['observed_max_r']:>10.4f} {obs_pct:>9.1f}% "
            f"{ratio:>6.1f}x"
        )

    def _mean_row(label, subset):
        if not subset:
            return
        s_pcts   = [r.get("observed_pct_exceeding_p95", float("nan")) for r in subset]
        s_means  = [r["observed_mean_r"] for r in subset]
        s_maxes  = [r["observed_max_r"]  for r in subset]
        s_ratios = [r["observed_mean_r"] / r["null_mean_r"] if r["null_mean_r"] > 0 else float("inf") for r in subset]
        print(
            f"{label:<14} {'':<12} "
            f"{sum(r['null_mean_r'] for r in subset)/len(subset):>12.4f} "
            f"{sum(r['null_p95_r'] for r in subset)/len(subset):>9.4f} "
            f"{sum(s_means)/len(s_means):>12.4f} "
            f"{sum(s_maxes)/len(s_maxes):>10.4f} "
            f"{sum(s_pcts)/len(s_pcts):>9.1f}% "
            f"{sum(s_ratios)/len(s_ratios):>6.1f}x"
        )

    print(sep)
    _mean_row("MEAN", rows)
    if feature == "bpm":
        excl = [r for r in rows if r["dataset"] != "vocals"]
        if len(excl) < len(rows):
            _mean_row("MEAN (excl. vocals)", excl)
    print()


print_table("pitch")
print_table("bpm")
