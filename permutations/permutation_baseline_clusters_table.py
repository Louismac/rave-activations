"""
Prints tables summarising Spearman permutation baseline results for clusters
across all model×dataset combos.

Structure of permutation_baseline_clusters.json:
  {section: {cluster_id: {feature: {null_mean_r, null_p95_r, observed_mean_r, ...}}}}

When use_global_null=True the null columns are the same across all clusters within
a model×dataset×feature combo (borrowed from permutation_baseline.json).
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "6_cluster"
SECTIONS    = ["early", "middle", "late"]
MODELS      = ["strings", "drum_loops", "taylor_vocal","encodec"]
DATASETS    = ["strings", "drum_loops", "stimuli", "vocals"]
FEATURES    = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]

# ── load all records ──────────────────────────────────────────────────────────
records = []
for model in MODELS:
    for dataset in DATASETS:
        path = RESULTS_DIR / model / dataset / "permutation_baseline_clusters.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for section in SECTIONS:
            if section not in data:
                continue
            for cluster_key, cluster_data in data[section].items():
                cluster_id = int(cluster_key.split("_")[1])
                for feature, vals in cluster_data.items():
                    records.append({
                        "model":     model,
                        "dataset":   dataset,
                        "section":   section,
                        "cluster":   cluster_id,
                        "feature":   feature,
                        **vals,
                    })


def print_table(feature):
    rows = [r for r in records if r["feature"] == feature]
    if not rows:
        print(f"No data for {feature}\n")
        return

    use_global = any(r.get("use_global_null") for r in rows)
    null_label = "Global null" if use_global else "Per-cluster null"

    header = (
        f"{'Model':<14} {'Dataset':<12} {'Section':<8} {'Cl':>3} {'N_neur':>7} "
        f"{'Null mean|r|':>12} {'Null p95':>9} "
        f"{'Obs mean|r|':>12} {'Obs max|r|':>11} {'Obs %>p95':>10}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  ({null_label})")
    print(f"{'='*len(header)}")
    print(header)

    prev_key = None
    for r in sorted(rows, key=lambda x: (x["model"], x["dataset"], x["section"], x["cluster"])):
        key = (r["model"], r["dataset"])
        if key != prev_key:
            print(sep)
            prev_key = key
        obs_pct = r.get("observed_pct_exceeding_p95", float("nan"))
        print(
            f"{r['model']:<14} {r['dataset']:<12} {r['section']:<8} {r['cluster']:>3} "
            f"{r['n_neurons']:>7} "
            f"{r['null_mean_r']:>12.4f} {r['null_p95_r']:>9.4f} "
            f"{r['observed_mean_r']:>12.4f} {r['observed_max_r']:>11.4f} {obs_pct:>9.1f}%"
        )

    print(sep)

    # Summary row per section across all model×dataset combos
    for section in SECTIONS:
        subset = [r for r in rows if r["section"] == section]
        if not subset:
            continue
        pcts   = [r.get("observed_pct_exceeding_p95", float("nan")) for r in subset]
        means  = [r["observed_mean_r"] for r in subset]
        maxes  = [r["observed_max_r"]  for r in subset]
        print(
            f"{'MEAN ' + section.upper():<14} {'':<12} {section:<8} {'':<3} "
            f"{sum(r['n_neurons'] for r in subset)//len(subset):>7} "
            f"{sum(r['null_mean_r'] for r in subset)/len(subset):>12.4f} "
            f"{sum(r['null_p95_r']  for r in subset)/len(subset):>9.4f} "
            f"{sum(means)/len(means):>12.4f} "
            f"{sum(maxes)/len(maxes):>11.4f} "
            f"{sum(pcts)/len(pcts):>9.1f}%"
        )
    print()


for feature in FEATURES:
    print_table(feature)

csv_path = Path(__file__).parent / "permutation_baseline_clusters_table.csv"
if records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {csv_path}")
