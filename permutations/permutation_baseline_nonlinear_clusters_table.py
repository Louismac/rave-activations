"""
Prints tables summarising nonlinear-probe permutation baseline results for clusters
across all model×dataset combos.

Structure of permutation_baseline_nonlinear_clusters.json:
  {section: {cluster_id: {feature: {null_mean_r2, null_p95_r2, observed_r2, ...}}}}
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
        path = RESULTS_DIR / model / dataset / "permutation_baseline_nonlinear_clusters.json"
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

has_linear = any("linear_observed_r2" in r for r in records)


def print_table(feature):
    rows = [r for r in records if r["feature"] == feature]
    if not rows:
        print(f"No data for {feature}\n")
        return

    header = (
        f"{'Model':<14} {'Dataset':<12} {'Section':<8} {'Cl':>3} {'N_ch':>6} {'h_dim':>6} "
        f"{'Null R²':>8} {'Null p95':>9} "
        f"{'Obs R²':>8} {'±std':>6} {'ΔR²':>7} {'>p95':>5}"
    )
    if has_linear:
        header += f"  {'Lin R²':>7} {'NL gain':>8}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (nonlinear probe, hidden_dim scaled per cluster)")
    print(f"{'='*len(header)}")
    print(header)

    prev_key = None
    for r in sorted(rows, key=lambda x: (x["model"], x["dataset"], x["section"], x["cluster"])):
        key = (r["model"], r["dataset"])
        if key != prev_key:
            print(sep)
            prev_key = key
        exceeds = "Y" if r.get("exceeds_null_p95") else "n"
        line = (
            f"{r['model']:<14} {r['dataset']:<12} {r['section']:<8} {r['cluster']:>3} "
            f"{r['n_channels']:>6} {r.get('hidden_dim', '?'):>6} "
            f"{r['null_mean_r2']:>8.4f} {r['null_p95_r2']:>9.4f} "
            f"{r['observed_r2']:>8.4f} {r.get('observed_r2_std', float('nan')):>6.4f} "
            f"{r['delta_r2']:>+7.4f} {exceeds:>5}"
        )
        if has_linear:
            lin  = r.get("linear_observed_r2", float("nan"))
            gain = r.get("nonlinear_gain",     float("nan"))
            line += f"  {lin:>7.4f} {gain:>+8.4f}"
        print(line)

    print(sep)

    # Summary rows per section
    for section in SECTIONS:
        subset = [r for r in rows if r["section"] == section]
        if not subset:
            continue
        n_exceeds = sum(1 for r in subset if r.get("exceeds_null_p95"))
        line = (
            f"{'MEAN ' + section.upper():<14} {'':<12} {section:<8} {'':<3} "
            f"{sum(r['n_channels'] for r in subset)//len(subset):>6} "
            f"{'':<6} "
            f"{sum(r['null_mean_r2'] for r in subset)/len(subset):>8.4f} "
            f"{sum(r['null_p95_r2']  for r in subset)/len(subset):>9.4f} "
            f"{sum(r['observed_r2']  for r in subset)/len(subset):>8.4f} "
            f"{'':<6} "
            f"{sum(r['delta_r2'] for r in subset)/len(subset):>+7.4f} "
            f"{n_exceeds}/{len(subset):>3}"
        )
        if has_linear:
            lin_vals  = [r.get("linear_observed_r2", float("nan")) for r in subset]
            gain_vals = [r.get("nonlinear_gain",     float("nan")) for r in subset]
            lin_vals  = [v for v in lin_vals  if v == v]  # drop nan
            gain_vals = [v for v in gain_vals if v == v]
            lin_mean  = sum(lin_vals)  / len(lin_vals)  if lin_vals  else float("nan")
            gain_mean = sum(gain_vals) / len(gain_vals) if gain_vals else float("nan")
            line += f"  {lin_mean:>7.4f} {gain_mean:>+8.4f}"
        print(line)
    print()


for feature in FEATURES:
    print_table(feature)

csv_path = Path(__file__).parent / "permutation_baseline_nonlinear_clusters_table.csv"
if records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {csv_path}")
