"""
Plots % neurons exceeding permutation null p95 threshold per layer for each feature (pitch, bpm).
One figure per feature; each line is a model×dataset combo.
Data sources: variance_correlation.json (per-neuron |r|) + permutation_baseline.json (null_p95_r).
Saves: plot_pct_exceeding_per_layer_pitch.png and plot_pct_exceeding_per_layer_bpm.png
"""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results" / "6_cluster"

MODELS = ["strings", "drum_loops", "taylor_vocal", "encodec"]
DATASETS = ["strings", "drum_loops", "stimuli", "vocals"]

OUT_DIR = Path(__file__).parent

PROPS  = ("pitch", "bpm","spectral_centroid","spectral_bandwidth")


def load_per_layer():
    """Return {(model, dataset, feature): {layer_key: pct_exceeding}}"""
    data = {}
    for model in MODELS:
        for dataset in DATASETS:
            var_path  = RESULTS_DIR / model / dataset / "variance_correlation.json"
            perm_path = RESULTS_DIR / model / dataset / "permutation_baseline.json"
            if not var_path.exists() or not perm_path.exists():
                continue

            with open(var_path) as f:
                var = json.load(f)
            with open(perm_path) as f:
                perm = json.load(f)

            null_p95 = {
                prop: perm[prop]["null_p95_r"]
                for prop in PROPS
                if prop in perm and "null_p95_r" in perm[prop]
            }

            for feature in PROPS:
                if feature not in null_p95:
                    continue
                threshold = null_p95[feature]
                layer_pct = {}
                for layer_name, layer_data in var.items():
                    if layer_name.startswith("section_"):
                        continue
                    if feature not in layer_data:
                        continue
                    corrs = np.array(layer_data[feature].get("all_correlations", []))
                    if len(corrs) == 0:
                        continue
                    layer_pct[layer_name] = float((corrs > threshold).mean() * 100)
                if layer_pct:
                    data[(model, dataset, feature)] = layer_pct
    return data


def short_layer_label(key):
    parts = key.split(".")
    nums = [p for p in parts if p.isdigit()]
    if len(nums) == 1:
        return nums[0]
    elif len(nums) >= 2:
        return f"{nums[0]}.b{nums[1]}.{nums[-1]}"
    return key


def plot_feature(feature, all_data):
    combos = {(m, d): v for (m, d, f), v in all_data.items() if f == feature}
    if not combos:
        print(f"No data for {feature}")
        return

    layer_keys = list(next(iter(combos.values())).keys())
    x = np.arange(len(layer_keys))
    xlabels = [short_layer_label(k) for k in layer_keys]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = cm.tab10(np.linspace(0, 1, len(combos)))
    for (model, dataset), layer_pct, color in zip(combos.keys(), combos.values(), colors):
        y = [layer_pct.get(k, float("nan")) for k in layer_keys]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2,
                label=f"{model} / {dataset}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("% neurons exceeding p95")
    ax.set_title(f"% neurons exceeding p95 per layer — {feature.upper()}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "plots" / f"plot_pct_exceeding_per_layer_{feature}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def print_table(feature, all_data):
    combos = {(m, d): v for (m, d, f), v in all_data.items() if f == feature}
    if not combos:
        print(f"No data for {feature}\n")
        return []

    null_p95_lookup = {}
    for model in MODELS:
        for dataset in DATASETS:
            p = RESULTS_DIR / model / dataset / "permutation_baseline.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            if feature in d:
                null_p95_lookup[(model, dataset)] = d[feature].get("null_p95_r", float("nan"))

    header = (
        f"{'Model':<14} {'Dataset':<12} {'Layer':<14} "
        f"{'Null p95':>9} {'Obs %>p95':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (% neurons > global null p95)")
    print(f"{'='*len(header)}")
    print(header)

    csv_records = []
    for (model, dataset), layer_pct in sorted(combos.items()):
        print(sep)
        null_p95 = null_p95_lookup.get((model, dataset), float("nan"))
        pcts = []
        for layer_name, pct in layer_pct.items():
            pcts.append(pct)
            csv_records.append({
                "model": model, "dataset": dataset, "feature": feature,
                "layer": layer_name, "layer_short": short_layer_label(layer_name),
                "null_p95_r": null_p95, "obs_pct_exceeding": pct,
            })
            print(
                f"{model:<14} {dataset:<12} {short_layer_label(layer_name):<14} "
                f"{null_p95:>9.4f} {pct:>9.1f}%"
            )
        if pcts:
            print(
                f"  {'↳ mean':<12} {'':<12} {'':<14} "
                f"{null_p95:>9.4f} {sum(pcts)/len(pcts):>9.1f}%"
            )

    print(sep)
    print()
    return csv_records


all_data = load_per_layer()
plot_feature("pitch", all_data)
plot_feature("bpm", all_data)
plot_feature("spectral_centroid", all_data)
plot_feature("spectral_bandwidth", all_data)

all_csv_records = []
for _feat in PROPS:
    all_csv_records += print_table(_feat, all_data)

csv_path = OUT_DIR / "plot_pct_exceeding_per_layer.csv"
if all_csv_records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_csv_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_csv_records)
    print(f"Saved {csv_path}")
