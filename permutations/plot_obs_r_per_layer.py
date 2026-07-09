"""
Plots mean observed |r| per layer for each feature (pitch, bpm).
One figure per feature; each line is a model×dataset combo.
Data source: variance_correlation.json (mean_correlation per layer).
Saves: plot_obs_r_per_layer_pitch.png and plot_obs_r_per_layer_bpm.png
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
    """Return {(model, dataset, feature): {layer_key: mean_r}}"""
    data = {}
    for model in MODELS:
        for dataset in DATASETS:
            var_path = RESULTS_DIR / model / dataset / "variance_correlation.json"
            if not var_path.exists():
                continue
            with open(var_path) as f:
                var = json.load(f)
            for feature in PROPS:
                layer_r = {}
                for layer_name, layer_data in var.items():
                    if layer_name.startswith("section_"):
                        continue
                    if feature not in layer_data:
                        continue
                    mean_r = layer_data[feature].get("mean_correlation")
                    if mean_r is not None:
                        layer_r[layer_name] = mean_r
                if layer_r:
                    data[(model, dataset, feature)] = layer_r
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
    print(f"feature {feature}")
    combos = {(m, d): v for (m, d, f), v in all_data.items() if f == feature}
    if not combos:
        print(f"No data for {feature}")
        return

    layer_keys = list(next(iter(combos.values())).keys())
    x = np.arange(len(layer_keys))
    xlabels = [short_layer_label(k) for k in layer_keys]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = cm.tab10(np.linspace(0, 1, len(combos)))
    for (model, dataset), layer_r, color in zip(combos.keys(), combos.values(), colors):
        print(model, dataset)
        y = [layer_r.get(k, float("nan")) for k in layer_keys]
        print(f"mean r", np.mean(y))
        # print(y)
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2,
                label=f"{model} / {dataset}", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean observed |r|")
    ax.set_title(f"Mean observed |r| per layer — {feature.upper()}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "plots" / f"plot_obs_r_per_layer_{feature}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def print_table(feature, all_data):
    combos = {(m, d): v for (m, d, f), v in all_data.items() if f == feature}
    if not combos:
        print(f"No data for {feature}\n")
        return []

    null_stats = {}
    for model in MODELS:
        for dataset in DATASETS:
            p = RESULTS_DIR / model / dataset / "permutation_baseline.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            if feature in d:
                null_stats[(model, dataset)] = d[feature]

    header = (
        f"{'Model':<14} {'Dataset':<12} {'Layer':<14} "
        f"{'Null mean|r|':>12} {'Null p95':>9} {'Obs mean|r|':>12} {'Ratio':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (obs mean|r| from variance_correlation.json)")
    print(f"{'='*len(header)}")
    print(header)

    csv_records = []
    for (model, dataset), layer_r in sorted(combos.items()):
        print(sep)
        ns = null_stats.get((model, dataset), {})
        null_mean = ns.get("null_mean_r", float("nan"))
        null_p95  = ns.get("null_p95_r",  float("nan"))
        layer_means = []
        for layer_name, mean_r in layer_r.items():
            ratio = mean_r / null_mean if null_mean > 0 else float("nan")
            layer_means.append(mean_r)
            csv_records.append({
                "model": model, "dataset": dataset, "feature": feature,
                "layer": layer_name, "layer_short": short_layer_label(layer_name),
                "null_mean_r": null_mean, "null_p95_r": null_p95,
                "obs_mean_r": mean_r, "ratio": ratio,
            })
            print(
                f"{model:<14} {dataset:<12} {short_layer_label(layer_name):<14} "
                f"{null_mean:>12.4f} {null_p95:>9.4f} {mean_r:>12.4f} {ratio:>6.2f}x"
            )
        if layer_means:
            mean_obs = sum(layer_means) / len(layer_means)
            ratio = mean_obs / null_mean if null_mean > 0 else float("nan")
            print(
                f"  {'↳ mean':<12} {'':<12} {'':<14} "
                f"{null_mean:>12.4f} {null_p95:>9.4f} {mean_obs:>12.4f} {ratio:>6.2f}x"
            )

    print(sep)
    print()
    return csv_records


all_data = load_per_layer()
# plot_feature("pitch_class", all_data)
plot_feature("pitch", all_data)
plot_feature("bpm", all_data)
plot_feature("spectral_centroid", all_data)
plot_feature("spectral_flatness", all_data)

all_csv_records = []
for _feat in PROPS:
    all_csv_records += print_table(_feat, all_data)

csv_path = OUT_DIR / "plot_obs_r_per_layer.csv"
if all_csv_records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_csv_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_csv_records)
    print(f"Saved {csv_path}")
