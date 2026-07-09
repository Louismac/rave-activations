"""
Plots observed R² per layer for each feature (pitch, bpm).
One figure per feature; each line is a model×dataset combo.
Saves: plot_obs_r2_per_layer_pitch.png and plot_obs_r2_per_layer_bpm.png
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
    """Return {(model, dataset, feature): {layer_key: observed_r2}}"""
    data = {}
    for model in MODELS:
        for dataset in DATASETS:
            path = RESULTS_DIR / model / dataset / "permutation_baseline_nonlinear.json"
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
            for feature, layers in raw.items():
                data[(model, dataset, feature)] = {
                    layer: v["observed_r2"] for layer, v in layers.items()
                }
    return data


def short_layer_label(key):
    """Collapse long module paths to something readable on the x-axis."""
    # e.g. 'net.3.aligned.branches.0.net.1' -> '3.b0.1'
    parts = key.split(".")
    # first number after 'net' is the block index
    nums = [p for p in parts if p.isdigit()]
    if len(nums) == 1:
        return nums[0]
    elif len(nums) >= 2:
        return f"{nums[0]}.b{nums[1]}.{nums[-1]}"
    return key


def plot_feature(feature, all_data):
    print(feature)
    combos = {(m, d): v for (m, d, f), v in all_data.items() if f == feature}
    if not combos:
        print(f"No data for {feature}")
        return

    # Use layer order from the first available combo
    layer_keys = list(next(iter(combos.values())).keys())
    x = np.arange(len(layer_keys))
    xlabels = [short_layer_label(k) for k in layer_keys]

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = cm.tab10(np.linspace(0, 1, len(combos)))
    for (model, dataset), layer_r2, color in zip(combos.keys(), combos.values(), colors):
        print(model, dataset)
        y = [layer_r2.get(k, float("nan")) for k in layer_keys]
        print(y)
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2,
                label=f"{model} / {dataset}", color=color)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Observed R²")
    ax.set_title(f"Observed R² per layer — {feature.upper()} (nonlinear model)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()

    out = OUT_DIR / "plots" / f"plot_obs_r2_per_layer_{feature}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def print_table(feature):
    has_linear = False
    for model in MODELS:
        for dataset in DATASETS:
            p = RESULTS_DIR / model / dataset / "permutation_baseline_nonlinear.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                if feature in d:
                    has_linear = "linear_observed_r2" in next(iter(d[feature].values()), {})
                break
        if has_linear:
            break

    header = (
        f"{'Model':<14} {'Dataset':<12} {'Layer':<14} "
        f"{'N_ch':>6} {'h_dim':>6} "
        f"{'Null R²':>8} {'Null p95':>9} "
        f"{'Obs R²':>8} {'±std':>6} {'ΔR²':>7} {'>p95':>5}"
    )
    if has_linear:
        header += f"  {'Lin R²':>7} {'NL gain':>8}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  Feature: {feature.upper()}  (nonlinear probe per layer)")
    print(f"{'='*len(header)}")
    print(header)

    csv_records = []
    for model in MODELS:
        for dataset in DATASETS:
            path = RESULTS_DIR / model / dataset / "permutation_baseline_nonlinear.json"
            if not path.exists():
                continue
            with open(path) as f:
                raw = json.load(f)
            if feature not in raw:
                continue

            print(sep)
            obs_r2s = []
            for layer_name, v in raw[feature].items():
                exceeds = "Y" if v.get("exceeds_null_p95") else "n"
                obs_r2s.append(v["observed_r2"])
                rec = {
                    "model": model, "dataset": dataset, "feature": feature,
                    "layer": layer_name, "layer_short": short_layer_label(layer_name),
                    "n_channels": v.get("n_channels"), "hidden_dim": v.get("hidden_dim"),
                    "null_mean_r2": v["null_mean_r2"], "null_p95_r2": v["null_p95_r2"],
                    "observed_r2": v["observed_r2"],
                    "observed_r2_std": v.get("observed_r2_std"),
                    "delta_r2": v["delta_r2"],
                    "exceeds_null_p95": v.get("exceeds_null_p95"),
                    "linear_observed_r2": v.get("linear_observed_r2"),
                    "nonlinear_gain": v.get("nonlinear_gain"),
                }
                csv_records.append(rec)
                line = (
                    f"{model:<14} {dataset:<12} {short_layer_label(layer_name):<14} "
                    f"{v.get('n_channels', 0):>6} {v.get('hidden_dim', 0):>6} "
                    f"{v['null_mean_r2']:>8.4f} {v['null_p95_r2']:>9.4f} "
                    f"{v['observed_r2']:>8.4f} {v.get('observed_r2_std', float('nan')):>6.4f} "
                    f"{v['delta_r2']:>+7.4f} {exceeds:>5}"
                )
                if has_linear:
                    line += f"  {v.get('linear_observed_r2', float('nan')):>7.4f} {v.get('nonlinear_gain', float('nan')):>+8.4f}"
                print(line)

            if obs_r2s:
                print(f"  {'↳ mean':<12} {'':<12} {'':<14} "
                      f"{'':>6} {'':>6} {'':>8} {'':>9} {sum(obs_r2s)/len(obs_r2s):>8.4f}")

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
    all_csv_records += print_table(_feat)

csv_path = OUT_DIR / "plot_obs_r2_per_layer.csv"
if all_csv_records:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_csv_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_csv_records)
    print(f"Saved {csv_path}")
