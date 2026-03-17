"""
How stable is the full cluster structure across consecutive k values?

For each section (early/middle/late), for each k→k+1 transition:
  - Build a label vector over all neurons in the section
  - Compute Adjusted Rand Index between the two partitions
  - Also compute mean best-match Jaccard (each cluster in k matched greedily
    to its highest-overlap cluster in k+1)

Both metrics are averaged across all 12 model × dataset combos and plotted.
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

import pathlib
base_path = pathlib.Path(__file__).parent.resolve()
base_path = base_path / "results/6_cluster'"
MODELS   = ['drum_loops', 'strings', 'taylor_vocal']
DATASETS = ['drum_loops', 'stimuli', 'strings', 'vocals']
SECTIONS = ['early', 'middle', 'late']
K_RANGE  = range(4, 11)
transitions = list(zip(range(4, 10), range(5, 11)))


def load_data(model, dataset):
    data = {}
    for k in K_RANGE:
        f = base_path / f"{k}_cluster" / model / dataset / 'cross_layer_cluster_correlation.json'
        if f.exists():
            with open(f) as fh:
                data[k] = json.load(fh)
    return data


def section_labels(data, k, section):
    """Return dict: neuron_tuple -> cluster_int_label for all neurons in section."""
    labels = {}
    for idx, (ck, info) in enumerate(data[k].get(section, {}).items()):
        for n in info.get('neuron_origins', []):
            labels[tuple(n)] = idx
    return labels


def ari_between(labels_i, labels_j):
    """ARI between two label dicts, restricted to neurons present in both."""
    common = sorted(set(labels_i) & set(labels_j))
    if len(common) < 2:
        return np.nan
    a = [labels_i[n] for n in common]
    b = [labels_j[n] for n in common]
    return adjusted_rand_score(a, b)


def mean_best_match_jaccard(data, ki, kj, section):
    """For each cluster in ki, find its best-matching cluster in kj by Jaccard,
    return the size-weighted mean of those Jaccards."""
    clusters_i = {ck: frozenset(tuple(n) for n in info.get('neuron_origins', []))
                  for ck, info in data[ki].get(section, {}).items()}
    clusters_j = {ck: frozenset(tuple(n) for n in info.get('neuron_origins', []))
                  for ck, info in data[kj].get(section, {}).items()}

    if not clusters_i or not clusters_j:
        return np.nan

    weighted_sum, total_size = 0.0, 0
    for ni in clusters_i.values():
        if not ni:
            continue
        best_j = max(
            (len(ni & nj) / len(ni | nj) for nj in clusters_j.values() if ni | nj),
            default=0.0
        )
        weighted_sum += best_j * len(ni)
        total_size   += len(ni)

    return weighted_sum / total_size if total_size else np.nan


# ── collect metrics across all combos ────────────────────────────────────────
ari_vals  = defaultdict(lambda: defaultdict(list))   # [section][transition]
jaccard_vals = defaultdict(lambda: defaultdict(list))

for model in MODELS:
    for dataset in DATASETS:
        all_data = load_data(model, dataset)
        ks = sorted(all_data.keys())

        for section in SECTIONS:
            for ki, kj in transitions:
                if ki not in all_data or kj not in all_data:
                    continue

                li = section_labels(all_data, ki, section)
                lj = section_labels(all_data, kj, section)
                ari  = ari_between(li, lj)
                jacq = mean_best_match_jaccard(all_data, ki, kj, section)

                ari_vals[section][(ki, kj)].append(ari)
                jaccard_vals[section][(ki, kj)].append(jacq)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

x_labels = [f"k={ki}→{kj}" for ki, kj in transitions]
x = np.arange(len(transitions))

colours = {'early': '#e07b39', 'middle': '#9b59b6', 'late': '#2ecc71'}
styles  = {'early': ':', 'middle': '--', 'late': '-'}
markers = {'early': 'o', 'middle': 's', 'late': '^'}

for ax, (metric_vals, title, ylabel) in zip(
    axes,
    [
        (ari_vals,     "Adjusted Rand Index",           "ARI (0=random, 1=identical)"),
        (jaccard_vals, "Mean best-match Jaccard",       "Mean Jaccard (size-weighted)"),
    ]
):
    for section in SECTIONS:
        means, sems = [], []
        for trans in transitions:
            vals = [v for v in metric_vals[section][trans] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        ax.plot(x, means,
                color=colours[section],
                linestyle=styles[section],
                marker=markers[section],
                linewidth=2,
                markersize=7,
                label=section)
        ax.fill_between(x,
                         np.array(means) - np.array(sems),
                         np.array(means) + np.array(sems),
                         color=colours[section],
                         alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Consecutive k transition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.legend(title="section", fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Cluster stability across k  (mean ± SEM, 12 model × dataset combos)",
             fontsize=13, y=1.02)
plt.tight_layout()
out = base_path.parent / "analyse_cross_layer_correlations" / "clustering_stability_across_k.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
