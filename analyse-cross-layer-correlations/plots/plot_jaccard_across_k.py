"""
Plot average Jaccard (across 12 model/dataset combos) of the best-cluster
neuron overlap between consecutive k values (k→k+1).

6 lines: early/middle/late × bpm/pitch
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import pathlib
base_path = pathlib.Path(__file__).parent.resolve()
base_path = base_path / "results/6_cluster'"
MODELS   = ['drum_loops', 'strings', 'taylor_vocal']
DATASETS = ['drum_loops', 'stimuli', 'strings', 'vocals']
FEATURES = ['bpm', 'pitch']
SECTIONS = ['early', 'middle', 'late']
K_RANGE  = range(4, 11)


def load_data(model, dataset):
    data = {}
    for k in K_RANGE:
        f = base_path / f"{k}_cluster" / model / dataset / 'cross_layer_cluster_correlation.json'
        if f.exists():
            with open(f) as fh:
                data[k] = json.load(fh)
    return data


def best_neurons(data, k, section, feature):
    best_corr, best_set = -1, frozenset()
    for ck, info in data[k].get(section, {}).items():
        corr = info.get('properties', {}).get(feature, {}).get('mean_correlation', 0)
        if corr > best_corr:
            best_corr = corr
            best_set = frozenset(tuple(n) for n in info.get('neuron_origins', []))
    return best_set


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# ── collect per-transition Jaccard for each combo ────────────────────────────
# jaccard_values[(section, feature)][transition_idx] = [values across combos]
transitions = list(zip(range(4, 10), range(5, 11)))   # (4,5),(5,6),...,(9,10)
jaccard_values = defaultdict(lambda: defaultdict(list))

for model in MODELS:
    for dataset in DATASETS:
        all_data = load_data(model, dataset)
        ks = sorted(all_data.keys())
        if len(ks) < 2:
            continue

        for feature in FEATURES:
            for section in SECTIONS:
                for ki, kj in transitions:
                    if ki not in all_data or kj not in all_data:
                        continue
                    ni = best_neurons(all_data, ki, section, feature)
                    nj = best_neurons(all_data, kj, section, feature)
                    j = jaccard(ni, nj)
                    jaccard_values[(section, feature)][(ki, kj)].append(j)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x_labels = [f"k={ki}→{kj}" for ki, kj in transitions]
x = np.arange(len(transitions))

colours = {'bpm': '#e07b39', 'pitch': '#4a90d9'}
styles  = {'early': ':', 'middle': '--', 'late': '-'}
markers = {'early': 'o', 'middle': 's', 'late': '^'}

for feature in FEATURES:
    for section in SECTIONS:
        means, sems = [], []
        for trans in transitions:
            vals = jaccard_values[(section, feature)][trans]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))

        label = f"{section} / {feature}"
        ax.plot(x, means,
                color=colours[feature],
                linestyle=styles[section],
                marker=markers[section],
                linewidth=1.8,
                markersize=6,
                label=label)
        ax.fill_between(x,
                         np.array(means) - np.array(sems),
                         np.array(means) + np.array(sems),
                         color=colours[feature],
                         alpha=0.10)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel("Consecutive k transition", fontsize=11)
ax.set_ylabel("Mean Jaccard (best cluster overlap)", fontsize=11)
ax.set_title("Neuron overlap of best cluster between consecutive k values\n"
             "(mean ± SEM across 12 model × dataset combos)", fontsize=12)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
ax.legend(title="section / feature", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out = base_path.parent / "analyse_cross_layer_correlations" / "jaccard_across_k.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
