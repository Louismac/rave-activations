"""
Plot average size of rank-1, rank-2, ... rank-N clusters across k values,
one subplot per section, averaged over all 12 model × dataset combos.
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
SECTIONS = ['early', 'middle', 'late']
K_RANGE  = list(range(4, 11))
N_RANKS  = 6   # how many ranked clusters to show

def load_data(model, dataset):
    data = {}
    for k in K_RANGE:
        f = base_path / f"{k}_cluster" / model / dataset / 'cross_layer_cluster_correlation.json'
        if f.exists():
            with open(f) as fh:
                data[k] = json.load(fh)
    return data

# size_by_rank[section][k][rank] = list of sizes across combos
size_by_rank = {sec: {k: defaultdict(list) for k in K_RANGE} for sec in SECTIONS}

for model in MODELS:
    for dataset in DATASETS:
        all_data = load_data(model, dataset)
        for section in SECTIONS:
            for k in K_RANGE:
                if k not in all_data:
                    continue
                sizes = sorted(
                    [len(info.get('neuron_origins', []))
                     for info in all_data[k].get(section, {}).values()],
                    reverse=True
                )
                for rank, sz in enumerate(sizes[:N_RANKS]):
                    size_by_rank[section][k][rank].append(sz)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

rank_colours = plt.cm.viridis(np.linspace(0.1, 0.9, N_RANKS))

for ax, section in zip(axes, SECTIONS):
    for rank in range(N_RANKS):
        means, sems = [], []
        for k in K_RANGE:
            vals = size_by_rank[section][k][rank]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(np.nan)
                sems.append(np.nan)

        means = np.array(means)
        sems  = np.array(sems)
        valid = ~np.isnan(means)

        label = f"rank {rank + 1}"
        ax.plot(np.array(K_RANGE)[valid], means[valid],
                color=rank_colours[rank],
                marker='o', linewidth=2, markersize=6,
                label=label)
        ax.fill_between(np.array(K_RANGE)[valid],
                         (means - sems)[valid],
                         (means + sems)[valid],
                         color=rank_colours[rank], alpha=0.15)

    ax.set_title(section, fontsize=13)
    ax.set_xlabel("k (number of clusters)", fontsize=11)
    ax.set_ylabel("Mean cluster size (neurons)", fontsize=11)
    ax.set_xticks(K_RANGE)
    ax.legend(title="cluster rank", fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Average size of ranked clusters across k\n"
             "(mean ± SEM across 12 model × dataset combos)", fontsize=13, y=1.02)
plt.tight_layout()
out = base_path.parent / "analyse_cross_layer_correlations" / "cluster_size_ranks_across_k.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
