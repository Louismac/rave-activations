"""
Summary figure for k robustness: silhouette, imbalance, and responsive neuron counts across k.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from util import load_data

base_path = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results")
models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']
sections = ['early', 'middle', 'late']
K_RANGE  = list(range(4, 11))

# ── collect metrics ───────────────────────────────────────────────────────────
silhouettes      = []   # mean silhouette per k
pct_imbalanced   = []   # % imbalanced per k
pitch_responsive = []   # total unique pitch-responsive neurons per k
bpm_responsive   = []   # total unique bpm-responsive neurons per k

for k in K_RANGE:
    cluster_path = base_path / f"{k}_cluster"

    # silhouette
    sil_scores = []
    imb, bal = 0, 0
    for model in models:
        for dataset in datasets:
            cross_file = cluster_path / model / dataset / 'cross_layer_clustering_results.json'
            if not cross_file.exists():
                continue
            with open(cross_file) as f:
                cross_data = json.load(f)
            for section in sections:
                if section not in cross_data:
                    continue
                sil_scores.append(cross_data[section]['silhouette_score'])
                labels = cross_data[section]['cluster_labels']
                _, counts = np.unique(labels, return_counts=True)
                max_pct = counts.max() / len(labels)
                min_pct = counts.min() / len(labels)
                if max_pct > 0.6 or min_pct < 0.05:
                    imb += 1
                else:
                    bal += 1

    silhouettes.append(np.mean(sil_scores))
    total = imb + bal
    pct_imbalanced.append(100 * imb / total if total else 0)

    # mean and max correlations (max filtered to >= MIN_NEURONS)
    MIN_NEURONS = 10
    _, cluster_df = load_data(k)
    pitched = cluster_df[cluster_df['dataset'] != 'drum_loops']
    pitched_filtered = pitched[pitched['n_neurons'] >= MIN_NEURONS]
    cluster_filtered = cluster_df[cluster_df['n_neurons'] >= MIN_NEURONS]
    pitch_responsive.append((pitched['pitch_mean_corr'].mean(),
                             pitched_filtered['pitch_mean_corr'].max()))
    bpm_responsive.append((cluster_df['bpm_mean_corr'].mean(),
                           cluster_filtered['bpm_mean_corr'].max()))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

colour_main   = '#2c7bb6'
colour_second = '#d7191c'

# --- panel 1: silhouette ---
ax = axes[0]
ax.plot(K_RANGE, silhouettes, marker='o', color=colour_main, linewidth=2, markersize=7)
ax.axvline(6, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='k=6 (chosen)')
ax.set_xlabel('k (number of clusters)', fontsize=11)
ax.set_ylabel('Mean silhouette score', fontsize=11)
ax.set_title('Clustering quality', fontsize=12)
ax.set_xticks(K_RANGE)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# --- panel 2: imbalance ---
ax = axes[1]
ax.plot(K_RANGE, pct_imbalanced, marker='s', color=colour_second, linewidth=2, markersize=7)
ax.axvline(6, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='k=6 (chosen)')
ax.set_xlabel('k (number of clusters)', fontsize=11)
ax.set_ylabel('Imbalanced partitions (%)', fontsize=11)
ax.set_title('Cluster size imbalance', fontsize=12)
ax.set_xticks(K_RANGE)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

# --- panel 3: mean and max correlations ---
pitch_means = [v[0] for v in pitch_responsive]
pitch_maxes = [v[1] for v in pitch_responsive]
bpm_means   = [v[0] for v in bpm_responsive]
bpm_maxes   = [v[1] for v in bpm_responsive]

ax = axes[2]
ax.plot(K_RANGE, pitch_means, marker='^', color='#9b59b6', linewidth=2,
        markersize=7, linestyle='-',  label='Pitch mean')
ax.plot(K_RANGE, pitch_maxes, marker='^', color='#9b59b6', linewidth=2,
        markersize=7, linestyle='--', label='Pitch max (≥10 neurons)')
ax.plot(K_RANGE, bpm_means,   marker='o', color='#e07b39', linewidth=2,
        markersize=7, linestyle='-',  label='BPM mean')
ax.plot(K_RANGE, bpm_maxes,   marker='o', color='#e07b39', linewidth=2,
        markersize=7, linestyle='--', label='BPM max (≥10 neurons)')
ax.axvline(6, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='k=6 (chosen)')
ax.set_xlabel('k (number of clusters)', fontsize=11)
ax.set_ylabel('Mean correlation', fontsize=11)
ax.set_title('Feature correlation across k', fontsize=12)
ax.set_xticks(K_RANGE)
ax.set_ylim(0, 0.65)
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('k robustness summary', fontsize=13, y=1.02)
plt.tight_layout()
out = base_path.parent / "analyse_cross_layer_correlations" / "k_robustness_summary.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
