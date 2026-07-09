"""
Summary figure for k robustness: silhouette, imbalance, and responsive neuron counts across k.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from util import load_data

base_path = Path("/home/louis/Documents/notebooks/rave-activations/rave-activations/results_500_all_ft_balanced")
models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']
sections = ['early', 'middle', 'late']
K_RANGE  = list(range(4, 11))

# ── collect metrics ───────────────────────────────────────────────────────────
silhouettes      = []   # mean silhouette per k
pct_imbalanced   = []   # % imbalanced per k
pitch_responsive = []   # total unique pitch-responsive neurons per k
bpm_responsive   = []   # total unique bpm-responsive neurons per k
sc_responsive    = []   # total unique spectral-centroid-responsive neurons per k
sb_responsive    = []   # total unique spectral-bandwidth-responsive neurons per k

for k in K_RANGE:
    cluster_path = base_path / f"{k}_cluster"

    # silhouette
    sil_scores = []
    imb, bal = 0, 0
    for model in models:
        for dataset in datasets:
            cross_file = cluster_path / model / dataset / 'cross_layer_clustering_results_all_neurons.json'
            if not cross_file.exists():
                continue
            with open(cross_file) as f:
                cross_data = json.load(f)
            for section in sections:
                if section not in cross_data:
                    continue
                sil = cross_data[section]['silhouette_score']
                # silhouette_score may be a pre-aggregated scalar or a per-cluster
                # list/dict depending on how it was saved — take the mean either way
                if hasattr(sil, '__iter__'):
                    sil_scores.append(np.mean(list(sil.values()) if isinstance(sil, dict) else sil))
                else:
                    sil_scores.append(sil)
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
    sc_responsive.append((cluster_df['spectral_centroid_mean_corr'].mean(),
                         cluster_filtered['spectral_centroid_mean_corr'].max()))
    sb_responsive.append((cluster_df['spectral_bandwidth_mean_corr'].mean(),
                         cluster_filtered['spectral_bandwidth_mean_corr'].max()))

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
sc_means    = [v[0] for v in sc_responsive]
sc_maxes    = [v[1] for v in sc_responsive]
sb_means    = [v[0] for v in sb_responsive]
sb_maxes    = [v[1] for v in sb_responsive]

ax = axes[2]
ax.plot(K_RANGE, pitch_means, marker='^', color='#9b59b6', linewidth=2,
        markersize=7, linestyle='-',  label='Pitch mean')
ax.plot(K_RANGE, pitch_maxes, marker='^', color='#9b59b6', linewidth=2,
        markersize=7, linestyle='--', label='Pitch max (≥10 neurons)')
ax.plot(K_RANGE, bpm_means,   marker='o', color='#e07b39', linewidth=2,
        markersize=7, linestyle='-',  label='BPM mean')
ax.plot(K_RANGE, bpm_maxes,   marker='o', color='#e07b39', linewidth=2,
        markersize=7, linestyle='--', label='BPM max (≥10 neurons)')
ax.plot(K_RANGE, sc_means,    marker='s', color='#2ca25f', linewidth=2,
        markersize=7, linestyle='-',  label='Spec. centroid mean')
ax.plot(K_RANGE, sc_maxes,    marker='s', color='#2ca25f', linewidth=2,
        markersize=7, linestyle='--', label='Spec. centroid max (≥10 neurons)')
ax.plot(K_RANGE, sb_means,    marker='D', color='#3182bd', linewidth=2,
        markersize=7, linestyle='-',  label='Spec. bandwidth mean')
ax.plot(K_RANGE, sb_maxes,    marker='D', color='#3182bd', linewidth=2,
        markersize=7, linestyle='--', label='Spec. bandwidth max (≥10 neurons)')
ax.axvline(6, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='k=6 (chosen)')
ax.set_xlabel('k (number of clusters)', fontsize=11)
ax.set_ylabel('Mean correlation', fontsize=11)
ax.set_title('Feature correlation across k', fontsize=12)
ax.set_xticks(K_RANGE)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('k robustness summary', fontsize=13, y=1.02)
plt.tight_layout()
out = base_path.parent / "compare_k" / "k_robustness_summary.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
