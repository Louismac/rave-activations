#!/usr/bin/env python3
"""
top_clusters_p95.py — Top Performing Clusters ranked by % neurons above null p95
Replaces mean_corr ranking with pct_above_p95 (combo-specific null_p95_r threshold).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

MIN_NEURONS = 5
N_CLUSTERS  = 6
base_path   = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results') / f'{N_CLUSTERS}_cluster'

models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']
sections = ['early', 'middle', 'late']

# ── load null_p95_r per combo ─────────────────────────────────────────────────

null_p95 = {}
for model in models:
    for dataset in datasets:
        perm_path = base_path / model / dataset / 'permutation_baseline.json'
        if not perm_path.exists():
            continue
        with open(perm_path) as f:
            perm = json.load(f)
        for prop in ['pitch', 'bpm']:
            if prop in perm and 'null_p95_r' in perm[prop]:
                null_p95[(model, dataset, prop)] = perm[prop]['null_p95_r']

# ── load cluster-level % above null p95 ──────────────────────────────────────

rows = []
for model in models:
    for dataset in datasets:
        corr_file = base_path / model / dataset / 'cross_layer_cluster_correlation.json'
        if not corr_file.exists():
            continue
        with open(corr_file) as f:
            data = json.load(f)

        p_thr = null_p95.get((model, dataset, 'pitch'))
        b_thr = null_p95.get((model, dataset, 'bpm'))

        for section in sections:
            if section not in data:
                continue
            for cluster_key, cluster_info in data[section].items():
                props    = cluster_info.get('properties', {})
                n_neurons = len(cluster_info.get('neuron_origins', []))

                def pct_above(prop, thr):
                    corrs = np.array(props.get(prop, {}).get('all_correlations', []))
                    if len(corrs) == 0 or thr is None:
                        return np.nan
                    return float((corrs > thr).mean() * 100)

                rows.append({
                    'model':          model,
                    'dataset':        dataset,
                    'section':        section,
                    'cluster':        cluster_key,
                    'n_neurons':      n_neurons,
                    'pitch_pct_p95':  pct_above('pitch', p_thr),
                    'bpm_pct_p95':    pct_above('bpm',   b_thr),
                })

cluster_df = pd.DataFrame(rows)
cluster_df_filtered = cluster_df[cluster_df['n_neurons'] >= MIN_NEURONS]

# ── top clusters ─────────────────────────────────────────────────────────────

n = 10
top_pitch = (cluster_df_filtered[cluster_df_filtered['dataset'] != 'drum_loops']
             .nlargest(n, 'pitch_pct_p95').copy())
top_bpm   = (cluster_df_filtered[cluster_df_filtered['dataset'] != 'vocals']
             .nlargest(n, 'bpm_pct_p95').copy())

print(f"--- TOP {n} CLUSTERS BY % NEURONS ABOVE NULL P95 ---")
print(f"\nPITCH (excl. drum_loops, min {MIN_NEURONS} neurons):")
print(top_pitch[['model','dataset','section','cluster','n_neurons','pitch_pct_p95']].to_string(index=False))

print(f"\nBPM (excl. vocals, min {MIN_NEURONS} neurons):")
print(top_bpm[['model','dataset','section','cluster','n_neurons','bpm_pct_p95']].to_string(index=False))

print("\n--- SECTION COUNTS IN TOP 10 ---")
print(f"\n{'Section':<10s} {'Pitch top10':>12s} {'BPM top10':>10s}")
print("-"*35)
for section in sections:
    print(f"{section:<10s} {top_pitch['section'].value_counts().get(section, 0):>12d} "
          f"{top_bpm['section'].value_counts().get(section, 0):>10d}")

# ── plot ──────────────────────────────────────────────────────────────────────

section_to_pos = {'early': 0, 'middle': 1, 'late': 2}
model_to_pos   = {'strings': 0, 'drum_loops': 1, 'taylor_vocal': 2}
dataset_colors = {
    'strings':    '#1f77b4',
    'vocals':     '#ff7f0e',
    'stimuli':    '#2ca02c',
    'drum_loops': '#d62728',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Top {n} Cross-Layer Clusters: % Neurons Above Null P95',
             fontsize=16, fontweight='bold')

for ax, top_df, pct_col, title in [
    (axes[0], top_pitch, 'pitch_pct_p95', 'Pitch Hz'),
    (axes[1], top_bpm,   'bpm_pct_p95',   'BPM'),
]:
    top_df = top_df.copy()
    top_df['x_pos'] = top_df['section'].map(section_to_pos)
    top_df['y_pos'] = top_df['model'].map(model_to_pos)

    for _, row in top_df.iterrows():
        size  = (row[pct_col] / 100) ** 2 * 3000
        color = dataset_colors[row['dataset']]
        jitter_x = np.random.uniform(-0.15, 0.15)
        jitter_y = np.random.uniform(-0.15, 0.15)
        ax.scatter(row['x_pos'] + jitter_x, row['y_pos'] + jitter_y,
                   s=size, c=color, alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Network Depth', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Early\n(Layers 0-7)', 'Middle\n(Layers 8-14)', 'Late\n(Layers 15-21)'], fontsize=11)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Strings', 'Drum Loops', 'Vocal'], fontsize=11)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.2, linestyle='--')

    dataset_legend_elements = [
        Patch(facecolor=dataset_colors[ds], edgecolor='black', alpha=0.6,
              label=ds.replace('_', ' ').title())
        for ds in ['strings', 'vocals', 'stimuli', 'drum_loops']
    ]
    dataset_legend = ax.legend(handles=dataset_legend_elements, loc='upper left',
                               fontsize=10, title='Dataset', framealpha=0.9)
    ax.add_artist(dataset_legend)

    size_legend_pcts = [50, 80]
    size_legend_elements = [
        ax.scatter([], [], s=(p/100)**2*3000, c='gray', alpha=0.6,
                   edgecolors='black', linewidth=0.5, label=f'{p}% above p95')
        for p in size_legend_pcts
    ]
    ax.legend(handles=size_legend_elements, loc='lower left', fontsize=9,
              title='% above null p95', framealpha=0.9, scatterpoints=1)

plt.tight_layout()
output_path = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/plots') / 'top_clusters_p95_spatial.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")
plt.close()
