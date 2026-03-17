# ============================================================================
# 5. Top Performing Clusters
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from util import load_data
sections = ['early', 'middle', 'late']
MIN_NEURONS = 5  # ignore clusters smaller than this

#iterate through values of k
k = range(4,11)
k = [6]
for i in k:
    layer_df, cluster_df = load_data(i)

    # Apply minimum neuron threshold
    cluster_df_filtered = cluster_df[cluster_df['n_neurons'] >= MIN_NEURONS]

    # Top clusters: exclude drum_loops for pitch, vocals for BPM
    top_pitch = cluster_df_filtered[cluster_df_filtered['dataset'] != 'drum_loops'].nlargest(10, 'pitch_mean_corr').copy()
    top_bpm   = cluster_df_filtered[cluster_df_filtered['dataset'] != 'vocals'].nlargest(10, 'bpm_mean_corr').copy()

    print("\n--- SECTION COUNTS IN TOP 10 (k={}) ---".format(i))
    pitch_section_counts = top_pitch['section'].value_counts()
    bpm_section_counts = top_bpm['section'].value_counts()
    print(f"\n{'Section':<10s} {'Pitch top10':>12s} {'BPM top10':>10s}")
    print("-"*35)
    for section in sections:
        pitch_count = pitch_section_counts.get(section, 0)
        bpm_count = bpm_section_counts.get(section, 0)
        print(f"{section:<10s} {pitch_count:>12d} {bpm_count:>10d}")

# top_pitch / top_bpm now hold results for the last k in the loop
# Extend to top 20 for plotting using the same filtered df and exclusions
n = 10
topn_pitch = cluster_df_filtered[cluster_df_filtered['dataset'] != 'drum_loops'].nlargest(n, 'pitch_mean_corr').copy()
topn_bpm   = cluster_df_filtered[cluster_df_filtered['dataset'] != 'vocals'].nlargest(n, 'bpm_mean_corr').copy()

# ============================================================================
# PLOT TOP  CLUSTERS: PITCH AND BPM
# ============================================================================

print("\n" + "="*80)
print("PLOTTING TOP  CLUSTERS")
print("="*80)

# Map section to x-axis position
section_to_pos = {'early': 0, 'middle': 1, 'late': 2}
topn_pitch['x_pos'] = topn_pitch['section'].map(section_to_pos)
topn_bpm['x_pos'] = topn_bpm['section'].map(section_to_pos)

# Map model to y-axis position
model_to_pos = {'strings': 0, 'drum_loops': 1, 'taylor_vocal': 2}
topn_pitch['y_pos'] = topn_pitch['model'].map(model_to_pos)
topn_bpm['y_pos'] = topn_bpm['model'].map(model_to_pos)

# Dataset colors
dataset_colors = {
    'strings': '#1f77b4',      # blue
    'vocals': '#ff7f0e',       # orange
    'stimuli': '#2ca02c',      # green
    'drum_loops': '#d62728'    # red
}

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Top {n} Cross-Layer Clusters: Spatial Distribution', fontsize=16, fontweight='bold')

# Plot Pitch
ax = axes[0]
for _, row in topn_pitch.iterrows():
    # Size proportional to correlation (scale for visibility)
    size = (row['pitch_mean_corr'] ** 2) * 3000
    color = dataset_colors[row['dataset']]

    # Add jitter to avoid overlapping points at same position
    jitter_x = np.random.uniform(-0.15, 0.15)
    jitter_y = np.random.uniform(-0.15, 0.15)

    ax.scatter(row['x_pos'] + jitter_x, row['y_pos'] + jitter_y,
               s=size, c=color, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Network Depth', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Pitch Hz', fontsize=14, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Early\n(Layers 0-7)', 'Middle\n(Layers 8-14)', 'Late\n(Layers 15-21)'], fontsize=11)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Strings', 'Drum Loops', 'Vocal'], fontsize=11)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.grid(True, alpha=0.2, linestyle='--')

# Add legends
from matplotlib.patches import Patch

dataset_legend_elements = [Patch(facecolor=dataset_colors[ds], edgecolor='black', alpha=0.6,
                                  label=ds.replace('_', ' ').title())
                            for ds in ['strings', 'vocals', 'stimuli', 'drum_loops']]
dataset_legend = ax.legend(handles=dataset_legend_elements, loc='upper left',
                           fontsize=10, title='Dataset', framealpha=0.9)
ax.add_artist(dataset_legend)  # Keep this legend when adding the next one

# Size legend (correlation strength)
size_legend_corrs = [0.3, 0.5]
size_legend_elements = []
for corr in size_legend_corrs:
    size = (corr ** 2) * 3000
    size_legend_elements.append(ax.scatter([], [], s=size, c='gray', alpha=0.6,
                                          edgecolors='black', linewidth=0.5,
                                          label=f'r={corr:.1f}'))
ax.legend(handles=size_legend_elements, loc='lower left', fontsize=9,
          title='Correlation', framealpha=0.9, scatterpoints=1)

# Plot BPM
ax = axes[1]
for _, row in topn_bpm.iterrows():
    # Size proportional to correlation (scale for visibility)
    size = (row['bpm_mean_corr'] ** 2) * 3000
    color = dataset_colors[row['dataset']]

    # Add jitter to avoid overlapping points at same position
    jitter_x = np.random.uniform(-0.15, 0.15)
    jitter_y = np.random.uniform(-0.15, 0.15)

    ax.scatter(row['x_pos'] + jitter_x, row['y_pos'] + jitter_y,
               s=size, c=color, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Network Depth', fontsize=13, fontweight='bold')
ax.set_ylabel('Model', fontsize=13, fontweight='bold')
ax.set_title('BPM', fontsize=14, fontweight='bold')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Early\n(Layers 0-7)', 'Middle\n(Layers 8-14)', 'Late\n(Layers 15-21)'], fontsize=11)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Strings', 'Drum Loops', 'Vocal'], fontsize=11)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.grid(True, alpha=0.2, linestyle='--')


plt.tight_layout()

# Save figure
import pathlib
base_path = pathlib.Path(__file__).parent.parent.resolve()
base_path = base_path / "results" / "6_cluster"
output_path = base_path / 'top_clusters_spatial.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")

plt.close()

