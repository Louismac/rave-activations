# ============================================================================
# 3. CRITICAL COMPARISON: Cluster vs Layer Correlations
# ============================================================================
import numpy as np
from pathlib import Path
from util import load_data
MIN_NEURONS = 5  # ignore clusters smaller than this

bpm_mean = []
pitch_mean = []
bpm_max = []
pitch_max = []
k = range(4,11)
k = [6]
for i in k:
    layer_df, cluster_df = load_data(i)
    print(layer_df, cluster_df)
    print("\n" + "="*80)
    print("3. CRITICAL COMPARISON: CLUSTER vs LAYER CORRELATIONS")
    print("="*80)

    # For pitched datasets only (exclude drum_loops — no pitch data)
    pitched_cluster = cluster_df[cluster_df['dataset'] != 'drum_loops']
    pitched_layer = layer_df[layer_df['dataset'] != 'drum_loops']

    # For BPM: exclude vocals (at-chance permutation baseline)
    bpm_cluster = cluster_df[cluster_df['dataset'] != 'vocals']
    bpm_layer = layer_df[layer_df['dataset'] != 'vocals']

    # Threshold applied only for best-cluster lookups
    pitched_cluster_filtered = pitched_cluster[pitched_cluster['n_neurons'] >= MIN_NEURONS]
    bpm_cluster_filtered = bpm_cluster[bpm_cluster['n_neurons'] >= MIN_NEURONS]

    print(f"num pitched clusters cross layer {len(pitched_cluster)}")
    print(f"num pitched clusters within layer {len(pitched_layer)}")

    print("\n--- PITCH Hz CORRELATION ---")
    print(f"\nCross-layer CLUSTERS:")
    pitch_mean.append(pitched_cluster['pitch_mean_corr'].mean())
    pitch_max.append(pitched_cluster_filtered['pitch_mean_corr'].max())
    print(f"  Mean (all clusters): {pitched_cluster['pitch_mean_corr'].mean():.3f}")
    print(f"  Max  (>={MIN_NEURONS} neurons): {pitched_cluster_filtered['pitch_mean_corr'].max():.3f}")
    print(f"  From Best cluster: {pitched_cluster_filtered.loc[pitched_cluster_filtered['pitch_mean_corr'].idxmax()][['model', 'dataset', 'section', 'cluster', 'n_neurons']].to_dict()}")

    print(f"\nPer-LAYER (baseline):")
    print(f"  Mean: {pitched_layer['pitch_corr'].mean():.3f}")
    print(f"  Max:  {pitched_layer['pitch_corr'].max():.3f}")
    print(f"  From Best layer: {pitched_layer.loc[pitched_layer['pitch_corr'].idxmax()][['model', 'dataset', 'layer']].to_dict()}")

    pitch_improvement = pitched_cluster_filtered['pitch_mean_corr'].max() / pitched_layer['pitch_corr'].max()
    print(f"\n🎯 PITCH IMPROVEMENT IN CEILING (MAX): {pitch_improvement:.2f}× (cluster vs layer)")

    print(f"num bpm clusters cross layer {len(cluster_df)}")
    print(f"num bpm clusters within layer {len(layer_df)}")

    print("\n--- BPM CORRELATION ---")
    print(f"\nCross-layer CLUSTERS:")
    bpm_mean.append(bpm_cluster['bpm_mean_corr'].mean())
    bpm_max.append(bpm_cluster_filtered['bpm_mean_corr'].max())
    print(f"  Mean (all clusters): {bpm_cluster['bpm_mean_corr'].mean():.3f}")
    print(f"  Max  (>={MIN_NEURONS} neurons): {bpm_cluster_filtered['bpm_mean_corr'].max():.3f}")
    print(f"  From Best cluster: {bpm_cluster_filtered.loc[bpm_cluster_filtered['bpm_mean_corr'].idxmax()][['model', 'dataset', 'section', 'cluster', 'n_neurons']].to_dict()}")

    print(f"\nPer-LAYER (baseline):")
    print(f"  Mean: {bpm_layer['bpm_corr'].mean():.3f}")
    print(f"  Max:  {bpm_layer['bpm_corr'].max():.3f}")
    print(f"  From Best layer: {bpm_layer.loc[bpm_layer['bpm_corr'].idxmax()][['model', 'dataset', 'layer']].to_dict()}")

    bpm_improvement = bpm_cluster_filtered['bpm_mean_corr'].max() / bpm_layer['bpm_corr'].max()
    print(f"\n🎯 BPM IMPROVEMENT IN CEILING (MAX): {bpm_improvement:.2f}× (cluster vs layer)")

print(pitch_mean)
print(bpm_mean)
print(pitch_max)
print(bpm_max)


datasets_ordered = ['strings', 'vocals', 'stimuli', 'drum_loops']
ds_pitch_mean, ds_pitch_max = [], []
ds_bpm_mean,   ds_bpm_max   = [], []

for dataset in datasets_ordered:
    dc = cluster_df[cluster_df['dataset'] == dataset]
    dl = layer_df[layer_df['dataset'] == dataset]

    if dataset != 'drum_loops':
        pm = dc['pitch_mean_corr'].mean() / dl['pitch_corr'].mean()
        px = dc['pitch_mean_corr'].max()  / dl['pitch_corr'].max()
        ds_pitch_mean.append(pm)
        ds_pitch_max.append(px)
    else:
        ds_pitch_mean.append(np.nan)
        ds_pitch_max.append(np.nan)

    if dataset != 'vocals':
        bm = dc['bpm_mean_corr'].mean() / dl['bpm_corr'].mean()
        bx = dc['bpm_mean_corr'].max()  / dl['bpm_corr'].max()
        ds_bpm_mean.append(bm)
        ds_bpm_max.append(bx)
    else:
        ds_bpm_mean.append(np.nan)
        ds_bpm_max.append(np.nan)

models_ordered = ['strings', 'drum_loops', 'taylor_vocal']
model_labels = {'strings': 'Strings', 'drum_loops': 'Drum Loops', 'taylor_vocal': 'Vocal'}
m_pitch_mean, m_pitch_max = [], []
m_bpm_mean,   m_bpm_max   = [], []

for model in models_ordered:
    mc = cluster_df[cluster_df['model'] == model]
    ml = layer_df[layer_df['model'] == model]
    print(f"\n{model.upper()}:")

    # Pitch: exclude drum_loops
    mcp = mc[mc['dataset'] != 'drum_loops']
    mlp = ml[ml['dataset'] != 'drum_loops']
    pm = mcp['pitch_mean_corr'].mean() / mlp['pitch_corr'].mean()
    px = mcp['pitch_mean_corr'].max()  / mlp['pitch_corr'].max()
    m_pitch_mean.append(pm)
    m_pitch_max.append(px)

    # BPM: exclude vocals
    mcb = mc[mc['dataset'] != 'vocals']
    mlb = ml[ml['dataset'] != 'vocals']
    bm = mcb['bpm_mean_corr'].mean() / mlb['bpm_corr'].mean()
    bx = mcb['bpm_mean_corr'].max()  / mlb['bpm_corr'].max()
    m_bpm_mean.append(bm)
    m_bpm_max.append(bx)

# ============================================================================
# PLOT DATASET-SPECIFIC AND MODEL-SPECIFIC PATTERNS
# ============================================================================
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("PLOTTING DATASET-SPECIFIC AND MODEL-SPECIFIC PATTERNS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Cross-Layer Clustering Improvement: Dataset and Model Stratification', fontsize=16, fontweight='bold')
width = 0.18
bar_offset = width * 1.5

def draw_bars(ax, group_labels, pitch_mean, pitch_max, bpm_mean, bpm_max, xlabel):
    x = np.arange(len(group_labels))
    ax.bar(x - bar_offset, pitch_mean, width, label='Pitch (Mean)', alpha=0.8, color='steelblue')
    ax.bar(x - width/2,    pitch_max,  width, label='Pitch (Max)',  alpha=0.8, color='lightsteelblue')
    ax.bar(x + width/2,    bpm_mean,   width, label='BPM (Mean)',   alpha=0.8, color='coral')
    ax.bar(x + bar_offset, bpm_max,    width, label='BPM (Max)',    alpha=0.8, color='lightsalmon')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No improvement')
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel('Improvement Ratio (Cluster / Layer)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    all_vals = [v for arr in [pitch_mean, pitch_max, bpm_mean, bpm_max]
                for v in arr if not np.isnan(v)]
    ax.set_ylim([0.85, max(all_vals) * 1.15])
    ax.text(0.02, 0.98, 'Descriptive comparison only\n(clusters derived from layers)',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

draw_bars(axes[0],
          [ds.replace('_', ' ').title() for ds in datasets_ordered],
          ds_pitch_mean, ds_pitch_max, ds_bpm_mean, ds_bpm_max,
          'Dataset')
axes[0].set_title('Dataset-Specific Performance', fontsize=14, fontweight='bold')

draw_bars(axes[1],
          [model_labels[m] for m in models_ordered],
          m_pitch_mean, m_pitch_max, m_bpm_mean, m_bpm_max,
          'Model')
axes[1].set_title('Model-Specific Performance', fontsize=14, fontweight='bold')

plt.tight_layout()
import pathlib
base_path = pathlib.Path(__file__).parent.parent.resolve()
base_path = base_path / "results" / "6_cluster"
output_path = base_path / 'clustering_improvement_stratified.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
