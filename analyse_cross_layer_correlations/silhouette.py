import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("="*80)
print("COMPREHENSIVE CROSS-LAYER CLUSTERING ANALYSIS")
print("="*80)
for i in range (4,11):
    print("="*80)
    print(f"COMPREHENSIVE CROSS-LAYER CLUSTERING ANALYSIS for k={i}")
    print("="*80)
    base_path = Path(f"/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results/{i}_cluster")
    models = ['strings', 'drum_loops', 'taylor_vocal']
    datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

    # Load original per-layer results for comparison
    orig_path = base_path

    # ============================================================================
    # 1. Overall Quality Comparison: Cross-Layer vs Within-Layer
    # ============================================================================
    print("\n" + "="*80)
    print("1. CLUSTERING QUALITY: CROSS-LAYER vs WITHIN-LAYER")
    print("="*80)

    cross_layer_scores = []
    within_layer_scores = []

    for model in models:
        for dataset in datasets:
            # Cross-layer
            cross_file = base_path / model / dataset / 'cross_layer_clustering_results.json'
            if cross_file.exists():
                with open(cross_file, 'r') as f:
                    cross_data = json.load(f)
                
                for section in ['early', 'middle', 'late']:
                    if section in cross_data:
                        cross_layer_scores.append({
                            'model': model,
                            'dataset': dataset,
                            'section': section,
                            'silhouette': cross_data[section]['silhouette_score'],
                            'n_clusters': cross_data[section]['n_clusters'],
                            'n_neurons': cross_data[section]['n_neurons'],
                            'n_samples': len(cross_data[section]['cluster_labels'])
                        })
            
            # Within-layer (original)
            orig_file = orig_path / model / dataset / 'table1_layer_statistics.csv'
            if orig_file.exists():
                stats_df = pd.read_csv(orig_file).dropna()
                within_layer_scores.extend(stats_df['Silhouette'].values)

    cross_df = pd.DataFrame(cross_layer_scores)
    within_array = np.array(within_layer_scores)

    # print(f"\nCROSS-LAYER (sectioned):") 
    print(f"  Mean silhouette: {cross_df['silhouette'].mean():.3f} ± {cross_df['silhouette'].std():.3f}")
    print(f"  Range: {cross_df['silhouette'].min():.3f} to {cross_df['silhouette'].max():.3f}")
    # print(f"  Sections analyzed: {len(cross_df)}")

    # print(f"\nWITHIN-LAYER (original):")
    # print(f"  Mean silhouette: {within_array.mean():.3f} ± {within_array.std():.3f}")
    # print(f"  Range: {within_array.min():.3f} to {within_array.max():.3f}")
    # print(f"  Layers analyzed: {len(within_array)}")

    # # Statistical test
    # t_stat, p_val = stats.ttest_ind(cross_df['silhouette'], within_array)
    # improvement = ((cross_df['silhouette'].mean() - within_array.mean()) / within_array.mean()) * 100

    # print(f"\n📊 STATISTICAL COMPARISON:")
    # print(f"  Cross-layer improvement: {improvement:+.1f}%")
    # print(f"  t-test: t={t_stat:.2f}, p={p_val:.4f}")
    # if p_val < 0.001:
    #     print(f"  *** HIGHLY SIGNIFICANT improvement (p<0.001)")
    # elif p_val < 0.05:
    #     print(f"  ** Significant improvement (p<0.05)")
    # else:
    #     print(f"  No significant difference")

    # Quality distribution
    cross_good = (cross_df['silhouette'] > 0.5).sum()
    within_good = (within_array > 0.5).sum()

    # print(f"\nGOOD CLUSTERING (silhouette > 0.5):")
    # print(f"  Cross-layer: {cross_good}/{len(cross_df)} ({100*cross_good/len(cross_df):.1f}%)")
    # print(f"  Within-layer: {within_good}/{len(within_array)} ({100*within_good/len(within_array):.1f}%)")
    # print(f"  Improvement: {100*(cross_good/len(cross_df) - within_good/len(within_array)):.1f} percentage points")

