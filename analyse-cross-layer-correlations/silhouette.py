import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("="*80)
print("COMPREHENSIVE CROSS-LAYER CLUSTERING ANALYSIS")
print("="*80)
k = range (4,11)
k = [6]
for i in k:
    print("="*80)
    print(f"COMPREHENSIVE CROSS-LAYER CLUSTERING ANALYSIS for k={i}")
    print("="*80)
    import pathlib
    base_path = pathlib.Path(__file__).parent.parent.resolve()
    base_path = base_path / "results" 
    base_path = base_path / f"{i}_cluster"
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
            cross_file = base_path / model / dataset / 'cross_layer_clustering_results_all_neurons.json'
            print(cross_file)
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

    cross_df = pd.DataFrame(cross_layer_scores)
    within_array = np.array(within_layer_scores)

    # print(f"\nCROSS-LAYER (sectioned):") 
    print(f"  Mean silhouette: {cross_df['silhouette'].mean():.3f} ± {cross_df['silhouette'].std():.3f}")
    print(f"  Range: {cross_df['silhouette'].min():.3f} to {cross_df['silhouette'].max():.3f}")
    print(f"  Sections analyzed: {len(cross_df)}")

    print("\n" + "="*80)
    print("2. HIERARCHICAL SECTION COMPARISON")
    print("="*80)

    for section in ['early', 'middle', 'late']:
        section_df = cross_df[cross_df['section'] == section]
        print(f"\n{section.upper()} SECTION (layers 0-7 / 8-14 / 15+):")
        print(f"  Mean silhouette: {section_df['silhouette'].mean():.3f} ± {section_df['silhouette'].std():.3f}")
        print(f"  Best: {section_df['silhouette'].max():.3f} ({section_df.loc[section_df['silhouette'].idxmax(), 'model']} + {section_df.loc[section_df['silhouette'].idxmax(), 'dataset']})")
        print(f"  Good clustering (>0.5): {(section_df['silhouette'] > 0.5).sum()}/{len(section_df)} ({100*(section_df['silhouette'] > 0.5).sum()/len(section_df):.1f}%)")
        print(f"  Mean neurons: {section_df['n_neurons'].mean():.0f}")
        print(f"  Mean samples: {section_df['n_samples'].mean():.0f}")

