import numpy as np
from pathlib import Path
import json
sections = ['early', 'middle', 'late']

    
    
base_path = Path(f"/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results")
models = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']
                
for i in range (4,11):
    cluster_path = base_path / f"{i}_cluster"
    imbalanced_count = 0
    balanced_count = 0
    for model in models:
        for dataset in datasets:
            cross_file = cluster_path / model / dataset / 'cross_layer_clustering_results.json'
            if cross_file.exists():
                with open(cross_file, 'r') as f:
                    cross_data = json.load(f)
                
                for section in ['early', 'middle', 'late']:
                    if section in cross_data:
                        labels = cross_data[section]['cluster_labels']
                        unique, counts = np.unique(labels, return_counts=True)
                        
                        # Check if any cluster dominates (>60%) or is tiny (<5%)
                        max_pct = counts.max() / len(labels)
                        min_pct = counts.min() / len(labels)
                        
                        if max_pct > 0.6 or min_pct < 0.05:
                            imbalanced_count += 1
                        else:
                            balanced_count += 1

    total = imbalanced_count + balanced_count
    print(i)
    print(f"Balanced clusters: {balanced_count}/{total} ({100*balanced_count/total:.1f}%)")
    print(f"Imbalanced clusters: {imbalanced_count}/{total} ({100*imbalanced_count/total:.1f}%)\n")