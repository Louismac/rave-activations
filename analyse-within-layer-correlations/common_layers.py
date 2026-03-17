import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
base_path = pathlib.Path(__file__).parent.parent.resolve()
base_path = base_path / "results" /  "6_cluster"
models = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

# Load all data
all_data = {}
for model in models:
    all_data[model] = {}
    for dataset in datasets:
        path = base_path / model / dataset
        table1_path = path / 'table1_layer_statistics.csv'
        if table1_path.exists():
            df = pd.read_csv(table1_path)
            df.columns = ['layer', 'silhouette', 'pitch_hz_resp_pct', 'pitch_hz_correlation', 
                         'pitch_class_resp_pct', 'pitch_class_correlation',
                         'bpm_resp_pct', 'bpm_correlation']
            all_data[model][dataset] = {'layer_stats': df}


print(f"\n✓ Loaded data for {len(models)} models × {len(datasets)} datasets\n")

def get_top_layers_for_property(all_data, models, dataset, property_name, correlation_col, top_n=5):
    """
    Get top N layers for a given property across all models.

    Args:
        all_data: Dictionary of model -> dataset -> stats
        models: List of model names
        dataset: Dataset name
        property_name: Human-readable property name (for display)
        correlation_col: Column name in stats dataframe
        top_n: Number of top layers to return

    Returns:
        Dictionary mapping model name to set of top layer names
    """
    layers = {}
    for model in models:
        if dataset in all_data[model]:
            stats = all_data[model][dataset]['layer_stats']
            sorted_stats = stats.iloc[abs(stats[correlation_col]).argsort()[::-1]]
            layers[model] = set(sorted_stats.head(top_n)['layer'].values)
    return layers

def find_common_layers(layer_sets, models):
    """
    Find layers common to all models.

    Args:
        layer_sets: Dictionary mapping model name to set of layers
        models: List of all model names

    Returns:
        Set of layers common to all models, or None if not all models present
    """
    if len(layer_sets) == len(models):
        return set.intersection(*[layer_sets[model] for model in models])
    return None

print("\n" + "="*80)
print("UNIVERSAL LAYER ORGANIZATION")
print("="*80)

for dataset in datasets:
    print(f"\n--- Dataset: {dataset.upper()} ---")
    bpm_common = None
    pitch_hz_common = None
    pitch_class_common = None

    if dataset == 'drum_loops':
        # Only BPM for drums
        bpm_layers = get_top_layers_for_property(all_data, models, dataset, 'BPM', 'bpm_correlation')
        bpm_common = find_common_layers(bpm_layers, models)
    elif dataset == 'vocals':
        pitch_hz_layers = get_top_layers_for_property(all_data, models, dataset, 'Pitch Hz', 'pitch_hz_correlation')
        pitch_class_layers = get_top_layers_for_property(all_data, models, dataset, 'Pitch Class', 'pitch_class_correlation')
        pitch_hz_common = find_common_layers(pitch_hz_layers, models)
        pitch_class_common = find_common_layers(pitch_class_layers, models)
    else:
        bpm_layers = get_top_layers_for_property(all_data, models, dataset, 'BPM', 'bpm_correlation')
        bpm_common = find_common_layers(bpm_layers, models)
        pitch_hz_layers = get_top_layers_for_property(all_data, models, dataset, 'Pitch Hz', 'pitch_hz_correlation')
        pitch_class_layers = get_top_layers_for_property(all_data, models, dataset, 'Pitch Class', 'pitch_class_correlation')
        pitch_hz_common = find_common_layers(pitch_hz_layers, models)
        pitch_class_common = find_common_layers(pitch_class_layers, models)    

        

    if bpm_common is not None:
        print(f"  BPM-sensitive layers common to ALL models: {sorted(bpm_common) if bpm_common else 'None'}")
    if pitch_hz_common is not None:
        print(f"  Pitch Hz-sensitive layers common to ALL models: {sorted(pitch_hz_common) if pitch_hz_common else 'None'}")
    if pitch_class_common is not None:
        print(f"  Pitch Class-sensitive layers common to ALL models: {sorted(pitch_class_common) if pitch_class_common else 'None'}")


