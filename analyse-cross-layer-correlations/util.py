import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

print("="*80)
print("CROSS-LAYER CLUSTER CORRELATION ANALYSIS")
print("="*80)



def load_data(i):
    print(f"loading data for cluster {i}")
    base_path = Path(f"/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results")
    models = ['strings', 'drum_loops', 'taylor_vocal']
    datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']
    sections = ['early', 'middle', 'late']
    base_path = base_path / f"{i}_cluster"

    cluster_correlations = []

    for model in models:
        for dataset in datasets:
            corr_file = base_path / model / dataset / 'cross_layer_cluster_correlation_all_neurons.json'
            
            if corr_file.exists():
                with open(corr_file, 'r') as f:
                    data = json.load(f)
                
                for section in sections:
                    if section in data:
                        section_data = data[section]
                        
                        # Each section has clusters
                        for cluster_key, cluster_info in section_data.items():
                            if 'properties' in cluster_info:
                                props = cluster_info['properties']
                                
                                # Extract pitch correlation
                                pitch_corr = props.get('pitch', {}).get('mean_correlation', 0)
                                pitch_max = props.get('pitch', {}).get('max_correlation', 0)
                                pitch_responsive = props.get('pitch', {}).get('n_responsive_neurons', 0)
                                
                                # Extract BPM correlation
                                bpm_corr = props.get('bpm', {}).get('mean_correlation', 0)
                                bpm_max = props.get('bpm', {}).get('max_correlation', 0)
                                bpm_responsive = props.get('bpm', {}).get('n_responsive_neurons', 0)
                                n_neurons = len(cluster_info['neuron_origins'])
                                cluster_correlations.append({
                                    'model': model,
                                    'dataset': dataset,
                                    'section': section,
                                    'cluster': cluster_key,
                                    'n_neurons': n_neurons,
                                    'pitch_mean_corr': abs(pitch_corr),
                                    'pitch_max_corr': abs(pitch_max),
                                    'pitch_responsive': pitch_responsive,
                                    'bpm_mean_corr': abs(bpm_corr),
                                    'bpm_max_corr': abs(bpm_max),
                                    'bpm_responsive': bpm_responsive
                                })

    cluster_df = pd.DataFrame(cluster_correlations)

    layer_correlations = []

    for model in models:
        for dataset in datasets:
            layer_file = base_path / model / dataset / 'table1_layer_statistics.csv'
            
            if layer_file.exists():
                df = pd.read_csv(layer_file)
                
                for _, row in df.iterrows():
                    pitch_corr = abs(row['Pitch Hz Corr']) if dataset != 'drum_loops' else 0
                    layer_correlations.append({
                        'model': model,
                        'dataset': dataset,
                        'layer': row['Layer'],
                        'pitch_corr': pitch_corr,
                        'bpm_corr': abs(row['BPM Corr'])
                    })

    layer_df = pd.DataFrame(layer_correlations)

    return layer_df, cluster_df
