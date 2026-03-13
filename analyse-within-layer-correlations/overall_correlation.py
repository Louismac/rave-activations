import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

k = range (4,11)
k = [6]

for i in k:
    print(i)
    base_path = Path(f"/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results/{i}_cluster")
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

    print("\n" + "="*80)
    print("CRITICAL COMPARISON: Continuous Pitch Hz vs Categorical Pitch Class vs BPM")
    print("="*80)

    all_pitch_hz = []
    all_pitch_class = []
    all_bpm = []

    for model in models:
        for dataset in datasets:
            if dataset in all_data[model]:
                stats = all_data[model][dataset]['layer_stats']
                #exclude vocals are 
                if dataset != 'vocals':
                    all_bpm.extend(stats['bpm_correlation'].abs().values)
                
                if dataset != 'drum_loops':
                    all_pitch_hz.extend(stats['pitch_hz_correlation'].abs().values)
                    all_pitch_class.extend(stats['pitch_class_correlation'].abs().values)

    print(f"\nAcross all models and pitched datasets:")
    print(f"  Pitch Hz (100ms):       mean={np.mean(all_pitch_hz):.3f} ± {np.std(all_pitch_hz):.3f}, max={np.max(all_pitch_hz):.3f}")
    print(f"  Pitch Class (4s):    mean={np.mean(all_pitch_class):.3f} ± {np.std(all_pitch_class):.3f}, max={np.max(all_pitch_class):.3f}")
    print(f"  BPM (4s):               mean={np.mean(all_bpm):.3f} ± {np.std(all_bpm):.3f}, max={np.max(all_bpm):.3f}")

    print(f"\nRatios:")
    print(f"  Pitch Hz / Pitch Class: {np.mean(all_pitch_hz) / np.mean(all_pitch_class):.2f}×")
    print(f"  BPM / Pitch Hz:         {np.mean(all_bpm) / np.mean(all_pitch_hz):.2f}×")
    print(f"  BPM / Pitch Class:      {np.mean(all_bpm) / np.mean(all_pitch_class):.2f}×")