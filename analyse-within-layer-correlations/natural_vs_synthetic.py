#!/usr/bin/env python3
"""
natural_vs_synthetic_v2.py
Statistical Tests: Synthetic vs Natural Audio Robustness
4-panel plot: mean |r| (top) and % neurons above null p95 (bottom) for pitch and BPM.
Uses Wilcoxon signed-rank throughout for consistency (BPM differences non-normal).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

print("="*80)
print("STATISTICAL TESTS: SYNTHETIC vs NATURAL AUDIO")
print("Metrics: mean |r|  AND  % neurons above null p95")
print("="*80)

import pathlib
results_path = pathlib.Path(__file__).parent.parent.resolve()
results_path = results_path / "results" /  "6_cluster"
models            = ["strings", "drum_loops", "taylor_vocal"]
natural_datasets  = ["strings", "vocals", "drum_loops"]
synthetic_datasets = ["stimuli"]

# ── load per-layer mean |r| and pct_above_p95 ────────────────────────────────

all_data = []

for model in models:
    for dataset in natural_datasets + synthetic_datasets:
        var_path  = results_path / model / dataset / 'variance_correlation.json'
        perm_path = results_path / model / dataset / 'permutation_baseline.json'

        if not var_path.exists():
            continue

        with open(var_path) as f:
            var = json.load(f)

        null_p95 = {}
        if perm_path.exists():
            with open(perm_path) as f:
                perm = json.load(f)
            for prop in ['pitch', 'bpm']:
                if prop in perm and 'null_p95_r' in perm[prop]:
                    null_p95[prop] = perm[prop]['null_p95_r']

        content_type = 'synthetic' if dataset in synthetic_datasets else 'natural'

        for layer_name, layer_data in var.items():
            if layer_name.startswith('section_'):
                continue
            for prop in ['pitch', 'bpm']:
                if prop not in layer_data:
                    continue
                corrs = np.array(layer_data[prop].get('all_correlations', []))
                if len(corrs) == 0:
                    continue
                mean_r = float(np.mean(np.abs(corrs)))
                pct = float((corrs > null_p95[prop]).mean() * 100) if prop in null_p95 else np.nan
                all_data.append({
                    'model':        model,
                    'dataset':      dataset,
                    'content_type': content_type,
                    'layer':        layer_name,
                    'property':     prop,
                    'mean_r':       mean_r,
                    'pct_above_p95': pct,
                    'null_p95':     null_p95.get(prop, np.nan),
                })

df = pd.DataFrame(all_data)

print(f"\nLoaded {len(df)} layer×property observations")
print(f"  Natural:   {len(df[df['content_type']=='natural'])} obs")
print(f"  Synthetic: {len(df[df['content_type']=='synthetic'])} obs")

# ── layer-matched comparison ──────────────────────────────────────────────────

print("\n" + "="*80)
print("LAYER-MATCHED WILCOXON (Within-Layer, synthetic > natural)")
print("="*80)

results = {}   # (feature, metric) -> comp_df, w_stat, w_p

for feature, excl_dataset in [('pitch', 'drum_loops'), ('bpm', 'vocals')]:
    print(f"\n--- {feature.upper()} ---")
    feat_df = df[(df['property'] == feature) & (df['dataset'] != excl_dataset)].copy()

    for metric in ['mean_r', 'pct_above_p95']:
        layer_comparison = []
        for model in models:
            model_df = feat_df[feat_df['model'] == model]
            for layer in model_df['layer'].unique():
                layer_df = model_df[model_df['layer'] == layer]
                nat = layer_df[layer_df['content_type'] == 'natural'][metric].mean()
                syn = layer_df[layer_df['content_type'] == 'synthetic'][metric].mean()
                if not (np.isnan(nat) or np.isnan(syn)):
                    layer_comparison.append({
                        'model': model, 'layer': layer,
                        'natural': nat, 'synthetic': syn,
                        'difference': syn - nat,
                    })

        comp_df = pd.DataFrame(layer_comparison)
        w_stat, w_p = wilcoxon(comp_df['synthetic'], comp_df['natural'], alternative='greater')

        unit = '%' if metric == 'pct_above_p95' else ''
        print(f"\n  [{metric}]  N pairs={len(comp_df)}")
        print(f"    Mean natural:   {comp_df['natural'].mean():.3f}{unit}")
        print(f"    Mean synthetic: {comp_df['synthetic'].mean():.3f}{unit}")
        print(f"    Wilcoxon  W={w_stat:.1f}, p={w_p:.6f}", end='  ')
        print("***" if w_p < 0.001 else ("**" if w_p < 0.01 else ("*" if w_p < 0.05 else "n.s.")))

        results[(feature, metric)] = (comp_df, w_stat, w_p)

# ── 4-panel plot ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Natural vs Synthetic: Layer-Matched Comparison',
             fontsize=14, fontweight='bold')

row_configs = [
    ('mean_r',       'Mean |r|',                  None),
    ('pct_above_p95','% neurons above null p95',  5.0),
]

col_configs = [
    ('pitch', 'drum_loops', 'Pitch'),
    ('bpm',   'vocals',     'BPM'),
]

for row_idx, (metric, ylabel, null_line) in enumerate(row_configs):
    for col_idx, (feature, _, feature_name) in enumerate(col_configs):
        ax = axes[row_idx][col_idx]
        comp_df, w_stat, w_p = results[(feature, metric)]

        for _, row in comp_df.iterrows():
            ax.plot([0, 1], [row['natural'], row['synthetic']], 'o-',
                    alpha=0.3, linewidth=1, markersize=4, color='gray')

        nat_mean = comp_df['natural'].mean()
        syn_mean = comp_df['synthetic'].mean()
        ax.plot([0, 1], [nat_mean, syn_mean], 'ro-',
                linewidth=3, markersize=12, label='Mean', zorder=10)

        if null_line is not None:
            ax.axhline(y=null_line, color='blue', linestyle=':', linewidth=1.5,
                       alpha=0.6, label=f'Null ({null_line}%)')

        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Natural', 'Synthetic'], fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{feature_name}\nW={w_stat:.0f}, p={w_p:.4f}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        sig = '***' if w_p < 0.001 else ('**' if w_p < 0.01 else ('*' if w_p < 0.05 else 'n.s.'))
        ax.text(0.5, 0.95, sig, ha='center', va='top',
                transform=ax.transAxes, fontsize=22, fontweight='bold')

        if metric == 'pct_above_p95':
            nat_label, syn_label = f'{nat_mean:.1f}%', f'{syn_mean:.1f}%'
        else:
            nat_label, syn_label = f'{nat_mean:.3f}', f'{syn_mean:.3f}'
        ax.text(-0.15, nat_mean, nat_label, ha='right', va='center',
                fontsize=9, fontweight='bold')
        ax.text(1.15, syn_mean, syn_label, ha='left', va='center',
                fontsize=9, fontweight='bold')

plt.tight_layout()
output_path = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/plots') / 'natural_vs_synthetic_4panel.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to {output_path}")
plt.close()
