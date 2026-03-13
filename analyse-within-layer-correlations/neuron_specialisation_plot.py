#!/usr/bin/env python3
"""
neuron_specialisation_plot.py
Stacked horizontal bar chart of neuron specialisation categories.
Three panels: Overall | By Model | By Dataset
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

base_path = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results/6_cluster')
models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

# ── load data (same as dense_sparse_v2.py) ───────────────────────────────────

all_data = []

for model in models:
    for dataset in datasets:
        combo_path = base_path / model / dataset
        var_path   = combo_path / 'variance_correlation.json'
        perm_path  = combo_path / 'permutation_baseline.json'

        if not var_path.exists():
            continue

        with open(var_path) as f:
            var = json.load(f)

        null_p95 = {}
        if perm_path.exists():
            with open(perm_path) as f:
                perm = json.load(f)
            for prop in ['pitch', 'bpm']:
                if prop in perm:
                    null_p95[prop] = perm[prop].get('null_p95_r', None)

        for layer_name, layer_data in var.items():
            if layer_name.startswith('section_'):
                continue
            for prop in ['pitch', 'bpm']:
                if prop not in layer_data:
                    continue
                corrs = np.array(layer_data[prop].get('all_correlations', []))
                if len(corrs) == 0:
                    continue
                all_data.append({
                    'model':    model,
                    'dataset':  dataset,
                    'layer':    layer_name,
                    'property': prop,
                    'corrs':    corrs,
                    'null_p95': null_p95.get(prop, None),
                })

df = pd.DataFrame(all_data)

# ── specialisation counter ────────────────────────────────────────────────────

def compute_specialisation(subset_df):
    """Return dict of counts for pitch_only/bpm_only/dual/neither."""
    cats = {'pitch_only': 0, 'bpm_only': 0, 'dual': 0, 'neither': 0}
    for (model, dataset, layer), grp in subset_df.groupby(['model', 'dataset', 'layer']):
        if dataset in ('drum_loops', 'vocals'):
            continue
        p_rows = grp[grp['property'] == 'pitch']
        b_rows = grp[grp['property'] == 'bpm']
        if len(p_rows) == 0 or len(b_rows) == 0:
            continue
        p_corrs = p_rows.iloc[0]['corrs']
        b_corrs = b_rows.iloc[0]['corrs']
        p_thr   = p_rows.iloc[0]['null_p95']
        b_thr   = b_rows.iloc[0]['null_p95']
        if p_thr is None or b_thr is None:
            continue
        if len(p_corrs) != len(b_corrs):
            n = min(len(p_corrs), len(b_corrs))
            p_corrs, b_corrs = p_corrs[:n], b_corrs[:n]
        p_sig = p_corrs > p_thr
        b_sig = b_corrs > b_thr
        cats['pitch_only'] += int(( p_sig & ~b_sig).sum())
        cats['bpm_only']   += int((~p_sig &  b_sig).sum())
        cats['dual']       += int(( p_sig &  b_sig).sum())
        cats['neither']    += int((~p_sig & ~b_sig).sum())
    return cats


def to_pcts(cats):
    total = sum(cats.values())
    if total == 0:
        return {k: 0.0 for k in cats}, 0
    return {k: 100 * v / total for k, v in cats.items()}, total


# ── collect data for each panel ───────────────────────────────────────────────

model_labels   = {'strings': 'Strings', 'drum_loops': 'Drum Loops', 'taylor_vocal': 'Vocal'}
dataset_labels = {'strings': 'Strings (natural)', 'stimuli': 'Stimuli (synthetic)'}

panel_data = {}

# Overall
cats = compute_specialisation(df)
pcts, total = to_pcts(cats)
panel_data['overall'] = [('Overall', pcts, total)]

# By model
panel_data['model'] = []
for model in models:
    cats = compute_specialisation(df[df['model'] == model])
    pcts, total = to_pcts(cats)
    panel_data['model'].append((model_labels[model], pcts, total))

# By dataset (only strings and stimuli are valid for both features)
panel_data['dataset'] = []
for dataset in ['strings', 'stimuli']:
    cats = compute_specialisation(df[df['dataset'] == dataset])
    pcts, total = to_pcts(cats)
    panel_data['dataset'].append((dataset_labels[dataset], pcts, total))

# ── plot ──────────────────────────────────────────────────────────────────────

COLORS = {
    'pitch_only': '#4878CF',   # blue
    'bpm_only':   '#E84646',   # red
    'dual':       '#8B5CF6',   # purple
    'neither':    '#AAAAAA',   # grey
}
CAT_ORDER  = ['pitch_only', 'bpm_only', 'dual', 'neither']
CAT_LABELS = {'pitch_only': 'Pitch-only', 'bpm_only': 'BPM-only',
              'dual': 'Dual', 'neither': 'Neither'}

fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                         gridspec_kw={'width_ratios': [1, 3, 2]})
fig.suptitle('Neuron Specialisation: Pitch-only, BPM-only, Dual, Neither\n'
             '(threshold: null p95 per combo — chance gives 5% above)',
             fontsize=13, fontweight='bold')

panel_titles = ['Overall', 'By Model', 'By Dataset']

for ax, title, key in zip(axes, panel_titles, ['overall', 'model', 'dataset']):
    rows = panel_data[key]
    bar_labels = [r[0] for r in rows]
    y = np.arange(len(rows))

    lefts = np.zeros(len(rows))
    for cat in CAT_ORDER:
        vals = np.array([r[1][cat] for r in rows])
        bars = ax.barh(y, vals, left=lefts, color=COLORS[cat],
                       label=CAT_LABELS[cat], edgecolor='white', linewidth=0.5)
        # Label segments ≥ 8%
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 8:
                ax.text(lefts[i] + val / 2, i, f'{val:.0f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        lefts += vals

    # N= annotation at right edge
    for i, (_, _, total) in enumerate(rows):
        ax.text(101, i, f'N={total:,}', va='center', fontsize=8, color='#444')

    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% neurons', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axvline(x=5, color='black', linestyle=':', linewidth=1, alpha=0.4)
    ax.grid(True, axis='x', alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

# Single shared legend below the figure
handles = [mpatches.Patch(color=COLORS[c], label=CAT_LABELS[c]) for c in CAT_ORDER]
fig.legend(handles=handles, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
output_path = Path('/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/plots') / 'neuron_specialisation.png'
output_path.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved to {output_path}")
plt.close()
