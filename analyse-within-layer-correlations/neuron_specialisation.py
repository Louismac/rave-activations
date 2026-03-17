"""
dense_sparse_v2.py — neuron-level encoding analysis

Replaces the layer-mean approach in dense_sparse.py with individual neuron
correlations from variance_correlation.json, thresholded against the
data-derived null p95 from permutation_baseline.json.

Threshold: null_p95_r (pooled across layers per combo) — by definition
the null gives 5% above this, so any higher % is real signal.

NOTE: requires permutation_baseline.json to contain 'null_p95_r'
(run run_permutation_baseline.py to generate/update).

Exclusions (permutation-validated):
  - drum_loops: no pitch information
  - vocals: BPM at chance (0.1–1.1× null) — vocal stem lacks rhythmic cues
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import pathlib
base_path = pathlib.Path(__file__).parent.resolve()
base_path = base_path / "results/6_cluster'"
models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

# ── load neuron-level data ────────────────────────────────────────────────────

all_data = []

for model in models:
    for dataset in datasets:
        combo_path = base_path / model / dataset

        var_path  = combo_path / 'variance_correlation.json'
        perm_path = combo_path / 'permutation_baseline.json'

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
                continue  # skip aggregated section entries (pooled duplicates)
            for prop in ['pitch', 'bpm']:
                if prop not in layer_data:
                    continue
                corrs = np.array(layer_data[prop].get('all_correlations', []))
                if len(corrs) == 0:
                    continue
                threshold = null_p95.get(prop, None)
                pct_above = float((corrs > threshold).mean() * 100) if threshold is not None else None

                all_data.append({
                    'model':     model,
                    'dataset':   dataset,
                    'layer':     layer_name,
                    'property':  prop,
                    'n_neurons': len(corrs),
                    'corrs':     corrs,           # kept for Wilcoxon
                    'mean_r':    float(corrs.mean()),
                    'max_r':     float(corrs.max()),
                    'null_p95':  threshold,
                    'pct_above_p95': pct_above,
                })

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} layer observations across {df['model'].nunique()} models × {df['dataset'].nunique()} datasets")

# ── helper: flag combos missing null_p95 ─────────────────────────────────────

missing_p95 = df[df['null_p95'].isna()][['model', 'dataset', 'property']].drop_duplicates()
if len(missing_p95):
    print(f"\n⚠  Missing null_p95_r for {len(missing_p95)} combo(s) — run run_permutation_baseline.py:")
    print(missing_p95.to_string(index=False))

# ── section 1: overall % neurons above null p95 ──────────────────────────────

print("\n" + "="*80)
print("NEURONS ABOVE NULL P95 THRESHOLD")
print("="*80)
print("(by definition: null gives 5%; values substantially higher = real signal)")
print(f"{'':5s} NOTE: drum_loops excluded from pitch; vocals excluded from BPM\n")

pitch_df = df[(df['property'] == 'pitch') & (df['dataset'] != 'drum_loops')]
bpm_df   = df[(df['property'] == 'bpm')   & (df['dataset'] != 'vocals')]

for model in models:
    print(f"--- {model.upper()} ---")
    for dataset in datasets:
        if dataset == 'drum_loops':
            print(f"  {dataset:<12s}  Pitch: excluded (no pitch)   BPM: {bpm_df[(bpm_df['model']==model)&(bpm_df['dataset']==dataset)]['pct_above_p95'].mean():.1f}% above null p95")
            continue
        if dataset == 'vocals':
            print(f"  {dataset:<12s}  Pitch: {pitch_df[(pitch_df['model']==model)&(pitch_df['dataset']==dataset)]['pct_above_p95'].mean():.1f}% above null p95   BPM: excluded (at-chance)")
            continue
        p_pct = pitch_df[(pitch_df['model']==model) & (pitch_df['dataset']==dataset)]['pct_above_p95'].mean()
        b_pct = bpm_df[(bpm_df['model']==model)     & (bpm_df['dataset']==dataset)]['pct_above_p95'].mean()
        p_str = f"{p_pct:.1f}%" if p_pct is not None else "N/A"
        b_str = f"{b_pct:.1f}%" if b_pct is not None else "N/A"
        print(f"  {dataset:<12s}  Pitch: {p_str:>6s} above null p95   BPM: {b_str:>6s} above null p95")
    print()

# Aggregate totals
print("OVERALL (mean % across all valid combos):")
print(f"  Pitch: {pitch_df['pct_above_p95'].mean():.1f}%")
print(f"  BPM:   {bpm_df['pct_above_p95'].mean():.1f}%")

# ── section 2: per-layer profile (early/middle/late) ─────────────────────────

print("\n" + "="*80)
print("PER-SECTION PROFILE (mean % neurons above null p95)")
print("="*80)

# Assign layer index — layers are ordered keys; split into thirds
def assign_section(layer_name, all_layers_ordered):
    idx = list(all_layers_ordered).index(layer_name)
    n   = len(all_layers_ordered)
    if idx < n / 3:
        return 'early'
    elif idx < 2 * n / 3:
        return 'middle'
    else:
        return 'late'

# Compute section for each row
section_map = {}
for (model, dataset), grp in df.groupby(['model', 'dataset']):
    layers_ordered = list(dict.fromkeys(grp['layer']))  # preserve insertion order
    for layer in layers_ordered:
        section_map[(model, dataset, layer)] = assign_section(layer, layers_ordered)

df['section'] = df.apply(lambda r: section_map.get((r['model'], r['dataset'], r['layer']), 'unknown'), axis=1)

for prop, sub_df in [('pitch', pitch_df), ('bpm', bpm_df)]:
    sub_df = sub_df.copy()
    sub_df['section'] = sub_df.apply(lambda r: section_map.get((r['model'], r['dataset'], r['layer']), 'unknown'), axis=1)
    print(f"\n{prop.upper()}:")
    print(f"  {'Section':<10s} {'Mean % above null p95':>22s}")
    for section in ['early', 'middle', 'late']:
        pct = sub_df[sub_df['section'] == section]['pct_above_p95'].mean()
        print(f"  {section:<10s} {pct:>22.1f}%")


def neuron_specialisation(subset_df, label):
    """
    For each neuron in each layer, classify as:
      pitch-only  : pitch |r| > null_p95,  BPM |r| <= null_p95
      BPM-only    : BPM |r| > null_p95,    pitch |r| <= null_p95
      dual        : both above null_p95
      neither     : both below null_p95

    Only uses combos where both features are valid
    (excludes drum_loops dataset for pitch, vocals for BPM).
    """
    categories = {'pitch_only': 0, 'bpm_only': 0, 'dual': 0, 'neither': 0}

    for (model, dataset, layer), grp in subset_df.groupby(['model', 'dataset', 'layer']):
        if dataset == 'drum_loops':
            continue
        p_rows = grp[grp['property'] == 'pitch']
        b_rows = grp[grp['property'] == 'bpm']
        if len(p_rows) == 0 or len(b_rows) == 0:
            continue

        p_corrs   = p_rows.iloc[0]['corrs']
        b_corrs   = b_rows.iloc[0]['corrs']
        p_thr     = p_rows.iloc[0]['null_p95']
        b_thr     = b_rows.iloc[0]['null_p95']

        if p_thr is None or b_thr is None:
            continue
        if len(p_corrs) != len(b_corrs):
            min_n = min(len(p_corrs), len(b_corrs))
            p_corrs, b_corrs = p_corrs[:min_n], b_corrs[:min_n]

        p_sig = p_corrs > p_thr
        b_sig = b_corrs > b_thr

        categories['pitch_only'] += int(( p_sig & ~b_sig).sum())
        categories['bpm_only']   += int((~p_sig &  b_sig).sum())
        categories['dual']       += int(( p_sig &  b_sig).sum())
        categories['neither']    += int((~p_sig & ~b_sig).sum())

    total = sum(categories.values())
    if total == 0:
        print(f"{label}: insufficient data")
        return

    print(f"{label}  (N={total:,} neurons)")
    for cat, n in categories.items():
        print(f"  {cat:<12s}: {n:>6,}  ({100*n/total:>5.1f}%)")
    print()


print("\n" + "="*80)
print("NEURON SPECIALISATION (combos where both pitch & BPM valid)")
print("="*80)
print("Threshold: null_p95_r per combo (chance gives 5% above by definition)\n")

print("OVERALL")
neuron_specialisation(df, "OVERALL")

print("BY MODEL:")
for model in models:
    neuron_specialisation(df[df['model'] == model], f"  {model}")

print("BY DATASET:")
for dataset in ['strings', 'vocals', 'stimuli']:
    neuron_specialisation(df[df['dataset'] == dataset], f"  {dataset}")

# ── section 4: model × dataset summary table ─────────────────────────────────

print("\n" + "="*80)
print("SUMMARY TABLE: % NEURONS ABOVE NULL P95 (mean across layers)")
print("="*80)
print(f"\n{'Model':<15s} {'Dataset':<12s} {'Pitch % > p95':>14s} {'BPM % > p95':>12s} {'Pitch null_p95':>15s} {'BPM null_p95':>13s}")
print("-"*75)

for model in models:
    for dataset in datasets:
        p_rows = df[(df['model']==model) & (df['dataset']==dataset) & (df['property']=='pitch')]
        b_rows = df[(df['model']==model) & (df['dataset']==dataset) & (df['property']=='bpm')]

        p_str = f"{p_rows['pct_above_p95'].mean():.1f}%" if (dataset != 'drum_loops' and len(p_rows)) else "excl."
        b_str = f"{b_rows['pct_above_p95'].mean():.1f}%" if (dataset != 'vocals'     and len(b_rows)) else "excl."

        p_thr = f"{p_rows['null_p95'].iloc[0]:.4f}" if len(p_rows) and p_rows['null_p95'].iloc[0] is not None else "N/A"
        b_thr = f"{b_rows['null_p95'].iloc[0]:.4f}" if len(b_rows) and b_rows['null_p95'].iloc[0] is not None else "N/A"

        print(f"{model:<15s} {dataset:<12s} {p_str:>14s} {b_str:>12s} {p_thr:>15s} {b_thr:>13s}")
