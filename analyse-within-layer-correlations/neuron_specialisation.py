"""
dense_sparse_v3.py — neuron-level encoding analysis (4 features)

Updated for analysis across 4 features:
  pitch, bpm, spectral_centroid, spectral_bandwidth

Threshold: null_p95_r (pooled across layers per combo) — by definition
the null gives 5% above this, so any higher % is real signal.

NOTE: requires permutation_baseline.json to contain 'null_p95_r'
(run run_permutation_baseline.py to generate/update).

Exclusions (permutation-validated):
  - drum_loops dataset: weak encoding across features
  - vocals: BPM at chance in original analysis (now reversed with balanced
    sampling — included by default; remove from exclusions list if desired)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

import pathlib
base_path = pathlib.Path(__file__).parent.parent.resolve()
base_path = base_path / "results" / "6_cluster"
models = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

# All features to analyze
FEATURES = ['pitch', 'bpm', 'spectral_centroid', 'spectral_bandwidth']
# FEATURES = ['pitch', 'bpm']


# Per-feature exclusions: which (dataset, feature) combos to exclude
# Empty by default — adjust based on what your analysis warrants
EXCLUSIONS = {
    'pitch': ['drum_loops'],            # no pitch info in drums
    'bpm': [],
    'spectral_centroid': ['stimuli'],
    'spectral_bandwidth': ["stimuli"],
}

# Synthetic vs natural, and in- vs out-of-distribution groupings
SYNTHETIC_DATASETS = ['stimuli']
IN_DISTRIBUTION = {
    'strings': 'strings',
    'drum_loops': 'drum_loops',
    'taylor_vocal': 'vocals',
}
GROUPS = ['all', 'synthetic', 'natural', 'in_distribution', 'out_of_distribution']


def row_groups(model, dataset):
    """Which of GROUPS a given (model, dataset) row belongs to."""
    groups = {'all'}
    if dataset in SYNTHETIC_DATASETS:
        groups.add('synthetic')
    else:
        groups.add('natural')
        if IN_DISTRIBUTION.get(model) == dataset:
            groups.add('in_distribution')
        else:
            groups.add('out_of_distribution')
    return groups

# ── load neuron-level data ────────────────────────────────────────────────────

all_data = []

for model in models:
    for dataset in datasets:
        combo_path = base_path / model / dataset
        var_path = combo_path / 'variance_correlation.json'
        perm_path = combo_path / 'permutation_baseline.json'

        if not var_path.exists():
            print(f"skipping {var_path}")
            continue

        with open(var_path) as f:
            var = json.load(f)

        # Load null thresholds for each feature
        null_p95 = {}
        if perm_path.exists():
            with open(perm_path) as f:
                perm = json.load(f)
            for prop in FEATURES:
                if prop in perm:
                    null_p95[prop] = perm[prop].get('null_p95_r', None)

        for layer_name, layer_data in var.items():
            if layer_name.startswith('section_'):
                continue  # skip aggregated section entries

            for prop in FEATURES:
                if prop not in layer_data:
                    continue

                # Check if this combo is excluded for this feature
                if dataset in EXCLUSIONS.get(prop, []):
                    continue

                corrs = np.array(layer_data[prop].get('all_correlations', []))
                if len(corrs) == 0:
                    continue

                threshold = null_p95.get(prop, None)
                pct_above = float((corrs > threshold).mean() * 100) if threshold is not None else None

                all_data.append({
                    'model': model,
                    'dataset': dataset,
                    'layer': layer_name,
                    'property': prop,
                    'n_neurons': len(corrs),
                    'corrs': corrs,
                    'mean_r': float(corrs.mean()),
                    'max_r': float(corrs.max()),
                    'null_p95': threshold,
                    'pct_above_p95': pct_above,
                })

df = pd.DataFrame(all_data)
print(f"Loaded {len(df)} layer observations across {df['model'].nunique()} models × {df['dataset'].nunique()} datasets")

# ── flag combos missing null_p95 ─────────────────────────────────────────────

missing_p95 = df[df['null_p95'].isna()][['model', 'dataset', 'property']].drop_duplicates()
if len(missing_p95):
    print(f"\n⚠  Missing null_p95_r for {len(missing_p95)} combo(s):")
    print(missing_p95.to_string(index=False))

# ── section 1: overall % neurons above null p95 ──────────────────────────────

print("\n" + "="*80)
print("NEURONS ABOVE NULL P95 THRESHOLD (mean across layers)")
print("="*80)
print("(by definition: null gives 5% above; values substantially higher = real signal)\n")

# Print exclusion notes
for feat, excl in EXCLUSIONS.items():
    if excl:
        print(f"  Note: {feat} excludes dataset(s) {excl}")
print()

df['groups'] = df.apply(lambda r: row_groups(r['model'], r['dataset']), axis=1)

for group in GROUPS:
    print(f"--- {group.upper().replace('_', ' ')} ---")
    feat_results = []
    for feat in FEATURES:
        sub = df[(df['property'] == feat) &
                 (~df['dataset'].isin(EXCLUSIONS.get(feat, []))) &
                 df['groups'].apply(lambda g: group in g)]
        if len(sub) == 0:
            feat_results.append(f"{feat[:8]:<8s}: --N/A--")
        else:
            pct = sub['pct_above_p95'].mean()
            feat_results.append(f"{feat[:8]:<8s}: {pct:>5.1f}%")
    print("  " + "   ".join(feat_results))
    print()

# Aggregate totals per feature
print("OVERALL (mean % across all valid combos):")
for feat in FEATURES:
    sub = df[df['property'] == feat]
    pct = sub['pct_above_p95'].mean()
    print(f"  {feat:<22s}: {pct:.1f}%")

# ── section 3: neuron specialisation across 4 features ───────────────────────

def neuron_specialisation_multi(subset_df, label, features=None, min_features=2):
    """
    For each neuron, classify based on which features it encodes above null_p95.

    With 4 features, there are 2^4 = 16 possible combinations. We report:
      - The most common combinations
      - Aggregate counts: how many features each neuron encodes (0, 1, 2, 3, 4)
      - Per-feature prevalence

    A (model, dataset, layer) combo is included if at least min_features features
    have valid data (after applying EXCLUSIONS). Excluded features contribute 0 to
    per-feature prevalence for that combo.
    """
    if features is None:
        features = FEATURES

    # Counts for "how many features does this neuron encode"
    count_distribution = {i: 0 for i in range(len(features) + 1)}

    # Counts for specific combinations (using sorted tuple of feature names)
    combo_counts = {}

    # Per-feature individual prevalence (how many neurons encode this specific feature)
    feature_counts = {feat: 0 for feat in features}

    # Per-feature eligible pool: how many neurons this feature was actually tested on
    # (excludes neurons from combos where the feature was excluded/missing)
    feature_eligible = {feat: 0 for feat in features}

    total_neurons = 0

    # Group by (model, dataset, layer) — collect whichever features are available
    for (model, dataset, layer), grp in subset_df.groupby(['model', 'dataset', 'layer']):
        feat_data = {}
        for feat in features:
            if dataset in EXCLUSIONS.get(feat, []):
                continue
            rows = grp[grp['property'] == feat]
            if len(rows) == 0 or rows.iloc[0]['null_p95'] is None:
                continue
            feat_data[feat] = {
                'corrs': rows.iloc[0]['corrs'],
                'threshold': rows.iloc[0]['null_p95'],
            }

        available = list(feat_data.keys())
        if len(available) < min_features:
            continue

        min_n = min(len(feat_data[feat]['corrs']) for feat in available)
        if min_n == 0:
            continue

        sig_arrays = {
            feat: feat_data[feat]['corrs'][:min_n] > feat_data[feat]['threshold']
            for feat in available
        }

        for n_idx in range(min_n):
            encoded_features = tuple(sorted(
                feat for feat in available if sig_arrays[feat][n_idx]
            ))
            n_encoded = len(encoded_features)
            count_distribution[n_encoded] += 1

            combo_counts[encoded_features] = combo_counts.get(encoded_features, 0) + 1

            for feat in available:
                feature_eligible[feat] += 1
            for feat in encoded_features:
                feature_counts[feat] += 1

            total_neurons += 1

    if total_neurons == 0:
        print(f"{label}: insufficient data")
        return

    print(f"{label}  (N={total_neurons:,} neurons)")

    # Distribution: how many features encoded per neuron
    print(f"  Features encoded per neuron:")
    for n_feat in range(len(features) + 1):
        count = count_distribution[n_feat]
        pct = 100 * count / total_neurons
        bar = '█' * int(pct / 2)
        print(f"    {n_feat} features: {count:>7,} ({pct:>5.1f}%) {bar}")

    # Per-feature prevalence (denominator = neurons this feature was actually tested on)
    print(f"  Per-feature prevalence (neurons encoding this feature):")
    for feat in features:
        count = feature_counts[feat]
        eligible = feature_eligible[feat]
        pct = 100 * count / eligible if eligible else float('nan')
        print(f"    {feat:<22s}: {count:>7,} / {eligible:>7,} eligible ({pct:>5.1f}%)")

    # Top combinations
    print(f"  Most common encoding patterns:")
    sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])
    for combo, count in sorted_combos[:8]:  # top 8
        pct = 100 * count / total_neurons
        if len(combo) == 0:
            label_str = 'none'
        elif len(combo) == len(features):
            label_str = 'all features'
        else:
            label_str = '+'.join(c[:6] for c in combo)
        print(f"    {label_str:<35s}: {count:>7,} ({pct:>5.1f}%)")
    print()

print("\n" + "="*80)
print("NEURON SPECIALISATION (across 4 features)")
print("="*80)
print("Threshold: null_p95_r per combo (chance gives 5% above by definition)")
print(f"Features analyzed: {FEATURES}")
print(f"Note: layers with fewer than 3 features available (after EXCLUSIONS) are skipped\n")

print("OVERALL")
neuron_specialisation_multi(df, "OVERALL")

print("\nBY GROUP:")
for group in GROUPS:
    sub = df[df['groups'].apply(lambda g: group in g)]
    neuron_specialisation_multi(sub, f"  {group}")