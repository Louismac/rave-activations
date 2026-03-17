"""
common_layers_v2.py — find top layers using % neurons above null p95

Replaces mean_correlation ranking with the fraction of neurons whose
|r| exceeds the permutation-derived null p95 threshold (per combo).
By construction the null gives 5%, so higher values = real signal.

Exclusions: drum_loops dataset from pitch; vocals dataset from BPM.
"""

import json
import numpy as np
from pathlib import Path
import pathlib
base_path = pathlib.Path(__file__).parent.parent.resolve()
base_path = base_path / "results" /  "6_cluster"
models   = ['strings', 'drum_loops', 'taylor_vocal']
datasets = ['strings', 'drum_loops', 'vocals', 'stimuli']

# ── load per-layer pct_above_p95 ─────────────────────────────────────────────

all_data = {}
for model in models:
    all_data[model] = {}
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
                if prop in perm and 'null_p95_r' in perm[prop]:
                    null_p95[prop] = perm[prop]['null_p95_r']

        layer_stats = {}   # layer -> {prop -> pct_above_p95}
        for layer_name, layer_data in var.items():
            if layer_name.startswith('section_'):
                continue
            layer_stats[layer_name] = {}
            for prop in ['pitch', 'bpm']:
                if prop not in layer_data or prop not in null_p95:
                    continue
                corrs = np.array(layer_data[prop].get('all_correlations', []))
                if len(corrs) == 0:
                    continue
                layer_stats[layer_name][prop] = float((corrs > null_p95[prop]).mean() * 100)

        all_data[model][dataset] = layer_stats

print(f"Loaded data for {len(models)} models × {len(datasets)} datasets\n")


def get_top_layers(model, dataset, prop, top_n=5):
    """Return top_n layer names ranked by % neurons above null p95."""
    layer_stats = all_data.get(model, {}).get(dataset, {})
    scored = [
        (layer, stats[prop])
        for layer, stats in layer_stats.items()
        if prop in stats
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {layer for layer, _ in scored[:top_n]}, scored


def find_common_layers(layer_sets):
    if not layer_sets:
        return set()
    return set.intersection(*layer_sets)


# ── analysis ──────────────────────────────────────────────────────────────────

print("=" * 80)
print("TOP LAYERS BY % NEURONS ABOVE NULL P95")
print("=" * 80)
print(f"(null gives 5% by construction; top_n=5 layers per model)\n")

TOP_N = 5

for dataset in datasets:
    print(f"\n{'─'*60}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"{'─'*60}")

    props = []
    if dataset != 'drum_loops':
        props.append('pitch')
    if dataset != 'vocals':
        props.append('bpm')

    for prop in props:
        print(f"\n  {prop.upper()} — top {TOP_N} layers per model:")
        model_top_sets = []
        per_model_ranked = {}
        all_scores = {}   # layer -> {model -> pct}

        for model in models:
            if dataset not in all_data.get(model, {}):
                continue
            top_set, ranked = get_top_layers(model, dataset, prop, TOP_N)
            _, full_ranked = get_top_layers(model, dataset, prop, top_n=999)
            model_top_sets.append(top_set)
            per_model_ranked[model] = ranked[:TOP_N]
            for layer, pct in full_ranked:
                all_scores.setdefault(layer, {})[model] = pct

            scores_str = '  '.join(f"{l.split('.')[-1]}={s:.1f}%" for l, s in ranked[:TOP_N])
            print(f"    {model:<14s}: {scores_str}")

        def mean_pct_for_layers(layers):
            """Mean % across all models for a set of layers."""
            rows = []
            for layer in sorted(layers):
                model_pcts = [all_scores[layer][m] for m in models if m in all_scores.get(layer, {})]
                mean_pct = np.mean(model_pcts) if model_pcts else float('nan')
                rows.append((layer, mean_pct, model_pcts))
            return rows

        common = find_common_layers(model_top_sets)
        if common:
            print(f"  → Common to ALL models:")
            for layer, mean_pct, model_pcts in mean_pct_for_layers(common):
                per_model = '  '.join(f"{m[:6]}={p:.1f}%" for m, p in zip(models, model_pcts))
                print(f"       {layer:<45s}  mean={mean_pct:.1f}%  [{per_model}]")
        else:
            pairs = []
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    s1 = {l for l, _ in per_model_ranked.get(m1, [])}
                    s2 = {l for l, _ in per_model_ranked.get(m2, [])}
                    shared = s1 & s2
                    if shared:
                        pairs.append((m1, m2, shared))
            if pairs:
                print(f"  → No universal common layers. Pairwise overlaps:")
                for m1, m2, shared in pairs:
                    for layer, mean_pct, model_pcts in mean_pct_for_layers(shared):
                        print(f"       {m1}∩{m2}  {layer:<40s}  mean={mean_pct:.1f}%")
            else:
                print(f"  → No common layers across any pair of models")
