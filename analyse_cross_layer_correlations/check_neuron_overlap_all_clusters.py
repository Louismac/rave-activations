"""
Neuron overlap analysis across ALL k values, all model/dataset combos, both features.

For each (model, dataset, feature, section):
  - Best cluster identity and size per k
  - Pairwise Jaccard of best cluster across k values (min/mean shown for brevity)
  - Consecutive-k retention %: what fraction of best-cluster neurons survive to k+1
  - Stable core size (present in best cluster for all k)
  - Late-section cluster size distribution across k
"""
import json
from pathlib import Path

base_path = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results")

MODELS   = ['drum_loops', 'strings', 'taylor_vocal']
DATASETS = ['drum_loops', 'stimuli', 'strings', 'vocals']
FEATURES = ['bpm', 'pitch']
SECTIONS = ['early', 'middle', 'late']
K_RANGE  = range(4, 11)


def load_data(model, dataset):
    data = {}
    for k in K_RANGE:
        f = base_path / f"{k}_cluster" / model / dataset / 'cross_layer_cluster_correlation.json'
        if f.exists():
            with open(f) as fh:
                data[k] = json.load(fh)
    return data


def best_cluster_info(data, k, section, feature):
    """Return (cluster_key, corr, frozenset_of_neurons) for best cluster."""
    best_corr, best_key, best_neurons = -1, None, frozenset()
    for ck, info in data[k].get(section, {}).items():
        corr = info.get('properties', {}).get(feature, {}).get('mean_correlation', 0)
        if corr > best_corr:
            best_corr = corr
            best_key = ck
            best_neurons = frozenset(tuple(n) for n in info.get('neuron_origins', []))
    return best_key, best_corr, best_neurons


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ── main loop ─────────────────────────────────────────────────────────────────
for model in MODELS:
    for dataset in DATASETS:
        all_data = load_data(model, dataset)
        if not all_data:
            continue
        ks = sorted(all_data.keys())

        print(f"\n{'='*70}")
        print(f"  model={model}  dataset={dataset}")
        print(f"{'='*70}")

        for feature in FEATURES:
            print(f"\n  ── feature: {feature.upper()} ──")
            print(f"  {'section':<8} {'k':>3}  {'cluster':<12} {'corr':>6}  {'n':>5}  "
                  f"{'min_J_across_k':>14}  {'consecutive retention %':>25}")

            for section in SECTIONS:
                # collect best cluster info per k
                bests = {}
                for k in ks:
                    ck, corr, neurons = best_cluster_info(all_data, k, section, feature)
                    if ck is not None:
                        bests[k] = (ck, corr, neurons)

                if not bests:
                    continue

                bks = sorted(bests.keys())

                # pairwise Jaccard — just min and mean for summary
                js = []
                for i, ki in enumerate(bks):
                    for kj in bks[i+1:]:
                        js.append(jaccard(bests[ki][2], bests[kj][2]))
                min_j = min(js) if js else 1.0
                mean_j = sum(js) / len(js) if js else 1.0

                # consecutive retention %
                retentions = []
                for i in range(len(bks) - 1):
                    ki, kj = bks[i], bks[i+1]
                    ni, nj = bests[ki][2], bests[kj][2]
                    shared = ni & nj
                    pct = 100 * len(shared) / len(ni) if ni else 0
                    retentions.append(f"k{ki}→{kj}:{pct:.0f}%")

                # stable core
                core = set.intersection(*(set(bests[k][2]) for k in bks))

                # print one summary row per (section, feature)
                k0 = bks[0]
                ck0, corr0, n0 = bests[k0]
                ret_str = "  ".join(retentions)
                print(f"  {section:<8} {k0:>3}  {ck0:<12} {corr0:>6.4f}  {len(n0):>5}  "
                      f"minJ={min_j:.3f} meanJ={mean_j:.3f}  {ret_str}")
                print(f"  {'':8}      stable core={len(core)} neurons  "
                      f"(corr stable: {all(bests[k][1] == corr0 for k in bks)})")

        # ── cluster size distribution (late section) ───────────────────────
        print(f"\n  ── late-section cluster sizes across k ──")
        print(f"  {'k':>3}  sizes (sorted desc)")
        for k in ks:
            sizes = sorted(
                [len(info.get('neuron_origins', []))
                 for info in all_data[k].get('late', {}).values()],
                reverse=True
            )
            print(f"  k={k}  {sizes}")
