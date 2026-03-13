"""
Check whether the best BPM cluster across k values contains the same neurons.
Focuses on taylor_vocal/stimuli/late, which is consistently the top BPM cluster.
"""
import json
from pathlib import Path

base_path = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/results")

# Configurable — change to investigate a different combo
model = 'taylor_vocal'
dataset = 'stimuli'
section = 'late'
feature = 'bpm'  # 'bpm' or 'pitch'

best_clusters = {}  # k -> {'cluster': str, 'corr': float, 'neurons': frozenset}

for k in range(4, 11):
    corr_file = base_path / f"{k}_cluster" / model / dataset / 'cross_layer_cluster_correlation.json'
    if not corr_file.exists():
        print(f"k={k}: file not found")
        continue

    with open(corr_file) as f:
        data = json.load(f)

    if section not in data:
        print(f"k={k}: section '{section}' not found")
        continue

    best_corr = -1
    best_key = None
    best_neurons = None

    for cluster_key, cluster_info in data[section].items():
        props = cluster_info.get('properties', {})
        corr = props.get(feature, {}).get('mean_correlation', 0)
        if corr > best_corr:
            best_corr = corr
            best_key = cluster_key
            best_neurons = frozenset(tuple(n) for n in cluster_info.get('neuron_origins', []))

    best_clusters[k] = {'cluster': best_key, 'corr': best_corr, 'neurons': best_neurons}
    print(f"k={k}: best={best_key}  corr={best_corr:.4f}  n_neurons={len(best_neurons)}")

# Pairwise Jaccard overlap between all k values
print(f"\n--- Neuron overlap (Jaccard) between best {feature.upper()} clusters ---")
ks = sorted(best_clusters.keys())
header = f"{'':>5}" + "".join(f"  k={k}" for k in ks)
print(header)
for ki in ks:
    row = f"k={ki:>2}"
    ni = best_clusters[ki]['neurons']
    for kj in ks:
        nj = best_clusters[kj]['neurons']
        if ni and nj:
            jaccard = len(ni & nj) / len(ni | nj)
            row += f"  {jaccard:.2f}"
        else:
            row += "   N/A"
    print(row)

# Show which neurons are shared across ALL k values
all_neuron_sets = [best_clusters[k]['neurons'] for k in ks]
shared_all = set.intersection(*[set(n) for n in all_neuron_sets])
print(f"\nNeurons shared across ALL k values: {len(shared_all)}")
for neuron in sorted(shared_all):
    print(f"  layer={neuron[0]}  index={neuron[1]}")

# Show union size vs shared size for context
union_all = set.union(*[set(n) for n in all_neuron_sets])
print(f"Total unique neurons seen across any k: {len(union_all)}")
