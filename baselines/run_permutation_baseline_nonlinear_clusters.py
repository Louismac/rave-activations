"""
Run do_permutation_baseline_nonlinear_clusters over all model × dataset combos.

Mirrors run_permutation_baseline_nonlinear.py but probes each cross-layer
neuron cluster (from do_cross_layer_clustering) rather than whole layers.
If cross_layer_clustering_results_all_neurons.json is missing for a combo,
clustering is run first with the same n_clusters / pca_components used
throughout the project.

To refresh only specific properties (e.g. after a dataset fix that only
affects spectral_centroid/spectral_bandwidth) without rerunning pitch/bpm,
set UPDATE_FEATURES to a comma-separated list:

    UPDATE_FEATURES=spectral_centroid,spectral_bandwidth python3 run_permutation_baseline_nonlinear_clusters.py

This forces those properties to be recomputed and overwritten even if already
present, while every other property already in the output is left untouched.
Leave UPDATE_FEATURES unset for the normal full/resume run (existing
properties are skipped, matching prior behavior).

Results saved to:
  results_500/{n_clusters}_cluster/{model}/{dataset}/permutation_baseline_nonlinear_clusters.json
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from get_correlations_clusters import get_analyser, load_balanced_datasets, convert_channels
from encodec_adapter import EncodecActivationAnalyser, load_encodec

home        = Path(__file__).parent.parent
models      = ["drum_loops", "strings","vocals","encodec"]
n_clusters  = 6
pca_components = 2
device      = "cuda"
N_PERMS     = 100
N_FOLDS     = 5
N_EPOCHS    = 50

# None => full/resume run (skip properties already present). A set => force
# those specific properties to be recomputed, leaving everything else as-is.
_update_env = os.environ.get("UPDATE_FEATURES")
UPDATE_FEATURES = {f.strip() for f in _update_env.split(",") if f.strip()} if _update_env else None
FORCE = UPDATE_FEATURES is not None

# ── load balanced datasets once ─────────────────────────────────────────────
datasets_dict = load_balanced_datasets(home / "cache" / "500_pitch_100_bpm_4_sc_4_sb_4")

# ── loop ──────────────────────────────────────────────────────────────────────
for model_name in models:
    model_path  = home / "runs" / model_name / "best.ckpt"
    config_path = home / "runs" / model_name / "config.gin"
    output_base = home / "results_44100" / f"{n_clusters}_cluster" / model_name

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    for dataset_name, dataset in datasets_dict.items():
        audio_list = dataset['audio_list']
        metadata_list = dataset['metadata_list']
        feature_indices = dataset['feature_indices']
        output_dir = output_base / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # prop = ["pitch_class"]
        prop = ['pitch', 'bpm', 'spectral_centroid','spectral_bandwidth']
        if dataset_name == 'stimuli':
            #inlcuding "spectral_bandwidth" here doesnt really work as its not split on pitch /
            #bpm so basically its just a cinary classifier
            prop = ['pitch', 'bpm']
        if dataset_name == 'drum_loops':
            prop = ['bpm', 'spectral_centroid','spectral_bandwidth']

        if UPDATE_FEATURES is not None:
            prop = [p for p in prop if p in UPDATE_FEATURES]
            if not prop:
                print(f"  [{dataset_name}] none of {sorted(UPDATE_FEATURES)} apply here, skipping.")
                continue

        out_file = output_dir / "permutation_baseline_nonlinear_clusters.json"
        if out_file.exists() and not FORCE:
            with open(out_file) as _f:
                _existing = json.load(_f)
            all_done = all(
                p in cluster_data
                for sec_data in _existing.values()
                for cluster_data in sec_data.values()
                for p in prop
            )
            if all_done:
                print(f"  [{dataset_name}] already complete, skipping.")
                continue
            print(f"  [{dataset_name}] incomplete, resuming...")

        print(f"\n  --- {dataset_name} ---")
        if model_name == "encodec":
            model = load_encodec("facebook/encodec_32khz")        # music-oriented
            analyser = EncodecActivationAnalyser(model, device="cuda")
        else:
            analyser = get_analyser(model_path=model_path,
                                    config_path=config_path,
                                    device=device)
        chans = 1
        if hasattr(analyser.model, "n_channels"):
            chans = analyser.model.n_channels
        audio = convert_channels(audio_list, chans)
        analyser.activate(audio, metadata_list)
        analyser.set_balanced_feature_indices(feature_indices)

        clustering_file = output_dir / "cross_layer_clustering_results_all_neurons.json"
        if not clustering_file.exists():
            print(f"  Clustering results not found — running do_cross_layer_clustering first...")
            analyser.do_cross_layer_clustering(
                output_dir=output_dir,
                n_clusters=n_clusters,
                pca_components=pca_components,
            )

        analyser.do_permutation_baseline_nonlinear_clusters(
            output_dir=output_dir,
            prop=prop,
            n_permutations=N_PERMS,
            n_folds=N_FOLDS,
            n_epochs=N_EPOCHS,
            update=True,
            force=FORCE,
        )

print("\n✓ All cluster permutation baselines complete.")
