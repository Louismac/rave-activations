"""
Run do_permutation_baseline over all model × dataset combos.

Mirrors the loop in run_cluster_analysis but only runs activation collection
and the permutation test — does not redo clustering or correlation.

Results saved to:
  results/{n_clusters}_cluster/{model}/{dataset}/permutation_baseline.json
"""
import sys
from pathlib import Path

# reuse helpers from analyse.py
sys.path.insert(0, str(Path(__file__).parent))
import analyse as _analyse

home = Path("/home/louis/Documents/notebooks/rave-activations/RAVE-activations-2025/")
_analyse.home = home   # load_datasets() uses the module-level home variable

from analyse import get_analyser, load_datasets, convert_channels
models      = ["strings", "drum_loops", "taylor_vocal"]
n_clusters  = 6        # must match the results folder you want to populate
device      = "cuda"
N_PERMS     = 500     # reduce to 100 for a quick sanity check
THRESHOLD   = 0.15

# ── load datasets once ────────────────────────────────────────────────────────
datasets = load_datasets()
datasets_dict = {
    'strings':    (datasets[0], datasets[1]),
    'drum_loops': (datasets[2], datasets[3]),
    'stimuli':    (datasets[4], datasets[5]),
    'vocals':     (datasets[6], datasets[7]),
}

# ── loop ──────────────────────────────────────────────────────────────────────
for model_name in models:
    model_path  = home / "runs" / model_name / "best.ckpt"
    config_path = home / "runs" / model_name / "config.gin"
    output_base = home / "results" / f"{n_clusters}_cluster" / model_name

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    for dataset_name, (audio_list, metadata_list) in datasets_dict.items():
        output_dir = output_base / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        out_file = output_dir / "permutation_baseline.json"
        if out_file.exists():
            # Regenerate if missing observed_pct_exceeding_p95 (added in recalculation)
            import json
            with open(out_file) as _f:
                _existing = json.load(_f)
            if all('observed_pct_exceeding_p95' in v for v in _existing.values()):
                print(f"  [{dataset_name}] already up to date, skipping.")
                continue
            print(f"  [{dataset_name}] missing observed_pct_exceeding_p95, regenerating.")
            out_file.unlink()

        print(f"\n  --- {dataset_name} ---")
        analyser = get_analyser(model_path=model_path,
                                config_path=config_path,
                                device=device)

        audio = convert_channels(audio_list, analyser.model.n_channels)
        analyser.activate(audio, metadata_list)
        #there is no pitch info in the drum dataset
        prop = ['bpm'] if dataset_name == 'drum_loops' else ['pitch', 'bpm']
        analyser.do_permutation_baseline(
            output_dir=output_dir,
            prop=prop,
            n_permutations=N_PERMS,
            threshold=THRESHOLD,
        )

print("\n✓ All permutation baselines complete.")
