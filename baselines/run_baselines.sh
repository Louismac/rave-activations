#!/bin/bash
set -e
cd "$(dirname "$0")"

# Optional: pass a comma-separated feature list to only refresh those
# properties (forces recompute, overwriting existing entries for just those
# properties; everything else in the output is left untouched). Omit for a
# normal full/resume run.
#
#   ./run_baselines.sh                                    # full run
#   ./run_baselines.sh spectral_centroid,spectral_bandwidth  # spectral-only refresh
#
# UPDATE_FEATURES can also be set directly in the environment instead of as
# an argument (e.g. from another script that already exports it).
if [ -n "$1" ]; then
    export UPDATE_FEATURES="$1"
fi
if [ -n "$UPDATE_FEATURES" ]; then
    echo "Updating only: $UPDATE_FEATURES (all other properties left as-is)"
fi

echo "========================================"
echo "Stage 1: Spearman permutation baseline"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u run_permutation_baseline.py

echo ""
echo "========================================"
echo "Stage 2: Nonlinear-probe baseline"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u run_permutation_baseline_nonlinear.py

echo ""
echo "========================================"
echo "Stage 3: Spearman baseline (clusters)"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u run_permutation_baseline_clusters.py

echo ""
echo "========================================"
echo "Stage 4: Nonlinear-probe baseline (clusters)"
echo "========================================"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u run_permutation_baseline_nonlinear_clusters.py

echo ""
echo "✓ All baselines complete."
