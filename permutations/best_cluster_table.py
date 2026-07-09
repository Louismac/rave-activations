"""
Extract three "best" clusters per (model, dataset, feature, section) cell:
  - Best cluster by per-neuron mean |rho|
  - Best cluster by % > p95
  - Best cluster by joint nonlinear R^2

Loads from permutation_baseline_clusters_table.csv (per-neuron stats)
and permutation_baseline_nonlinear_clusters_table.csv (joint R^2 stats),
merges on cluster identity, then picks the best cluster per metric per cell
— same groupby-and-take-best approach as load_best_per_cell() in
analyse-cross-layer-correlations/cluster_layer_comparison.py, except the
section is part of the cell key here, so the best cluster is chosen
separately within each section rather than across all sections.

meanr_by_meanr gets a bootstrap CI computed over the cluster's individual
neuron |rho| values (cross_layer_cluster_correlation_all_neurons.json's
"all_correlations" -- mean matches observed_mean_r exactly). r2_by_r2 has
no such per-neuron decomposition (it's a single joint regression fit, not a
per-neuron statistic) and the underlying 5-fold CV scores were never
persisted, so it gets no CI.

Writes one wide CSV row per (model, dataset, feature, section) cell showing
all three best clusters side-by-side.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from confidence_intervals import bootstrap_ci
from confidence import N_BOOT, SEED, BOOT_METHOD


HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results_500" / "6_cluster"
GROUP_COLS = ["model", "dataset", "section", "cluster", "feature"]
CELL_KEYS  = ["model", "dataset", "feature", "section"]


def load_neuron_correlations(model_dataset_pairs):
    """(model, dataset, section, cluster, feature) -> raw per-neuron |rho| values."""
    lookup = {}
    for model, dataset in model_dataset_pairs:
        path = RESULTS_DIR / model / dataset / "cross_layer_cluster_correlation_all_neurons.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for section, clusters in data.items():
            for cluster_key, cluster_data in clusters.items():
                cluster_id = int(cluster_key.split("_")[1])
                for feature, props in cluster_data.get("properties", {}).items():
                    corrs = props.get("all_correlations")
                    if corrs:
                        lookup[(model, dataset, section, cluster_id, feature)] = corrs
    return lookup


def bootstrap_neuron_mean(corrs):
    vals = np.asarray(corrs, dtype=float)
    if len(vals) < 2:
        v = float(vals[0]) if len(vals) else np.nan
        return v, v
    _, lo, hi = bootstrap_ci(vals, statistic=np.mean,
                              n_boot=N_BOOT, method=BOOT_METHOD, seed=SEED)
    return lo, hi

# Ignore clusters smaller than this fraction of their section's total neuron
# count when picking the "best" cluster — avoids tiny clusters winning by
# chance (e.g. a 2-neuron cluster hitting 100% pct-exceeding).
MIN_CLUSTER_PCT = 0.01
CLUSTER_SIZE_COL = "n_neurons"


def best_by_metric(df, metric_col):
    """For each cell, return the cluster row maximising metric_col."""
    return (df.sort_values(metric_col, ascending=False)
              .groupby(CELL_KEYS, as_index=False)
              .first())


def apply_min_cluster_pct(clusters, min_cluster_pct, cluster_size_col=CLUSTER_SIZE_COL,
                           section_keys=CELL_KEYS):
    """Drop clusters smaller than min_cluster_pct of their section's total size."""
    if min_cluster_pct is None:
        return clusters

    if cluster_size_col not in clusters.columns:
        print(f"  WARNING: column '{cluster_size_col}' not found; "
              f"skipping min-size filter")
        return clusters

    totals = (clusters
              .groupby(section_keys)[cluster_size_col]
              .sum()
              .reset_index()
              .rename(columns={cluster_size_col: "_section_total"}))
    clusters = clusters.merge(totals, on=section_keys, how="left")
    clusters["_cluster_pct"] = (
        clusters[cluster_size_col] / clusters["_section_total"]
    )

    n_before = len(clusters)
    clusters = clusters[clusters["_cluster_pct"] >= min_cluster_pct].copy()
    n_after = len(clusters)
    print(f"  Cluster size filter (>= {min_cluster_pct:.1%} of section total): "
          f"{n_before} → {n_after} clusters retained")

    return clusters.drop(columns=["_section_total", "_cluster_pct"])


def main():
    linear    = pd.read_csv(HERE / "permutation_baseline_clusters_table.csv")
    nonlinear = pd.read_csv(HERE / "permutation_baseline_nonlinear_clusters_table.csv")

    df = pd.merge(linear, nonlinear, on=GROUP_COLS, how="inner")

    print(f"Merged rows: {len(df)}")
    print(f"Unique cells: {df[CELL_KEYS].drop_duplicates().shape[0]}")

    df = apply_min_cluster_pct(df, MIN_CLUSTER_PCT)

    # Best cluster per cell by each metric
    best_meanr = best_by_metric(df, "observed_mean_r")
    best_pct   = best_by_metric(df, "observed_pct_exceeding_p95")
    best_r2    = best_by_metric(df, "observed_r2")

    # Neuron-level bootstrap CI for meanr_by_meanr (resamples the cluster's
    # individual neuron |rho| values, not folds/layers -- r2_by_r2 has no
    # such per-neuron decomposition, so it gets no CI).
    corr_lookup = load_neuron_correlations(
        best_meanr[["model", "dataset"]].drop_duplicates().itertuples(index=False, name=None))
    ci_bounds = best_meanr.apply(
        lambda r: bootstrap_neuron_mean(
            corr_lookup.get((r["model"], r["dataset"], r["section"], r["cluster"], r["feature"]), [])),
        axis=1, result_type="expand")
    best_meanr = best_meanr.assign(meanr_ci_lo=ci_bounds[0], meanr_ci_hi=ci_bounds[1])

    # Rename per-metric columns and merge into one wide row per cell
    # (section is already part of CELL_KEYS, so only cluster varies per metric)
    cols_meanr = CELL_KEYS + ["cluster", "n_neurons",
                              "observed_mean_r", "observed_max_r",
                              "meanr_ci_lo", "meanr_ci_hi"]
    cols_pct   = CELL_KEYS + ["cluster", "n_neurons",
                              "observed_pct_exceeding_p95"]
    cols_r2    = CELL_KEYS + ["cluster", "n_channels",
                              "linear_observed_r2", "observed_r2",
                              "nonlinear_gain"]

    meanr_w = (best_meanr[cols_meanr]
               .rename(columns={
                   "cluster":         "cluster_by_meanr",
                   "n_neurons":       "n_by_meanr",
                   "observed_mean_r": "meanr_by_meanr",
                   "observed_max_r":  "maxr_by_meanr",
                   "meanr_ci_lo":     "meanr_by_meanr_lo",
                   "meanr_ci_hi":     "meanr_by_meanr_hi",
               }))

    pct_w = (best_pct[cols_pct]
             .rename(columns={
                 "cluster":                    "cluster_by_pct",
                 "n_neurons":                  "n_by_pct",
                 "observed_pct_exceeding_p95": "pct_by_pct",
             }))

    r2_w = (best_r2[cols_r2]
            .rename(columns={
                "cluster":            "cluster_by_r2",
                "n_channels":         "n_by_r2",
                "linear_observed_r2": "linear_r2_by_r2",
                "observed_r2":        "r2_by_r2",
                "nonlinear_gain":     "nl_gain_by_r2",
            }))

    wide = (meanr_w
            .merge(pct_w, on=CELL_KEYS)
            .merge(r2_w,  on=CELL_KEYS))

    # Flag whether the same physical cluster is best across all three metrics
    # (section is fixed by the groupby, so only cluster identity can differ)
    wide["same_meanr_r2"] = wide["cluster_by_meanr"] == wide["cluster_by_r2"]
    wide["same_pct_r2"]   = wide["cluster_by_pct"]   == wide["cluster_by_r2"]
    wide["same_all_three"] = wide["same_meanr_r2"] & wide["same_pct_r2"]

    # Sort for readability
    model_order   = {"strings": 0, "drum_loops": 1, "taylor_vocal": 2}
    dataset_order = {"strings": 0, "drum_loops": 1, "vocals": 2, "stimuli": 3}
    feature_order = {"pitch": 0, "bpm": 1,
                     "spectral_centroid": 2, "spectral_bandwidth": 3}
    section_order = {"early": 0, "middle": 1, "late": 2}

    wide["_m"] = wide["model"].map(model_order)
    wide["_d"] = wide["dataset"].map(dataset_order)
    wide["_f"] = wide["feature"].map(feature_order)
    wide["_s"] = wide["section"].map(section_order)
    wide = (wide.sort_values(["_m", "_d", "_f", "_s"])
                .drop(columns=["_m", "_d", "_f", "_s"])
                .reset_index(drop=True))

    # Output
    out_path = HERE / "best_clusters_per_metric.csv"
    wide.to_csv(out_path, index=False)
    print(f"\nWrote {len(wide)} rows to {out_path}")

    # Summary
    n_same = wide["same_all_three"].sum()
    n_diff_meanr_r2 = (~wide["same_meanr_r2"]).sum()
    print(f"\nCells where same cluster is best by all three metrics: "
          f"{n_same}/{len(wide)}")
    print(f"Cells where best-by-mean-|r| differs from best-by-R^2: "
          f"{n_diff_meanr_r2}/{len(wide)}")

    # Preview
    print("\nFirst few rows:")
    preview_cols = (CELL_KEYS +
                    ["cluster_by_meanr", "n_by_meanr", "meanr_by_meanr",
                     "meanr_by_meanr_lo", "meanr_by_meanr_hi",
                     "cluster_by_pct",   "n_by_pct",   "pct_by_pct",
                     "cluster_by_r2",    "n_by_r2",    "r2_by_r2",
                     "same_all_three"])
    print(wide[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()