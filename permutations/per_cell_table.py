"""
Per-cell summary table: one row per (model, dataset, feature).

Point estimates (null_p95_r, obs_max, % > p95, null_p95_r2, NL gain) come
from the pre-aggregated permutation_baseline_table.csv /
permutation_baseline_nonlinear_table.csv. Obs. mean, Linear (r2) and
Nonlinear (r2) are recomputed here with a bootstrap CI across the cell's
layers (same method as confidence.py: resample the per-layer values).

Output: per_cell_table.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from confidence_intervals import bootstrap_ci
from confidence import UPLOAD_DIR, N_BOOT, SEED, CELL_SUMMARY, BOOT_METHOD

OUT_DIR = Path(".")


def bootstrap_cell(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    stat = np.max if CELL_SUMMARY == "max" else np.mean
    return bootstrap_ci(vals, statistic=stat, n_boot=N_BOOT, method=BOOT_METHOD, seed=SEED)


def build_per_cell_table():
    """One row per (model, dataset, feature) with point estimates + per-layer
    bootstrap CIs for Obs. mean, Linear (r2), Nonlinear (r2)."""
    linear = pd.read_csv(UPLOAD_DIR / "permutation_baseline_table.csv")
    nonlinear = pd.read_csv(UPLOAD_DIR / "permutation_baseline_nonlinear_table.csv")
    agg = pd.merge(linear, nonlinear, on=["model", "dataset", "feature"], how="inner")

    r_layers = pd.read_csv(UPLOAD_DIR / "plot_obs_r_per_layer.csv")
    r2_layers = pd.read_csv(UPLOAD_DIR / "plot_obs_r2_per_layer.csv")

    rows = []
    for _, cell in agg.iterrows():
        model, dataset, feature = cell["model"], cell["dataset"], cell["feature"]

        r_cell = r_layers[(r_layers["model"] == model) & (r_layers["dataset"] == dataset)
                           & (r_layers["feature"] == feature)]
        r2_cell = r2_layers[(r2_layers["model"] == model) & (r2_layers["dataset"] == dataset)
                             & (r2_layers["feature"] == feature)]

        obs_mean, obs_mean_lo, obs_mean_hi = bootstrap_cell(r_cell["obs_mean_r"].values)
        lin_r2, lin_r2_lo, lin_r2_hi = bootstrap_cell(r2_cell["linear_observed_r2"].values)
        nl_r2, nl_r2_lo, nl_r2_hi = bootstrap_cell(r2_cell["observed_r2"].values)

        rows.append({
            "model":           model,
            "dataset":         dataset,
            "feature":         feature,
            "null_p95_r":      cell["null_p95_r"],
            "obs_mean":        obs_mean,
            "obs_mean_lo":     obs_mean_lo,
            "obs_mean_hi":     obs_mean_hi,
            "obs_max":         cell["observed_max_r"],
            "pct_exceeding":   cell["observed_pct_exceeding"],
            "null_p95_r2":     cell["mean_null_p95_r2"],
            "linear_r2":       lin_r2,
            "linear_r2_lo":    lin_r2_lo,
            "linear_r2_hi":    lin_r2_hi,
            "nonlinear_r2":    nl_r2,
            "nonlinear_r2_lo": nl_r2_lo,
            "nonlinear_r2_hi": nl_r2_hi,
            "nl_gain":         cell["mean_nonlinear_gain"],
        })

    return pd.DataFrame(rows)


def main():
    result = build_per_cell_table()
    result.to_csv(OUT_DIR / "per_cell_table.csv", index=False)

    def ci(v, lo, hi):
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    display = pd.DataFrame({
        "Model":             result["model"],
        "Dataset":           result["dataset"],
        "Feature":           result["feature"],
        "Null p95 (r)":      result["null_p95_r"].map("{:.3f}".format),
        "Obs. mean":         [ci(v, lo, hi) for v, lo, hi in
                               zip(result["obs_mean"], result["obs_mean_lo"], result["obs_mean_hi"])],
        "Obs. max":          result["obs_max"].map("{:.3f}".format),
        "% > p95":           result["pct_exceeding"].map("{:.1f}".format),
        "Null p95 (r2)":     result["null_p95_r2"].map("{:.3f}".format),
        "Linear (r2)":       [ci(v, lo, hi) for v, lo, hi in
                               zip(result["linear_r2"], result["linear_r2_lo"], result["linear_r2_hi"])],
        "Nonlinear (r2)":    [ci(v, lo, hi) for v, lo, hi in
                               zip(result["nonlinear_r2"], result["nonlinear_r2_lo"], result["nonlinear_r2_hi"])],
        "NL gain":           result["nl_gain"].map("{:.3f}".format),
    })

    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
