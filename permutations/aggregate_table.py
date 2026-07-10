"""
Aggregate version of per_cell_table.py: one row per (feature, condition) for
condition in {natural, synthetic, ID, OOD}, same sections as confidence.py.

Point columns (Null p95 (r), Obs. max, % > p95, Null p95 (r2), NL gain) are
the mean of the per-cell point estimates. Obs. mean, Linear (r2) and
Nonlinear (r2) get a bootstrap CI that resamples CELLS (the per-cell point
estimates from per_cell_table.py), not layers -- consistent with how
confidence.py's compute_aggregates resamples cells rather than layers.

Output: aggregate_table.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from confidence_intervals import bootstrap_ci
from confidence import N_BOOT, SEED, BOOT_METHOD, EXCLUSIONS, PROPERTIES, classify
from per_cell_table import build_per_cell_table

OUT_DIR = Path(__file__).parent

POINT_COLS = ["null_p95_r", "obs_max", "pct_exceeding", "null_p95_r2", "nl_gain"]
CI_COLS = ["obs_mean", "linear_r2", "nonlinear_r2"]


def bootstrap_cells(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    return bootstrap_ci(vals, statistic=np.mean, n_boot=N_BOOT, method=BOOT_METHOD, seed=SEED)


def main():
    cell_df = build_per_cell_table()

    cell_df = cell_df[cell_df["model"] != "encodec"]
    
    cell_df["stimulus"], cell_df["dist"] = zip(
        *cell_df.apply(lambda r: classify(r["model"], r["dataset"]), axis=1))

    rows = []
    for feat in PROPERTIES:
        excl = EXCLUSIONS.get(feat, [])
        fdf = cell_df[(cell_df["feature"] == feat)
                      & ~cell_df["dataset"].isin(excl)]
        if len(fdf) == 0:
            continue

        conditions = {
            "natural":   fdf[fdf["stimulus"] == "natural"],
            "synthetic": fdf[fdf["stimulus"] == "synthetic"],
            "ID":        fdf[fdf["dist"] == "ID"],
            "OOD":       fdf[fdf["dist"] == "OOD"],
        }
        for cond, sub in conditions.items():
            if len(sub) == 0:
                continue
            row = {"feature": feat, "condition": cond, "n_cells": len(sub)}
            for col in POINT_COLS:
                row[col] = sub[col].mean()
            for col in CI_COLS:
                point, lo, hi = bootstrap_cells(sub[col].values)
                row[col] = point
                row[f"{col}_lo"] = lo
                row[f"{col}_hi"] = hi
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "aggregate_table.csv", index=False)

    def ci(v, lo, hi):
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    display = pd.DataFrame({
        "Feature":        result["feature"],
        "Condition":      result["condition"],
        "N":              result["n_cells"],
        "Null p95 (r)":   result["null_p95_r"].map("{:.3f}".format),
        "Obs. mean":      [ci(v, lo, hi) for v, lo, hi in
                            zip(result["obs_mean"], result["obs_mean_lo"], result["obs_mean_hi"])],
        "Obs. max":       result["obs_max"].map("{:.3f}".format),
        "% > p95":        result["pct_exceeding"].map("{:.1f}".format),
        "Null p95 (r2)":  result["null_p95_r2"].map("{:.3f}".format),
        "Linear (r2)":    [ci(v, lo, hi) for v, lo, hi in
                            zip(result["linear_r2"], result["linear_r2_lo"], result["linear_r2_hi"])],
        "Nonlinear (r2)": [ci(v, lo, hi) for v, lo, hi in
                            zip(result["nonlinear_r2"], result["nonlinear_r2_lo"], result["nonlinear_r2_hi"])],
        "NL gain":        result["nl_gain"].map("{:.3f}".format),
    })

    with pd.option_context("display.width", 240, "display.max_rows", None):
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
