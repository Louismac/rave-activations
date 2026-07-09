"""
Per-cell and per-aggregate confidence intervals for each (feature x measure).

For every measure (mean |rho|, % > p95, joint R^2) and every feature:

  PER-CELL CIs
    One row per (model, dataset) cell. The cell value is the mean across the
    cell's 28 layers; the 95% CI is a bootstrap over those 28 layer values.
    (Set CELL_SUMMARY = "max" to use the best layer instead of the mean.)

  PER-AGGREGATE CIs
    One row per condition in {natural, synthetic, ID, OOD}. The value is the
    mean of the per-cell values in that condition; the 95% CI is a bootstrap
    that RESAMPLES CELLS (so the interval reflects between-cell variability,
    consistent with treating the cell as the unit of independent observation).

Outputs:
  - console tables
  - per_cell_cis.csv, per_aggregate_cis.csv
  - LaTeX: table_per_cell_cis.tex, table_aggregate_cis.tex
"""

from pathlib import Path
import numpy as np
import pandas as pd

from confidence_intervals import bootstrap_ci

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(__file__).parent  # adjust if needed

N_BOOT = 10000
SEED   = 0
CELL_SUMMARY = "mean"        # "mean" (across layers) or "max" (best layer)
BOOT_METHOD  = "bca"         # "bca" or "percentile"

STIMULUS_TYPE = {
    "strings": "natural", "vocals": "natural",
    "drum_loops": "natural", "stimuli": "synthetic",
}
ID_DATASET = {
    "strings": "strings", "drum_loops": "drum_loops", "taylor_vocal": "vocals",
}

PROPERTIES = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]

EXCLUSIONS = {
    "pitch":              ["drum_loops"],
    "spectral_centroid":  ["stimuli"],
    "spectral_bandwidth": ["stimuli"],
}

PROP_LABELS = {
    "pitch": "Pitch", "bpm": "BPM",
    "spectral_centroid": "Spec.\\ Centroid",
    "spectral_bandwidth": "Spec.\\ Bandwidth",
}


MEASURES = {
    "mean_r": {
        "label":     "Per-neuron mean $|\\rho|$",
        "short":     "meanrho",
        "csv":       UPLOAD_DIR / "plot_obs_r_per_layer.csv",
        "value_col": "obs_mean_r",
        "decimals":  3,
    },
    "pct": {
        "label":     "\\% $>$ p95",
        "short":     "pct",
        "csv":       UPLOAD_DIR / "plot_pct_exceeding_per_layer.csv",
        "value_col": "obs_pct_exceeding",
        "decimals":  1,
    },
    "r2": {
        "label":     "Joint $R^2$",
        "short":     "r2",
        "csv":       UPLOAD_DIR / "plot_obs_r2_per_layer.csv",
        "value_col": "observed_r2",
        "decimals":  3,
    },
}


def classify(model, dataset):
    stype = STIMULUS_TYPE.get(dataset, "unknown")
    if stype == "synthetic":
        return "synthetic", "synthetic"
    dist = "ID" if ID_DATASET.get(model) == dataset else "OOD"
    return "natural", dist


def summarise_cell(values):
    if CELL_SUMMARY == "max":
        return float(np.max(values))
    return float(np.mean(values))


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell CIs
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_cell(measure_key):
    info = MEASURES[measure_key]
    df = pd.read_csv(info["csv"])
    vcol = info["value_col"]

    rows = []
    for feat in PROPERTIES:
        sub = df[df["feature"] == feat]
        sub = sub[~sub["dataset"].isin(EXCLUSIONS.get(feat, []))]
        for (model, dataset), cell in sub.groupby(["model", "dataset"]):
            vals = cell[vcol].dropna().values.astype(float)
            if len(vals) < 2:
                continue
            stype, dist = classify(model, dataset)
            # cell point value
            point = summarise_cell(vals)
            # CI: bootstrap the within-cell layer values using the same summary
            stat = np.max if CELL_SUMMARY == "max" else np.mean
            _, lo, hi = bootstrap_ci(vals, statistic=stat,
                                     n_boot=N_BOOT, method=BOOT_METHOD, seed=SEED)
            rows.append({
                "measure":  measure_key,
                "feature":  feat,
                "model":    model,
                "dataset":  dataset,
                "stimulus": stype,
                "dist":     dist,
                "n_layers": len(vals),
                "value":    point,
                "ci_lo":    lo,
                "ci_hi":    hi,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-aggregate CIs  (natural / synthetic / ID / OOD)
# ─────────────────────────────────────────────────────────────────────────────

def compute_aggregates(per_cell_df, measure_key):
    """
    For each feature, pool the per-cell values by condition and bootstrap
    across cells. Conditions:
        natural   = all natural cells
        synthetic = all synthetic cells
        ID        = natural cells where dataset == model's training set
        OOD       = natural cells where dataset != model's training set
    """
    rows = []
    mdf = per_cell_df[per_cell_df["measure"] == measure_key]
    for feat in PROPERTIES:
        mdf = mdf[mdf["model"] != "encodec"]
        fdf = mdf[mdf["feature"] == feat]
        if len(fdf) == 0:
            continue

        conditions = {
            "natural":   fdf[fdf["stimulus"] == "natural"]["value"].values,
            "synthetic": fdf[fdf["stimulus"] == "synthetic"]["value"].values,
            "ID":        fdf[fdf["dist"] == "ID"]["value"].values,
            "OOD":       fdf[fdf["dist"] == "OOD"]["value"].values,
        }
        for cond, vals in conditions.items():
            vals = np.asarray(vals, dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            if len(vals) == 1:
                point, lo, hi = float(vals[0]), float(vals[0]), float(vals[0])
            else:
                point, lo, hi = bootstrap_ci(vals, statistic=np.mean,
                                             n_boot=N_BOOT, method=BOOT_METHOD,
                                             seed=SEED)
            rows.append({
                "measure":   measure_key,
                "feature":   feat,
                "condition": cond,
                "n_cells":   len(vals),
                "value":     point,
                "ci_lo":     lo,
                "ci_hi":     hi,
            })
    return pd.DataFrame(rows)



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_cell = []
    all_agg  = []
    for mk in MEASURES:
        cell_df = compute_per_cell(mk)
        all_cell.append(cell_df)
        all_agg.append(compute_aggregates(cell_df, mk))

    cell_df = pd.concat(all_cell, ignore_index=True)
    agg_df  = pd.concat(all_agg, ignore_index=True)

    # Console
    print("=" * 78)
    print(f"AGGREGATE CIs  (cell summary = {CELL_SUMMARY}, bootstrap = {BOOT_METHOD})")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(agg_df.to_string(index=False))

    print("\n" + "=" * 78)
    print("PER-CELL CIs")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(cell_df.to_string(index=False))

    # CSVs
    cell_df.to_csv(UPLOAD_DIR / "per_cell_cis.csv", index=False)
    agg_df.to_csv(UPLOAD_DIR / "per_aggregate_cis.csv", index=False)
