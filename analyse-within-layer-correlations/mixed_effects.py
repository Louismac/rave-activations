"""
Hierarchical pattern regression analysis: quadratic curvature of depth effects.

Loads per-layer metric data from three pre-computed CSVs. For each
(metric, stimulus_type, property) combination, fits a quadratic polynomial
per cell (value ~ depth_norm + depth_norm²) and tests whether the
curvature (quadratic) coefficient differs from zero across cells — i.e.
whether the depth effect is curved rather than purely linear.

The linear-only fit and the quadratic model's linear term are retained only
as central-tendency values used to draw the reference/curve lines in the
plot; they carry no significance test.

Metrics tested:
  - obs_mean_r         (plot_obs_r_per_layer.csv)
  - obs_pct_exceeding  (plot_pct_exceeding_per_layer.csv)
  - observed_r2        (plot_obs_r2_per_layer.csv)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from bh import bh_within_families, hodges_lehmann, bootstrap_hl_ci, rank_biserial_r
from perm_test import sign_permutation_p, icc_and_neff
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ─────────────────────────────────────────────────────────────

PERM_DIR = Path(__file__).parent.parent / "permutations"

STIMULUS_GROUPS = {
    "natural":   ["strings", "vocals", "drum_loops"],
    "synthetic": ["stimuli"],
}

# ── Alternate config: pooled "all stimulus" run (no natural/synthetic split) ──
# Uncomment this block (and comment out the STIMULUS_GROUPS block above) to
# run a single pooled analysis across all datasets together, instead of
# splitting into natural vs synthetic.
# STIMULUS_GROUPS = {
#     "all": ["strings", "vocals", "drum_loops", "stimuli"],
# }

# Stypes that get full inferential treatment (BH-FDR correction, significance
# stars, permutation tests) rather than being treated as merely descriptive.
# Computed automatically: everything except "synthetic" (too few cells for
# reliable significance testing there). With the pooled config above (no
# "synthetic" key at all), this naturally becomes every stype in the run.
FULL_STATS_STYPES = set(STIMULUS_GROUPS) - {"synthetic"}

PROPERTIES = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]

# Only keep rows whose "model" column contains one of these substrings
# (case-insensitive). Set to None to keep all models, e.g. ["encodec"].

OUTPUT_DIR = Path(__file__).parent

# MODEL_FILTER = ["encodec"]
# out = OUTPUT_DIR / "hierarchical_mixed_regressions_encodec.png"
# csv_path = OUTPUT_DIR / "mixed_effects_results_encodec.csv"


MODEL_FILTER = ["strings", "drum_loops","taylor_vocal"]
out = OUTPUT_DIR / "hierarchical_mixed_regressions.png"
csv_path = OUTPUT_DIR / "mixed_effects_results.csv"

# When using the pooled "all stimulus" STIMULUS_GROUPS above, also uncomment
# these two lines so the output doesn't overwrite the nat/synthetic results:
# out = OUTPUT_DIR / "hierarchical_mixed_regressions_all.png"
# csv_path = OUTPUT_DIR / "mixed_effects_results_all.csv"


out.parent.mkdir(exist_ok=True, parents=True)

EXCLUSIONS = {
    "pitch":             ["drum_loops"],
    "spectral_centroid": ["stimuli"],
    "spectral_bandwidth": ["stimuli"],
}

PROP_LABELS = {
    "pitch":              "Pitch",
    "bpm":                "BPM",
    "spectral_centroid":  "Spec. Centroid",
    "spectral_bandwidth": "Spec. Bandwidth",
}

METRICS = {
    "mean_r": {
        "csv":    PERM_DIR / "plot_obs_r_per_layer.csv",
        "col":    "obs_mean_r",
        "label":  "Mean |ρ|",
        "ylabel": "Mean |ρ|",
    },
    "pct_exceeding": {
        "csv":    PERM_DIR / "plot_pct_exceeding_per_layer.csv",
        "col":    "obs_pct_exceeding",
        "label":  "% neurons > null p95",
        "ylabel": "% neurons > p95",
    },
    "r2": {
        "csv":    PERM_DIR / "plot_obs_r2_per_layer.csv",
        "col":    "observed_r2",
        "label":  "Nonlinear probe R²",
        "ylabel": "Observed R²",
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def layer_depth(name: str):
    m = re.search(r"(?:net|layers)\.(\d+)", name)
    return int(m.group(1)) if m else None


def load_csv_data(csv_path, value_col, stimulus_groups, properties, exclusions,
                   model_filter=None):
    """
    Read a per-layer CSV. Returns a long-format DataFrame with columns:
      stimulus_type, property, model, dataset, cell_id, depth, value

    model_filter, if given, is a list of substrings; only rows whose
    "model" value contains one of them (case-insensitive) are kept.
    """
    if not csv_path.exists():
        print(f"  WARNING: {csv_path.name} not found — skipping")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    dataset_to_stype = {}
    for stype, datasets in stimulus_groups.items():
        for d in datasets:
            dataset_to_stype[d] = stype

    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"]
        prop    = row["feature"]
        model   = row.get("model", "unknown")
        stype   = dataset_to_stype.get(dataset)

        if stype is None or prop not in properties:
            continue
        if dataset in exclusions.get(prop, []):
            continue
        if model_filter and not any(
            f.lower() in str(model).lower() for f in model_filter
        ):
            continue

        depth = layer_depth(str(row["layer"]))
        if depth is None:
            continue

        val = row.get(value_col)
        if pd.isna(val):
            continue

        rows.append({
            "stimulus_type": stype,
            "property":      prop,
            "model":         model,
            "dataset":       dataset,
            "cell_id":       f"{model}__{dataset}",
            "depth":         depth,
            "value":         float(val),
        })

    return pd.DataFrame(rows)


def normalize_depth(group):
    """Map raw depths within a cell to [0, 1]."""
    group = group.copy()
    d_min, d_max = group["depth"].min(), group["depth"].max()
    if d_max == d_min:
        group["depth_norm"] = 0.5
    else:
        group["depth_norm"] = (group["depth"] - d_min) / (d_max - d_min)
    return group

def normalize_depth_inplace(df):
    """Add depth_norm column normalized to [0, 1] within each cell_id."""
    df = df.copy()
    df["depth_norm"] = (
        df.groupby("cell_id")["depth"]
          .transform(lambda x: (x - x.min()) / (x.max() - x.min())
                     if x.max() > x.min() else 0.5)
    )
    return df

def fit_per_cell_and_aggregate(sub_df):
    """
    Fit a quadratic polynomial per cell, then test whether the curvature
    (quadratic) coefficient differs from zero across cells using a
    one-sample Wilcoxon signed-rank test — this is the sole hypothesis
    test in this analysis (does depth have a curved, not just linear,
    effect on the metric?).

    Wilcoxon is used in place of a one-sample t-test because per-cell
    coefficient distributions cannot be assumed normal at small cell counts
    (N = 6-9). Note the resolution floor: the smallest two-sided Wilcoxon
    p-value is 1/2^(N-1), i.e. 0.031 at N=6 and 0.004 at N=9, regardless of
    effect magnitude. Effect sizes (Cohen's d_z) are therefore reported
    alongside and should be the primary basis for interpretation.

    The linear-only fit and the quadratic model's linear term are kept only
    as central-tendency values (no significance testing) — they're used to
    draw the reference/curve lines in the plot, not to test hypotheses.
    """
    from scipy.stats import wilcoxon

    lin_slopes = []        # β₁ from the quadratic fit (plotting only)
    quad_slopes = []       # β₂ from the quadratic fit (the curvature test)
    lin_only_slopes = []   # slope from the linear-only fit (plotting only)
    cell_models = []       # model label per cell (for model-stratified perm test)
    dataset_models = []       # model label per cell (for model-stratified perm test)

    for cell_id, cell_df in sub_df.groupby("cell_id"):
        if len(cell_df) < 4:
            continue
        d = cell_df["depth_norm"].values
        y = cell_df["value"].values

        # Linear fit
        coef_lin = np.polyfit(d, y, 1)          # [slope, intercept]
        lin_only_slopes.append(coef_lin[0])

        # Quadratic fit
        coef_quad = np.polyfit(d, y, 2)         # [β₂, β₁, intercept]
        lin_slopes.append(coef_quad[1])
        quad_slopes.append(coef_quad[0])

        # model label for this cell (all rows in a cell share it)
        cell_models.append(str(cell_df["model"].iloc[0]))
        dataset_models.append(str(cell_df["dataset"].iloc[0]))

    n_cells = len(quad_slopes)
    if n_cells < 1:
        return None

    def wilcoxon_one_sample(values):
        """
        One-sample Wilcoxon signed-rank test against zero, plus Cohen's d_z,
        Hodges–Lehmann estimate, bootstrap 95% CI, and rank-biserial r.
        Returns (W, p, d_z, median, mean, se, n, hl, ci_lo, ci_hi, rr).
        Handles the all-same-sign / all-nonzero edge cases gracefully.
        """
        vals = np.asarray(values, dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        d_z = mean / sd if (sd is not None and sd > 0) else np.nan

        W = np.nan
        p = np.nan
        if n >= 1 and np.any(vals != 0):
            try:
                W, p = wilcoxon(vals, zero_method="wilcox",
                                alternative="two-sided")
                W = float(W)
                p = float(p)
            except ValueError:
                W, p = np.nan, np.nan

        hl = hodges_lehmann(vals)
        ci_lo, ci_hi = bootstrap_hl_ci(vals)
        rr = rank_biserial_r(vals)

        return W, p, d_z, median, mean, se, n, hl, ci_lo, ci_hi, rr

    # Linear-only slope and the quadratic model's linear term: central
    # tendency only (no test) — used purely to draw the fitted lines.
    lin_only_median = float(np.median(lin_only_slopes))
    lin_only_mean   = float(np.mean(lin_only_slopes))
    quad_lin_mean   = float(np.mean(lin_slopes))

    # Quadratic (curvature) term — the one hypothesis test here
    (W_quad, p_quad, dz_quad,
     med_quad, mean_quad, se_quad, _,
     hl_quad, ci_lo_quad, ci_hi_quad, rr_quad) = wilcoxon_one_sample(quad_slopes)

    # ── Model-stratified sign-permutation p-values + ICC for the curvature term ──
    # Block: flips a whole model's sign together (tests generalisation across
    # models; floors at 0.25 for 3 models). Within: flips each cell (matches
    # naive Wilcoxon when between-model variance is low).
    models_arr = np.asarray(cell_models)
    datasets_arr = np.asarray(dataset_models)
    n_models = len(set(cell_models))

    quad_term_p_block_model,   _ = sign_permutation_p(quad_slopes, models_arr,   null="block")
    quad_term_p_block_dataset, _ = sign_permutation_p(quad_slopes, datasets_arr, null="block")
    quad_term_p_within,        _ = sign_permutation_p(quad_slopes, models_arr,   null="within")
    quad_term_ic_model   = icc_and_neff(quad_slopes, models_arr)
    quad_term_ic_dataset = icc_and_neff(quad_slopes, datasets_arr)

    def _iqr(arr):
        a = np.asarray(arr, dtype=float)
        return (float(np.percentile(a, 25)), float(np.percentile(a, 75))) if len(a) else (np.nan, np.nan)

    return {
        "n_cells": n_cells,
        "n_models": n_models,

        # Linear-only model slope (plot reference line only — no test)
        "lin_only_slope_mean":   lin_only_mean,
        "lin_only_slope_median": lin_only_median,

        # Linear term from quadratic model (plotting parameter only — no test)
        "quad_lin_slope_mean":   quad_lin_mean,

        # Quadratic (curvature) term
        "quad_term_mean":        mean_quad,
        "quad_term_median":      med_quad,
        "quad_term_se":          se_quad,
        "quad_term_q1":          _iqr(quad_slopes)[0],
        "quad_term_q3":          _iqr(quad_slopes)[1],
        "quad_term_W":           W_quad,
        "quad_term_p":           p_quad,
        "quad_term_dz":          dz_quad,
        "quad_term_hl":          hl_quad,
        "quad_term_ci_lo":       ci_lo_quad,
        "quad_term_ci_hi":       ci_hi_quad,
        "quad_term_rr":          rr_quad,
        "quad_term_p_block_model":   quad_term_p_block_model,
        "quad_term_p_block_dataset": quad_term_p_block_dataset,
        "quad_term_p_within":        quad_term_p_within,
        "quad_term_icc_model":       quad_term_ic_model["icc"],
        "quad_term_icc_dataset": quad_term_ic_dataset["icc"],
        "quad_term_n_eff":       quad_term_ic_model["n_eff"],
    }


def sig_stars(p):
    if p is None or np.isnan(p): return "?"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return " "


def plot_all(all_data, all_results, metrics, stimulus_groups, properties,
             full_stats_stypes=None):
    """
    Grid: flat list of (metric, stype) panels, laid out 2 columns wide.
    A metric is only paired with a stype if that stype has full inferential
    stats (full_stats_stypes), OR the metric is "mean_r" — i.e. every metric
    is shown for the full-stats stypes (e.g. "natural"), while descriptive-only
    stypes (e.g. "synthetic") only get the mean_r panel. With the default
    3-metric / 2-stype config this yields exactly 4 panels (2x2):
      mean_r-natural, mean_r-synthetic, pct_exceeding-natural, r2-natural.

    Each panel shows:
      - Per-depth mean ± SEM across cells (markers)
      - Aggregated linear fit (solid line, anchored at data centroid)
      - Aggregated quadratic fit (dashed line, anchored at data centroid)

    Slopes are means across per-cell OLS fits; intercepts are computed to
    pass through the data centroid (so lines visually match the data).
    """
    if full_stats_stypes is None:
        full_stats_stypes = set(stimulus_groups) - {"synthetic"}

    panels = [
        (metric_key, stype)
        for metric_key in metrics
        for stype in stimulus_groups
        if stype in full_stats_stypes or metric_key == "mean_r"
    ]

    n_cols = 2
    n_rows = -(-len(panels) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows),
                             squeeze=False)
    flat_axes = axes.flat

    colors    = cm.tab10(np.linspace(0, 1, len(properties)))
    color_map = dict(zip(properties, colors))

    for panel_idx, (metric_key, stype) in enumerate(panels):
        meta = metrics[metric_key]
        print(metric_key, stype)
        ax = flat_axes[panel_idx]
        for prop in properties:
            key = (metric_key, stype, prop)
            df_sub = all_data.get(key)
            results = all_results.get(key)
            
            if df_sub is None or len(df_sub) == 0:
                continue
            if results is None:
                continue

            # Aggregate per depth (across cells) for scatter
            depth_agg = df_sub.groupby("depth")["value"].agg(["mean", "sem"]).reset_index()

            plabel = PROP_LABELS.get(prop, prop)
            color  = color_map[prop]

            # Scatter: per-depth mean ± SEM
            ax.errorbar(
                depth_agg["depth"], depth_agg["mean"], yerr=depth_agg["sem"],
                fmt="o", color=color,
                alpha=0.55, markersize=5, capsize=2, zorder=3,
            )

            # X range in raw and normalized depth space
            d_min, d_max = df_sub["depth"].min(), df_sub["depth"].max()
            d_range = np.linspace(d_min, d_max, 50)
            d_norm  = (d_range - d_min) / (d_max - d_min + 1e-9)

            # Data centroids for anchoring lines
            observed_mean      = df_sub["value"].mean()
            depth_norm_mean    = df_sub["depth_norm"].mean()
            depth_norm_sq_mean = (df_sub["depth_norm"] ** 2).mean()

            label_parts = [plabel]

            # ── Linear fit (solid) — plot reference line only, no test ──
            is_full_stats = stype in full_stats_stypes
            lin_slope = results.get("lin_only_slope_median")
            if lin_slope is not None and not np.isnan(lin_slope):
                # Anchor line at data centroid
                intercept_plot = observed_mean - lin_slope * depth_norm_mean
                y_lin = intercept_plot + lin_slope * d_norm
                ax.plot(d_range, y_lin, "-", color=color,
                        linewidth=1.8, zorder=4)
                label_parts.append(f"β₁={lin_slope:+.3f}")

            # ── Quadratic fit (dashed) — the curvature hypothesis test ──
            quad_lin_slope = results.get("quad_lin_slope_mean")
            quad_term      = results.get("quad_term_median")
            quad_p         = (results.get("quad_term_p_within_adj") if is_full_stats
                               else results.get("quad_term_p_within"))
            if (quad_term is not None and not np.isnan(quad_term)
                    and quad_lin_slope is not None):
                # Anchor quadratic line at data centroid
                intercept_plot = (observed_mean
                                - quad_lin_slope * depth_norm_mean
                                - quad_term * depth_norm_sq_mean)
                y_quad = (intercept_plot
                        + quad_lin_slope * d_norm
                        + quad_term * d_norm**2)
                ax.plot(d_range, y_quad, "--", color=color,
                        linewidth=1.4, alpha=0.85, zorder=4)
                if is_full_stats:
                    label_parts.append(
                        f"β₂={quad_term:+.3f}{sig_stars(quad_p)}"
                    )
                else:
                    label_parts.append(
                        f"β₂={quad_term:+.3f}"
                    )

            n_cells = results.get("n_cells", "?")
            label_parts.append(f"(n={n_cells})")

            # Legend entry
            ax.plot([], [], "-", color=color, linewidth=1.8,
                    label="  ".join(label_parts))

        ax.set_xlabel("Layer depth", fontsize=10)
        ax.set_ylabel(meta["ylabel"], fontsize=10)
        print(f"{stype} {metric_key}")
        title = f"{meta['label']} — {stype}"
        if stype not in full_stats_stypes:
            title = f"{meta['label']} — {stype} - no significance tests possible (too few cells)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=7.5, framealpha=0.85)
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(
        "Hierarchical organisation: per-cell polynomial fits, aggregated across cells\n"
        "(solid = linear; dashed = quadratic; error bars = SEM across cells; "
        "lines anchored at data centroid)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("MIXED-EFFECTS REGRESSION ANALYSIS (linear + quadratic)")
    print("=" * 80)
    print(f"Properties : {PROPERTIES}")
    print(f"Model filter : {MODEL_FILTER if MODEL_FILTER else 'all models'}")
    print(f"Exclusions : { {k: v for k, v in EXCLUSIONS.items() if v} }")
    print(f"Metrics    : {list(METRICS)}")

    all_data  = {}
    all_mixed = {}

    for metric_key, meta in METRICS.items():
        print(f"\n{'='*80}")
        print(f"METRIC: {meta['label']}  ←  {meta['csv'].name}")
        print("=" * 80)

        long_df = load_csv_data(
            meta["csv"], meta["col"],
            STIMULUS_GROUPS, PROPERTIES, EXCLUSIONS,
            model_filter=MODEL_FILTER,
        )

        if len(long_df) == 0:
            continue

        # Normalize depth within each cell
        long_df = normalize_depth_inplace(long_df)

        for stype in STIMULUS_GROUPS:
            for prop in PROPERTIES:
                sub_df = long_df[(long_df["stimulus_type"] == stype)
                                 & (long_df["property"] == prop)]
                if len(sub_df) == 0:
                    continue

                key = (metric_key, stype, prop)
                all_data[key]  = sub_df
                all_mixed[key] = fit_per_cell_and_aggregate(sub_df)
                res = all_mixed[key]
                if res is None:
                    print(f"  skipped (too few cells: {sub_df['cell_id'].nunique()})")
                    continue

    # ── BH FDR correction (full-stats stypes only — "synthetic"-like stypes ───
    # are descriptive only). The curvature term is the only test family here,
    # corrected within measure, across the four features.
    bh_input = pd.DataFrame([
        {"measure": metric_key, "feature": prop, "p": res["quad_term_p"]}
        for (metric_key, stype, prop), res in all_mixed.items()
        if stype in FULL_STATS_STYPES and res is not None
    ])
    if not bh_input.empty:
        bh_out = bh_within_families(bh_input, "mixed_effects: quadratic (curvature) term")
        adj_lookup = {(row.measure, row.feature): row.p_adj for row in bh_out.itertuples()}
        for (metric_key, stype, prop), res in all_mixed.items():
            if stype in FULL_STATS_STYPES and res is not None:
                res["quad_term_p_adj"] = adj_lookup.get((metric_key, prop), np.nan)

    # BH FDR correction for the permutation (sign-flip, "within") p-value —
    # corrected separately from the Wilcoxon p.
    bh_input = pd.DataFrame([
        {"measure": metric_key, "feature": prop, "p": res["quad_term_p_within"]}
        for (metric_key, stype, prop), res in all_mixed.items()
        if stype in FULL_STATS_STYPES and res is not None
    ])
    if not bh_input.empty:
        bh_out = bh_within_families(bh_input, "mixed_effects: quadratic (curvature) term (permutation)")
        adj_lookup = {(row.measure, row.feature): row.p_adj for row in bh_out.itertuples()}
        for (metric_key, stype, prop), res in all_mixed.items():
            if stype in FULL_STATS_STYPES and res is not None:
                res["quad_term_p_within_adj"] = adj_lookup.get((metric_key, prop), np.nan)

    print("\n" + "=" * 80)
    print("BH-ADJUSTED RESULTS (full-stats stypes)")
    print("=" * 80)
    csv_rows = []
    for (metric_key, stype, prop), res in all_mixed.items():
        if stype not in FULL_STATS_STYPES or res is None:
            continue
        nan = float("nan")
        csv_rows.append({
            "analysis":        "mixed_effects",
            "metric":          metric_key,
            "stype":           stype,
            "feature":         prop,
            "test_type":       "quadratic_curvature",
            "N":               res["n_cells"],
            "n_models":        res.get("n_models", nan),
            "median":          res.get("quad_term_median", nan),
            "q1":              res.get("quad_term_q1", nan),
            "q3":              res.get("quad_term_q3", nan),
            "hl_estimate":     res.get("quad_term_hl", nan),
            "hl_ci_lo":        res.get("quad_term_ci_lo", nan),
            "hl_ci_hi":        res.get("quad_term_ci_hi", nan),
            "W":               res.get("quad_term_W", nan),
            "p":               res.get("quad_term_p", nan),
            "p_adj":           res.get("quad_term_p_adj", nan),
            "p_perm_block_model":   res.get("quad_term_p_block_model", nan),
            "p_perm_block_dataset": res.get("quad_term_p_block_dataset", nan),
            "p_perm_within":        res.get("quad_term_p_within", nan),
            "p_perm_within_adj":    res.get("quad_term_p_within_adj", nan),
            "icc_model":            res.get("quad_term_icc_model", nan),
            "icc_dataset":     res.get("quad_term_icc_dataset", nan),
            "n_eff":           res.get("quad_term_n_eff", nan),
            "rank_biserial_r": res.get("quad_term_rr", nan),
            "cohens_dz":       res.get("quad_term_dz", nan),
        })

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    fig = plot_all(all_data, all_mixed, METRICS, STIMULUS_GROUPS, PROPERTIES,
                   full_stats_stypes=FULL_STATS_STYPES)
    
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)