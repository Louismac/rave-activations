"""
Cross-layer cluster analysis: does cross-layer functional organization reveal
encoding patterns that single-layer analysis cannot capture?

Section-stratified version: each (model, dataset, section, feature) is a cell.
Within each section, the best cluster (max metric across that section's
6 clusters) is compared against the best layer (max metric across that
section's layers). This is the methodologically clean comparison because
K-means was run separately within each section.

Cells per measure (approx): 3 models × 3 natural datasets × 3 sections × 4
features = 108 cells, minus feature×dataset exclusions. Natural stimuli only.

Test: Wilcoxon signed-rank on cluster-minus-layer deltas across cells.

Splits reported:
  - Per feature
  - Cross-feature aggregate marked explicitly with [pooled] flag
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "analyse-within-layer-correlations"))
from rave_activation_clustering import RAVEActivationAnalyser
from perm_test import icc_and_neff, sign_permutation_p
from bh import bh_within_families, hodges_lehmann, bootstrap_hl_ci, rank_biserial_r
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ─────────────────────────────────────────────────────────────

PERM_DIR = Path(__file__).parent.parent / "permutations"

STIMULUS_TYPE = {
    "strings":    "natural",
    "vocals":     "natural",
    "drum_loops": "natural",
    "stimuli":    "synthetic",
}

PROPERTIES = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]

# Only keep rows whose "model" column contains one of these substrings
# (case-insensitive). Set to None to keep all models, e.g. ["encodec"].
OUTPUT_DIR = Path(__file__).parent

# MODEL_FILTER = ["encodec"]
# csv_out = OUTPUT_DIR / "cluster_layer_summary_encodec.csv"

MODEL_FILTER = ["strings", "drum_loops","taylor_vocal"]
csv_out = OUTPUT_DIR / "cluster_layer_summary.csv"


EXCLUSIONS = {
    "pitch": ["drum_loops"],
}

PROP_LABELS = {
    "pitch":              "Pitch",
    "bpm":                "BPM",
    "spectral_centroid":  "Spec. Centroid",
    "spectral_bandwidth": "Spec. Bandwidth",
}

# Section split: layer ordinal positions within each cell.
# Each cell has 28 layers; default split is 9/10/9.
# If your pipeline uses a different split, change these boundaries.
SECTION_BOUNDARIES = {
    "early":  (0,  9),   # layers 0..8 (9 layers)
    "middle": (9,  19),  # layers 9..18 (10 layers)
    "late":   (19, 28),  # layers 19..27 (9 layers)
}
SECTION_ORDER = ["early", "middle", "late"]


def layer_name_to_section(layer_name):
    """
    Map layer name to early/middle/late section, delegating to
    RAVEActivationAnalyser._assign_section so RAVE ('net.X...') and EnCodec
    ('layers.X...') naming schemes stay consistent with the section
    boundaries used elsewhere in the pipeline (e.g. the clustering that
    produced the cluster CSVs this script compares against).
    """
    return RAVEActivationAnalyser._assign_section(None, str(layer_name))


MEASURES = {
    "mean_r": {
        "label":          "Per-neuron mean |ρ|",
        "ylabel":         "Mean |ρ|",
        "layer_csv":      PERM_DIR / "plot_obs_r_per_layer.csv",
        "layer_col":      "obs_mean_r",
        "cluster_csv":    PERM_DIR / "permutation_baseline_clusters_table.csv",
        "cluster_col":    "observed_mean_r",
        "interpretation": "per-neuron",
    },
    "pct_exceeding": {
        "label":          "% neurons > null p95",
        "ylabel":         "% > p95",
        "layer_csv":      PERM_DIR / "plot_pct_exceeding_per_layer.csv",
        "layer_col":      "obs_pct_exceeding",
        "cluster_csv":    PERM_DIR / "permutation_baseline_clusters_table.csv",
        "cluster_col":    "observed_pct_exceeding_p95",
        "interpretation": "per-neuron",
    },
    "r2": {
        "label":            "Nonlinear probe R² (joint)",
        "ylabel":           "Observed R²",
        "layer_csv":        PERM_DIR / "plot_obs_r2_per_layer.csv",
        "layer_col":        "observed_r2",
        "layer_size_col":   "n_channels",
        "cluster_csv":      PERM_DIR / "permutation_baseline_nonlinear_clusters_table.csv",
        "cluster_col":      "observed_r2",
        "cluster_size_col": "n_channels",
        "interpretation":   "joint encoding",
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def categorise_cell(dataset):
    return STIMULUS_TYPE.get(dataset, "unknown")


def assign_layer_sections(layers_df):
    """
    Add a 'section' column to a layers DataFrame based on the numeric index
    in the layer name (matches the section logic used to define the
    cross-layer clusters).
    """
    layers_df = layers_df.copy()
    layers_df["section"] = layers_df["layer"].apply(layer_name_to_section)
    return layers_df


def load_best_per_cell(layer_csv, layer_col, cluster_csv, cluster_col,
                       min_cluster_pct=0.01, cluster_size_col="n_neurons",
                       layer_size_col=None, layer_n_lookup=None,
                       model_filter=None):
    """
    Load per-layer and per-cluster data, and return one row per
    (model, dataset, section, feature) cell with the best layer value
    and best cluster value within that section.

    Parameters
    ----------
    layer_csv, layer_col : Path, str
        CSV and column for the per-layer metric.
    cluster_csv, cluster_col : Path, str
        CSV and column for the per-cluster metric.
    min_cluster_pct : float or None
        If provided, exclude clusters smaller than this fraction of their
        section's total channel count.
    cluster_size_col : str
        Column in clusters DataFrame giving each cluster's size.
    layer_size_col : str or None
        Column in the layer CSV giving each layer's neuron/channel count.
        When provided the best layer's n is captured via idxmax.
    layer_n_lookup : DataFrame or None
        Fallback table with columns [model, dataset, layer, <n_col>] used
        when the layer CSV itself has no size column.  The <n_col> must be
        named the same as layer_size_col.
    model_filter : list[str] or None
        If given, only keep rows whose "model" value contains one of these
        substrings (case-insensitive).
    """
    if not layer_csv.exists() or not cluster_csv.exists():
        return pd.DataFrame()

    layers   = pd.read_csv(layer_csv)
    clusters = pd.read_csv(cluster_csv)

    if model_filter:
        pattern = "|".join(model_filter)
        layers   = layers[layers["model"].str.contains(pattern, case=False, na=False)]
        clusters = clusters[clusters["model"].str.contains(pattern, case=False, na=False)]

    # ── Assign each layer to a section ─────────────────────────────────────
    layers = assign_layer_sections(layers)
    n_unassigned = layers["section"].isna().sum()
    if n_unassigned > 0:
        layers = layers.dropna(subset=["section"])

    # ── Optional cluster-size filter (relative to section total) ───────────
    if min_cluster_pct is not None and cluster_size_col in clusters.columns:
        section_keys = ["model", "dataset", "feature", "section"]
        totals = (clusters
                  .groupby(section_keys)[cluster_size_col]
                  .sum()
                  .reset_index()
                  .rename(columns={cluster_size_col: "_section_total"}))
        clusters = clusters.merge(totals, on=section_keys, how="left")
        clusters["_cluster_pct"] = (
            clusters[cluster_size_col] / clusters["_section_total"]
        )
        clusters = clusters[clusters["_cluster_pct"] >= min_cluster_pct].copy()
        clusters = clusters.drop(columns=["_section_total", "_cluster_pct"])

    # ── Best layer per (model, dataset, section, feature) ──────────────────
    cell_keys = ["model", "dataset", "section", "feature"]

    # Augment layers with size info if not already present
    if layer_size_col and layer_size_col not in layers.columns and layer_n_lookup is not None:
        layers = layers.merge(
            layer_n_lookup[["model", "dataset", "layer", layer_size_col]],
            on=["model", "dataset", "layer"], how="left",
        )

    if layer_size_col and layer_size_col in layers.columns:
        idx_best_layer = layers.groupby(cell_keys)[layer_col].idxmax()
        best_layer = (layers
                      .loc[idx_best_layer, cell_keys + [layer_col, layer_size_col]]
                      .reset_index(drop=True)
                      .rename(columns={layer_col: "best_layer",
                                       layer_size_col: "best_layer_n"}))
    else:
        best_layer = (layers
                      .groupby(cell_keys)[layer_col]
                      .max()
                      .reset_index()
                      .rename(columns={layer_col: "best_layer"}))
        best_layer["best_layer_n"] = np.nan

    # ── Section total n (pre-size-filter: pool a layer sees) ───────────────
    has_size_col = cluster_size_col in clusters.columns
    if has_size_col:
        section_total = (clusters
                         .groupby(cell_keys)[cluster_size_col]
                         .sum()
                         .reset_index()
                         .rename(columns={cluster_size_col: "section_total_n"}))

    # ── Best cluster per cell (after optional filter) ──────────────────────
    if has_size_col:
        idx_best = clusters.groupby(cell_keys)[cluster_col].idxmax()
        best_cluster = (clusters
                        .loc[idx_best, cell_keys + [cluster_col, cluster_size_col]]
                        .reset_index(drop=True)
                        .rename(columns={cluster_col: "best_cluster",
                                         cluster_size_col: "best_cluster_n"}))
    else:
        best_cluster = (clusters
                        .groupby(cell_keys)[cluster_col]
                        .max()
                        .reset_index()
                        .rename(columns={cluster_col: "best_cluster"}))
        best_cluster["best_cluster_n"] = np.nan

    merged = best_layer.merge(best_cluster, on=cell_keys, how="inner")
    merged["delta"] = merged["best_cluster"] - merged["best_layer"]

    merged["stimulus_type"] = merged["dataset"].apply(categorise_cell)
    merged = merged[merged["stimulus_type"] == "natural"].reset_index(drop=True)

    return merged


def apply_exclusions(df, exclusions):
    rows = []
    for _, r in df.iterrows():
        if r["dataset"] in exclusions.get(r["feature"], []):
            continue
        rows.append(r)
    return pd.DataFrame(rows)


# ── plots ────────────────────────────────────────────────────────────────────

def plot_scatter(measures_results, output_path):
    """Scatter of best layer vs best cluster, one point per cell (natural stimuli only)."""
    n_measures = len(measures_results)
    fig, axes = plt.subplots(1, n_measures,
                             figsize=(5.5 * n_measures, 5),
                             squeeze=False)

    colors    = cm.tab10(np.linspace(0, 1, len(PROPERTIES)))
    feat_colors = dict(zip(PROPERTIES, colors))

    for col_idx, (measure_key, info) in enumerate(measures_results.items()):
        ax = axes[0][col_idx]
        df = info["df"]
        if len(df) == 0:
            continue

        lo = min(df["best_layer"].min(), df["best_cluster"].min())
        hi = max(df["best_layer"].max(), df["best_cluster"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.6, zorder=1)

        for feat in PROPERTIES:
            sub = df[df["feature"] == feat]
            if len(sub) == 0: continue
            ax.scatter(sub["best_layer"], sub["best_cluster"],
                       color=feat_colors[feat], marker="o",
                       s=70, alpha=0.75, edgecolor="white", linewidth=0.6,
                       zorder=3, label=feat)

        ax.set_xlabel(f"Best layer in section  ({MEASURES[measure_key]['ylabel']})", fontsize=10)
        ax.set_ylabel(f"Best cluster in section  ({MEASURES[measure_key]['ylabel']})", fontsize=10)
        ax.set_title(f"{info['label']}\n({info['interpretation']})",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=7, framealpha=0.85, ncol=2)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Within-section cluster vs best individual layer, natural stimuli "
        "(one point = one section of one cell)\n"
        "Above identity line: cluster > layer.  Below: layer > cluster.",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary_csv(measures_results):
    """
    Build a tidy summary DataFrame, one row per (measure × feature), for the
    natural-stimulus paired test (cluster vs layer).

    Permutation p-values use sign-flip tests on the cluster-minus-layer deltas.
    BH correction is applied within each measure family across the 4 features.
    """
    rows = []
    for measure_key, info in measures_results.items():
        df = info["df"]
        has_cluster_n = "best_cluster_n" in df.columns
        has_layer_n   = "best_layer_n"   in df.columns

        for feat in PROPERTIES:
            gdf = df[df["feature"] == feat]
            if len(gdf) < 2:
                continue
            deltas       = gdf["delta"].values
            layer_mean   = float(gdf["best_layer"].mean())
            cluster_mean = float(gdf["best_cluster"].mean())
            mean_delta   = float(np.mean(deltas))
            median_delta = float(np.median(deltas))
            pct_delta    = (mean_delta / layer_mean * 100
                            if layer_mean != 0 else np.nan)
            n_positive   = int(np.sum(deltas > 0))
            sem          = (float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
                            if len(deltas) > 1 else np.nan)

            models_g   = gdf["model"].values
            datasets_g = gdf["dataset"].values
            p_block_model,   _ = sign_permutation_p(deltas, models_g,   null="block")
            p_block_dataset, _ = sign_permutation_p(deltas, datasets_g, null="block")
            p_within,        _ = sign_permutation_p(deltas, models_g,   null="within")
            ic_model   = icc_and_neff(deltas, models_g)
            ic_dataset = icc_and_neff(deltas, datasets_g)
            hl_ci_lo, hl_ci_hi = bootstrap_hl_ci(deltas)

            rows.append({
                "measure":               info["label"],
                "measure_key":           measure_key,
                "feature":               PROP_LABELS.get(feat, feat),
                "n_cells":               len(gdf),
                "layer_mean":            layer_mean,
                "cluster_mean":          cluster_mean,
                "mean_delta":            mean_delta,
                "median_delta":          median_delta,
                "hl_estimate":           hodges_lehmann(deltas),
                "hl_ci_lo":              hl_ci_lo,
                "hl_ci_hi":              hl_ci_hi,
                "sem_delta":             sem,
                "pct_delta":             pct_delta,
                "n_positive":            n_positive,
                "rank_biserial_r":       rank_biserial_r(deltas),
                "avg_best_cluster_n":    float(gdf["best_cluster_n"].mean())
                                         if has_cluster_n else np.nan,
                "avg_best_layer_n":      float(gdf["best_layer_n"].mean())
                                         if has_layer_n else np.nan,
                "p_perm_block_model":    p_block_model,
                "p_perm_block_dataset":  p_block_dataset,
                "p_perm_within":         p_within,
                "p_adj":                 np.nan,   # filled below by BH
                "icc_model":             ic_model["icc"],
                "icc_dataset":           ic_dataset["icc"],
            })

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    # BH FDR correction: within each measure family, across features.
    # Uses p_perm_within as the primary test (cell-level sign flips).
    bh_input = pd.DataFrame([
        {"measure": r["measure_key"], "feature": r["feature"], "p": r["p_perm_within"]}
        for r in rows
    ])
    bh_out = bh_within_families(bh_input, "cluster_layer_comparison")
    adj_lookup = {(row.measure, row.feature): row.p_adj for row in bh_out.itertuples()}
    summary_df["p_adj"] = summary_df.apply(
        lambda r: adj_lookup.get((r["measure_key"], r["feature"]), np.nan), axis=1
    )

    return summary_df


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    measures_results = {}

    # Pre-load layer n lookup from the r2 CSV (has n_channels per model/dataset/layer).
    # Used as fallback for measures whose own layer CSV has no size column.
    _r2_layer_csv = MEASURES["r2"]["layer_csv"]
    layer_n_lookup = (pd.read_csv(_r2_layer_csv)[["model", "dataset", "layer", "n_channels"]]
                      .drop_duplicates() if _r2_layer_csv.exists() else None)

    for measure_key, info in MEASURES.items():
        df = load_best_per_cell(
            info["layer_csv"], info["layer_col"],
            info["cluster_csv"], info["cluster_col"],
            cluster_size_col=info.get("cluster_size_col", "n_neurons"),
            layer_size_col=info.get("layer_size_col", "n_channels"),
            layer_n_lookup=layer_n_lookup,
            model_filter=MODEL_FILTER,
        )
        if len(df) == 0:
            continue
        df = apply_exclusions(df, EXCLUSIONS)
        measures_results[measure_key] = {
            "df":             df,
            "label":          info["label"],
            "interpretation": info["interpretation"],
        }

    if measures_results:
        OUTPUT_DIR = Path(__file__).parent
        plot_scatter(measures_results,
                     OUTPUT_DIR / "cluster_vs_layer_scatter_section.png")

        summary_df = build_summary_csv(measures_results)
        summary_df.to_csv(csv_out, index=False)
