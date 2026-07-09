#!/usr/bin/env python3
"""
Statistical tests: synthetic vs natural audio robustness.
Metrics × features panel plot: mean |r| and % neurons above null p95.
Uses Wilcoxon signed-rank throughout (differences non-normal).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

from bh import bh_within_families, hodges_lehmann, bootstrap_hl_ci, rank_biserial_r
from perm_test import sign_permutation_p, icc_and_neff

# ── configuration ─────────────────────────────────────────────────────────────

results_path = Path(__file__).parent.parent.resolve() / "results" / "6_cluster"
OUTPUT_DIR = Path(__file__).parent

MODELS             = ["strings", "drum_loops", "taylor_vocal"]
csv_path = OUTPUT_DIR / "natural_vs_synthetic_results.csv"
out = OUTPUT_DIR / "natural_vs_synthetic.png"

# MODELS             = ["encodec"]
# csv_path = OUTPUT_DIR / "natural_vs_synthetic_encodec_results.csv"
# out = OUTPUT_DIR / "natural_vs_synthetic_encodec.png"

out.parent.mkdir(exist_ok=True, parents=True)

NATURAL_DATASETS   = ["strings", "vocals", "drum_loops"]
SYNTHETIC_DATASETS = ["stimuli"]

PROPERTIES = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]

# Datasets excluded per feature (no valid signal)
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

METRICS = ["mean_r", "pct_above_p95", "r2"]

METRIC_LABELS = {
    "mean_r":        "Mean |r|",
    "pct_above_p95": "% neurons above null p95",
    "r2":            "Nonlinear probe R²",
}

METRIC_NULL_LINES = {
    "mean_r":        None,
    "pct_above_p95": 5.0,    # null gives 5% by definition
    "r2":            None,
}



# ── load per-layer data ───────────────────────────────────────────────────────

print("=" * 80)
print("STATISTICAL TESTS: SYNTHETIC vs NATURAL AUDIO")
print(f"Properties: {PROPERTIES}")
print("=" * 80)

all_data = []

for model in MODELS:
    for dataset in NATURAL_DATASETS + SYNTHETIC_DATASETS:
        var_path  = results_path / model / dataset / "variance_correlation.json"
        perm_path = results_path / model / dataset / "permutation_baseline.json"
        nl_path   = results_path / model / dataset / "permutation_baseline_nonlinear.json"

        if not var_path.exists():
            continue

        with open(var_path) as f:
            var = json.load(f)

        null_p95 = {}
        if perm_path.exists():
            with open(perm_path) as f:
                perm = json.load(f)
            for prop in PROPERTIES:
                if prop in perm and "null_p95_r" in perm[prop]:
                    null_p95[prop] = perm[prop]["null_p95_r"]

        r2_by_layer = {}   # {prop: {layer: r2}}
        if nl_path.exists():
            with open(nl_path) as f:
                nl = json.load(f)
            for prop in PROPERTIES:
                if prop in nl:
                    r2_by_layer[prop] = {
                        layer: v["observed_r2"] for layer, v in nl[prop].items()
                    }

        content_type = "synthetic" if dataset in SYNTHETIC_DATASETS else "natural"

        for layer_name, layer_data in var.items():
            if layer_name.startswith("section_"):
                continue
            for prop in PROPERTIES:
                if prop not in layer_data:
                    continue
                corrs = np.array(layer_data[prop].get("all_correlations", []))
                if len(corrs) == 0:
                    continue
                threshold = null_p95.get(prop)
                pct = float((corrs > threshold).mean() * 100) if threshold is not None else np.nan
                r2  = r2_by_layer.get(prop, {}).get(layer_name, np.nan)
                all_data.append({
                    "model":        model,
                    "dataset":      dataset,
                    "content_type": content_type,
                    "layer":        layer_name,
                    "property":     prop,
                    "mean_r":       float(corrs.mean()),
                    "pct_above_p95": pct,
                    "null_p95":     threshold if threshold is not None else np.nan,
                    "r2":           r2,
                })

df = pd.DataFrame(all_data)
print(f"\nLoaded {len(df)} layer×property observations")
print(f"  Natural:   {(df['content_type']=='natural').sum()}")
print(f"  Synthetic: {(df['content_type']=='synthetic').sum()}")

# ── layer-matched Wilcoxon ────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("LAYER-MATCHED WILCOXON (within-layer, testing synthetic > natural)")
print("=" * 80)

results = {}   # (prop, metric) -> (comp_df, w_stat, w_p)

for prop in PROPERTIES:
    excluded = EXCLUSIONS.get(prop, [])
    feat_df = df[(df["property"] == prop) & (~df["dataset"].isin(excluded))].copy()

    if feat_df.empty:
        continue

    print(f"\n--- {PROP_LABELS.get(prop, prop).upper()} ---")
    if excluded:
        print(f"  (excluding datasets: {excluded})")

    for metric in METRICS:
        pairs = []
        for model in MODELS:
            model_df = feat_df[feat_df["model"] == model]
            for layer in model_df["layer"].unique():
                layer_df = model_df[model_df["layer"] == layer]
                nat = layer_df[layer_df["content_type"] == "natural"][metric].mean()
                syn = layer_df[layer_df["content_type"] == "synthetic"][metric].mean()
                if not (np.isnan(nat) or np.isnan(syn)):
                    pairs.append({
                        "model": model, "layer": layer,
                        "natural": nat, "synthetic": syn,
                        "difference": syn - nat,
                    })

        print(f"N={len(pairs)}")
        if len(pairs) < 2:
            print(f"  [{metric}]  insufficient paired data — skipping")
            continue

        comp_df = pd.DataFrame(pairs)
        w_stat, w_p = wilcoxon(comp_df["synthetic"], comp_df["natural"], alternative="greater")

        unit = "%" if metric == "pct_above_p95" else ""
        sig  = "***" if w_p < 0.001 else ("**" if w_p < 0.01 else ("*" if w_p < 0.05 else "n.s."))
        print(f"  [{metric}]  N pairs={len(comp_df)}")
        print(f"    Natural mean:   {comp_df['natural'].mean():.3f}{unit}")
        print(f"    Synthetic mean: {comp_df['synthetic'].mean():.3f}{unit}")
        print(f"    Wilcoxon  W={w_stat:.1f}, p={w_p:.6f}  {sig}")

        diffs = comp_df["synthetic"].values - comp_df["natural"].values
        ci_lo, ci_hi = bootstrap_hl_ci(diffs)

        # Model-stratified sign-permutation test + ICC (diffs share a model ->
        # naive Wilcoxon understates null variance; see perm_test.py). With a
        # single model (n_models=1) the block test is degenerate (p=1) and ICC
        # is undefined — only the within-model test is informative then.
        groups = comp_df["model"].values
        p_block, method_block = sign_permutation_p(diffs, groups, null="block")
        p_within, method_within = sign_permutation_p(diffs, groups, null="within")
        icc_info = icc_and_neff(diffs, groups)

        print(f"    Model-controlled: p_block={p_block:.4f} ({method_block})  "
              f"p_within={p_within:.4f} ({method_within})  "
              f"ICC={icc_info['icc']:.3f}  n_eff={icc_info['n_eff']:.2f} "
              f"(n_models={icc_info['n_models']})")

        results[(prop, metric)] = {
            "comp_df":         comp_df,
            "W": w_stat, "p": w_p, "p_adj": np.nan,
            "N":               len(comp_df),
            "nat_median":      float(np.median(comp_df["natural"])),
            "nat_q1":          float(np.percentile(comp_df["natural"], 25)),
            "nat_q3":          float(np.percentile(comp_df["natural"], 75)),
            "syn_median":      float(np.median(comp_df["synthetic"])),
            "syn_q1":          float(np.percentile(comp_df["synthetic"], 25)),
            "syn_q3":          float(np.percentile(comp_df["synthetic"], 75)),
            "hl_estimate":     hodges_lehmann(diffs),
            "hl_ci_lo":        ci_lo,
            "hl_ci_hi":        ci_hi,
            "rank_biserial_r": rank_biserial_r(diffs),
            "p_block":         p_block,
            "p_block_method":  method_block,
            "p_within":        p_within,
            "p_within_method": method_within,
            "p_within_adj":    np.nan,
            "icc":             icc_info["icc"],
            "n_eff":           icc_info["n_eff"],
            "n_models":        icc_info["n_models"],
        }

# ── Benjamini–Hochberg FDR correction (within each metric, across features) ────
# Two independent families: naive Wilcoxon p, and the model-controlled within-
# permutation p. (p_block is left unadjusted — with only a few models its
# values are too coarse/quantized for FDR correction to be meaningful.)

bh_input = pd.DataFrame([
    {"measure": metric, "feature": prop, "p": r["p"]}
    for (prop, metric), r in results.items()
])
bh_out = bh_within_families(bh_input, "natural_vs_synthetic")
p_adj_lookup = {(row.feature, row.measure): row.p_adj for row in bh_out.itertuples()}

bh_within_input = pd.DataFrame([
    {"measure": metric, "feature": prop, "p": r["p_within"]}
    for (prop, metric), r in results.items()
])
bh_within_out = bh_within_families(bh_within_input, "natural_vs_synthetic (p_within)")
p_within_adj_lookup = {(row.feature, row.measure): row.p_adj for row in bh_within_out.itertuples()}

print("\n" + "=" * 80)
print("BH-ADJUSTED RESULTS")
print("=" * 80)
for (prop, metric), r in results.items():
    p_adj = p_adj_lookup[(prop, metric)]
    r["p_adj"] = p_adj
    r["p_within_adj"] = p_within_adj_lookup[(prop, metric)]
    sig_adj = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "n.s."))
    print(f"  [{PROP_LABELS.get(prop, prop)} / {metric}]  "
          f"p={r['p']:.6f}  p_adj={p_adj:.6f}  {sig_adj}  "
          f"HL={r['hl_estimate']:.4f} [{r['hl_ci_lo']:.4f}, {r['hl_ci_hi']:.4f}]  "
          f"r={r['rank_biserial_r']:.3f}  "
          f"p_block={r['p_block']:.4f}  "
          f"p_within={r['p_within']:.4f}  p_within_adj={r['p_within_adj']:.4f}  "
          f"ICC={r['icc']:.3f} (n_eff={r['n_eff']:.2f})")

csv_rows = []
for (prop, metric), r in results.items():
    csv_rows.append({
        "analysis":        "natural_vs_synthetic",
        "feature":         prop,
        "metric":          metric,
        "N":               r["N"],
        "nat_median":      r["nat_median"],
        "nat_q1":          r["nat_q1"],
        "nat_q3":          r["nat_q3"],
        "syn_median":      r["syn_median"],
        "syn_q1":          r["syn_q1"],
        "syn_q3":          r["syn_q3"],
        "hl_estimate":     r["hl_estimate"],
        "hl_ci_lo":        r["hl_ci_lo"],
        "hl_ci_hi":        r["hl_ci_hi"],
        "W":               r["W"],
        "p":               r["p"],
        "p_adj":           r["p_adj"],
        "rank_biserial_r": r["rank_biserial_r"],
        "p_block":         r["p_block"],
        "p_block_method":  r["p_block_method"],
        "p_within":        r["p_within"],
        "p_within_adj":    r["p_within_adj"],
        "p_within_method": r["p_within_method"],
        "icc":             r["icc"],
        "n_eff":           r["n_eff"],
        "n_models":        r["n_models"],
    })
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"\nCSV saved: {csv_path}")

# ── plot ──────────────────────────────────────────────────────────────────────

n_rows = len(METRICS)
n_cols = len(PROPERTIES)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 5.5 * n_rows),
                         squeeze=False)
fig.suptitle("Natural vs Synthetic: Layer-Matched Comparison",
             fontsize=14, fontweight="bold")

for row_idx, metric in enumerate(METRICS):
    ylabel    = METRIC_LABELS[metric]
    null_line = METRIC_NULL_LINES[metric]

    for col_idx, prop in enumerate(PROPERTIES):
        ax = axes[row_idx][col_idx]

        if (prop, metric) not in results:
            ax.set_visible(False)
            continue

        r = results[(prop, metric)]
        comp_df, w_stat, w_p, p_adj = r["comp_df"], r["W"], r["p"], r["p_adj"]
        nat_mean = comp_df["natural"].mean()
        syn_mean = comp_df["synthetic"].mean()

        for _, row in comp_df.iterrows():
            ax.plot([0, 1], [row["natural"], row["synthetic"]], "o-",
                    alpha=0.25, linewidth=1, markersize=3, color="gray")

        ax.plot([0, 1], [nat_mean, syn_mean], "ro-",
                linewidth=3, markersize=11, label="Mean", zorder=10)

        if null_line is not None:
            ax.axhline(y=null_line, color="blue", linestyle=":", linewidth=1.5,
                       alpha=0.6, label=f"Null ({null_line}%)")

        sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "n.s."))
        ax.text(0.5, 0.97, sig, ha="center", va="top",
                transform=ax.transAxes, fontsize=20, fontweight="bold")

        if metric == "pct_above_p95":
            nat_label, syn_label = f"{nat_mean:.1f}%", f"{syn_mean:.1f}%"
        else:
            nat_label, syn_label = f"{nat_mean:.3f}", f"{syn_mean:.3f}"
        ax.text(-0.15, nat_mean, nat_label, ha="right", va="center",
                fontsize=9, fontweight="bold")
        ax.text(1.15, syn_mean, syn_label, ha="left", va="center",
                fontsize=9, fontweight="bold")

        excl = EXCLUSIONS.get(prop, [])
        title = PROP_LABELS.get(prop, prop)
        if excl:
            title += f"\n(excl. {', '.join(excl)})"
        ax.set_title(f"{title}\nW={w_stat:.0f}, p={w_p:.4f} (BH adj {p_adj:.4f})",
                     fontsize=10, fontweight="bold")
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Natural", "Synthetic"], fontsize=10)
        ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
