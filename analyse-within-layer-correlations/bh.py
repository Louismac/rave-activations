import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from scipy.stats import false_discovery_control

    def _fdr_bh(pvals):
        return false_discovery_control(pvals, method="bh")
except ImportError:
    # scipy < 1.11 doesn't have false_discovery_control; fall back to statsmodels
    from statsmodels.stats.multitest import multipletests

    def _fdr_bh(pvals):
        return multipletests(pvals, method="fdr_bh")[1]

FEATURES = ["pitch", "bpm", "spectral_centroid", "spectral_bandwidth"]
MEASURES = ["mean_r", "pct", "r2"]


def bh_within_families(results_df, analysis_name):
    """
    Apply Benjamini-Hochberg FDR within each (measure) family — i.e. across
    the four features for a given measure within one analysis.

    results_df : DataFrame with columns ['measure', 'feature', 'p']
                 (one row per feature-test; natural cells only — exclude
                 synthetic, which is descriptive).
    Returns the same frame with a 'p_adj' column and a 'sig_changed' flag.
    """
    out = results_df.copy()
    out["p_adj"] = np.nan

    for measure in out["measure"].unique():
        mask = out["measure"] == measure
        pvals = out.loc[mask, "p"].values
        # drop NaNs (e.g. a feature with no test) before correcting
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            continue
        adj = np.full_like(pvals, np.nan, dtype=float)
        adj[valid] = _fdr_bh(pvals[valid])
        out.loc[mask, "p_adj"] = adj

    # Flag any test whose significance verdict changes at alpha=0.05
    out["sig_raw"] = out["p"] < 0.05
    out["sig_adj"] = out["p_adj"] < 0.05
    out["sig_changed"] = out["sig_raw"] != out["sig_adj"]

    n_changed = out["sig_changed"].sum()
    return out


def rank_biserial_r(diffs):
    """
    Rank-biserial r for a one-sample or paired Wilcoxon (differences vs 0).
    r = (W+ − W−) / (W+ + W−), range [−1, 1]; positive means diffs tend > 0.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    W_pos = float(np.sum(ranks[d > 0]))
    W_neg = float(np.sum(ranks[d < 0]))
    denom = W_pos + W_neg
    return float((W_pos - W_neg) / denom) if denom > 0 else np.nan


def hodges_lehmann(values):
    """
    Hodges–Lehmann location estimate: median of all Walsh averages
    (v_i + v_j) / 2 for i ≤ j. For differences (paired) or raw values (one-sample).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return np.nan
    ii, jj = np.triu_indices(n)
    return float(np.median((v[ii] + v[jj]) / 2))


def bootstrap_hl_ci(values, n_boot=2000, ci=0.95, seed=42):
    """
    Bootstrap percentile CI on the Hodges–Lehmann estimate via resampling.
    Vectorised: draws all bootstrap samples at once then computes Walsh medians.
    Returns (ci_lo, ci_hi); both NaN if n < 2.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(n)
    # (n_boot, n) → (n_boot, n_walsh) Walsh averages, median across walsh axis
    samples = rng.choice(v, size=(n_boot, n), replace=True)
    boots = np.median((samples[:, ii] + samples[:, jj]) / 2, axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, alpha * 100)), float(np.percentile(boots, (1 - alpha) * 100))