"""
Unified confidence-interval helpers for the RAVE encoding paper.

Three families of quantity are reported in the paper, and each needs a
different CI method:

  1. A single correlation computed from n paired samples
       → Fisher-z CI (Spearman-corrected).
       e.g. "this neuron's |rho| over 500 samples is 0.34, 95% CI [.., ..]"

  2. An aggregate of many values (mean |rho| across neurons or cells,
     median curvature coefficient across cells)
       → percentile / BCa bootstrap CI.

  3. A paired difference between two conditions across cells
     (ID vs OOD, cluster vs layer)
       → bootstrap CI on the paired differences, plus the
         Hodges-Lehmann point estimate (the location estimate that
         pairs naturally with the Wilcoxon signed-rank test) and the
         rank-biserial effect size.

The public entry points are:
    fisher_z_ci(r, n, ...)             # family 1
    bootstrap_ci(values, ...)          # family 2
    paired_difference_ci(a, b, ...)    # family 3  (returns CI + HL + effect size)

All return plain Python floats so they drop straight into f-strings / LaTeX.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, wilcoxon


# ─────────────────────────────────────────────────────────────────────────────
# Family 1: single correlation  →  Fisher-z CI
# ─────────────────────────────────────────────────────────────────────────────

def fisher_z_ci(r, n, confidence=0.95, spearman=True, folded=False):
    """
    Confidence interval for a single correlation via the Fisher z transform.

    Parameters
    ----------
    r : float
        Observed correlation. If `folded` is True this is |rho|; see note.
    n : int
        Number of paired observations the correlation was computed from
        (for this study, the per-cell sample count, e.g. 500 — NOT the
        number of neurons or cells).
    confidence : float
        Confidence level (default 0.95).
    spearman : bool
        If True, apply the Fieller/Hartley-Pearson 1.03 variance inflation
        appropriate for Spearman rho. If False, use the Pearson SE.
    folded : bool
        Set True if `r` is an absolute correlation |rho|. The Fisher-z CI
        assumes a signed correlation; for |rho| clearly above the null it is
        a good approximation, but near zero it is optimistic. When True we
        clip the lower bound at 0 (an absolute correlation cannot be < 0) and
        the result should be treated as approximate.

    Returns
    -------
    (lo, hi) : tuple of floats
        Lower and upper CI bounds in correlation units. (nan, nan) if n <= 3.

    Notes
    -----
    The interval is asymmetric in r-space (correct: correlations near +-1
    have less room to vary). For an absolute correlation reported well above
    the null threshold (|rho| > ~0.2 here), the folding bias is negligible.
    """
    if n is None or n <= 3:
        return (float("nan"), float("nan"))

    r_clipped = float(np.clip(r, -0.9999999, 0.9999999))
    z = np.arctanh(r_clipped)
    se = (1.03 if spearman else 1.0) / np.sqrt(n - 3)
    crit = norm.ppf(1 - (1 - confidence) / 2)

    lo = float(np.tanh(z - crit * se))
    hi = float(np.tanh(z + crit * se))

    if folded:
        # |rho| cannot be negative; a CI crossing zero is reported from 0.
        lo = max(lo, 0.0)

    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Family 2: aggregate of many values  →  bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values, statistic=np.mean, n_boot=10000,
                 confidence=0.95, method="bca", seed=None):
    """
    Bootstrap confidence interval for a statistic of a set of values.

    Use for aggregates: mean |rho| across neurons, mean/median curvature
    coefficient across cells, etc.

    Parameters
    ----------
    values : array-like
        The sample (e.g. per-cell coefficients, or per-neuron correlations).
    statistic : callable
        Statistic to bootstrap (np.mean, np.median, etc.). Default np.mean.
    n_boot : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level.
    method : {"percentile", "bca"}
        "percentile" is the simple percentile interval. "bca" applies the
        bias-corrected and accelerated adjustment, which is more accurate for
        skewed statistics (recommended; falls back to percentile if the
        acceleration is undefined, e.g. n < 2).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    (point, lo, hi) : tuple of floats
        The observed statistic and its CI bounds.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    if n == 1:
        v = float(statistic(x))
        return (v, v, v)

    theta_hat = float(statistic(x))

    # Bootstrap distribution
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(x[i]) for i in idx], dtype=float)

    alpha = 1 - confidence
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    if method == "percentile":
        lo = float(np.percentile(boot, lo_q))
        hi = float(np.percentile(boot, hi_q))
        return theta_hat, lo, hi

    # ---- BCa ----
    # bias-correction factor z0
    prop_less = np.mean(boot < theta_hat)
    # guard against 0 or 1 proportions (z0 -> +-inf)
    prop_less = min(max(prop_less, 1.0 / (n_boot + 1)), 1 - 1.0 / (n_boot + 1))
    z0 = norm.ppf(prop_less)

    # acceleration via jackknife
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
    jack_mean = jack.mean()
    diffs = jack_mean - jack
    denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    if denom == 0:
        # acceleration undefined → fall back to percentile
        lo = float(np.percentile(boot, lo_q))
        hi = float(np.percentile(boot, hi_q))
        return theta_hat, lo, hi
    a = np.sum(diffs ** 3) / denom

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)

    def adjust(z_alpha):
        num = z0 + z_alpha
        return norm.cdf(z0 + num / (1 - a * num))

    p_lo = 100 * adjust(z_lo)
    p_hi = 100 * adjust(z_hi)
    lo = float(np.percentile(boot, p_lo))
    hi = float(np.percentile(boot, p_hi))
    return theta_hat, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Family 3: paired difference across cells  →  bootstrap CI + HL + effect size
# ─────────────────────────────────────────────────────────────────────────────

def hodges_lehmann_paired(diffs):
    """
    Hodges-Lehmann estimator for paired data: the median of the Walsh
    averages (pairwise means (d_i + d_j)/2, i <= j). This is the location
    estimate that the Wilcoxon signed-rank test is testing against zero.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    # Walsh averages
    walsh = (d[:, None] + d[None, :]) / 2.0
    iu = np.triu_indices(n, k=0)
    return float(np.median(walsh[iu]))


def rank_biserial_paired(diffs):
    """
    Matched-pairs rank-biserial correlation, the effect size that pairs with
    the Wilcoxon signed-rank test. Ranges -1..+1; sign follows the dominant
    direction of the differences. Computed as the difference between the
    proportion of signed-rank mass that is positive vs negative.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]                       # signed-rank drops zeros
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = _rankdata(np.abs(d))
    R_plus = ranks[d > 0].sum()
    R_minus = ranks[d < 0].sum()
    total = R_plus + R_minus
    if total == 0:
        return float("nan")
    return float((R_plus - R_minus) / total)


def _rankdata(a):
    """Average-rank of values (ties get mean rank). Small dependency-free helper."""
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    # find groups of equal values
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0  # average of 1-based ranks
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def paired_difference_ci(a, b, n_boot=10000, confidence=0.95,
                         method="bca", seed=None):
    """
    CI and effect size for a paired difference a - b across cells, designed
    to accompany a Wilcoxon signed-rank test.

    Parameters
    ----------
    a, b : array-like
        Paired values (e.g. cluster vs layer, or ID vs OOD) — one pair per cell.
    n_boot, confidence, method, seed : see bootstrap_ci.

    Returns
    -------
    dict with:
        n            : number of pairs
        mean_diff    : mean of (a - b)
        median_diff  : median of (a - b)
        hl_estimate  : Hodges-Lehmann location estimate (pairs with Wilcoxon)
        ci_lo, ci_hi : bootstrap CI on the Hodges-Lehmann estimate
        wilcoxon_W   : Wilcoxon signed-rank statistic
        wilcoxon_p   : two-sided p-value
        rank_biserial: matched-pairs rank-biserial effect size (-1..+1)
        cohen_dz     : standardized mean difference (mean / sd of diffs)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    n = len(diffs)

    out = {
        "n": n,
        "mean_diff": float(np.mean(diffs)) if n else float("nan"),
        "median_diff": float(np.median(diffs)) if n else float("nan"),
        "hl_estimate": hodges_lehmann_paired(diffs),
        "rank_biserial": rank_biserial_paired(diffs),
    }

    # Cohen's d_z
    if n > 1 and np.std(diffs, ddof=1) > 0:
        out["cohen_dz"] = float(np.mean(diffs) / np.std(diffs, ddof=1))
    else:
        out["cohen_dz"] = float("nan")

    # Wilcoxon
    if n >= 1 and np.any(diffs != 0):
        try:
            W, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_W"] = float(W)
            out["wilcoxon_p"] = float(p)
        except ValueError:
            out["wilcoxon_W"] = float("nan")
            out["wilcoxon_p"] = float("nan")
    else:
        out["wilcoxon_W"] = float("nan")
        out["wilcoxon_p"] = float("nan")

    # Bootstrap CI on the Hodges-Lehmann estimate
    if n >= 2:
        _, lo, hi = bootstrap_ci(diffs, statistic=hodges_lehmann_paired,
                                 n_boot=n_boot, confidence=confidence,
                                 method=method, seed=seed)
        out["ci_lo"], out["ci_hi"] = lo, hi
    else:
        out["ci_lo"], out["ci_hi"] = float("nan"), float("nan")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: format a CI for inline text / LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def fmt_ci(point, lo, hi, decimals=3):
    """Return a string like '0.340 [0.318, 0.362]'."""
    d = decimals
    return f"{point:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Family 1 — single Spearman correlation, n=500 samples")
    for r in [0.288, 0.435, 0.087]:
        lo, hi = fisher_z_ci(r, n=500, spearman=True, folded=True)
        print(f"  |rho| = {r:.3f}  →  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\nFamily 2 — mean of per-cell coefficients (bootstrap BCa)")
    rng = np.random.default_rng(0)
    coeffs = rng.normal(0.155, 0.10, size=6)   # mimic 6 per-cell beta2 values
    point, lo, hi = bootstrap_ci(coeffs, statistic=np.mean, seed=0)
    print(f"  mean beta2 = {fmt_ci(point, lo, hi)}")
    point, lo, hi = bootstrap_ci(coeffs, statistic=np.median, seed=0)
    print(f"  median beta2 = {fmt_ci(point, lo, hi)}")

    print("\nFamily 3 — paired difference (ID vs OOD style), 9 cells")
    id_vals  = rng.normal(0.576, 0.08, size=9)
    ood_vals = rng.normal(0.405, 0.08, size=9)
    res = paired_difference_ci(id_vals, ood_vals, seed=0)
    print(f"  n = {res['n']}")
    print(f"  HL estimate = {res['hl_estimate']:.3f}  "
          f"95% CI [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]")
    print(f"  Wilcoxon W = {res['wilcoxon_W']:.0f}, p = {res['wilcoxon_p']:.4f}")
    print(f"  rank-biserial = {res['rank_biserial']:.3f}, "
          f"Cohen dz = {res['cohen_dz']:.3f}")