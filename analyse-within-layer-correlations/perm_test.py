"""
Model-stratified sign-permutation test + ICC, for inclusion in the
mixed_effects and cluster_layer_comparison output CSVs.

For per-cell values (curvature coefficients, or cluster-minus-layer deltas)
that share a model, the units are positively correlated within model. For a
sign/rank test this UNDERSTATES the null variance -> the naive Wilcoxon is
anti-conservative. These functions build the null from sign flips that respect
the model grouping:

  block  : flip a whole model's sign together (2^n_models patterns; floors the
           two-sided p at 2 / 2^n_models -> 0.25 for 3 models). Primary test for
           whether an effect generalises across models.
  within : flip each cell independently (2^N patterns). Matches the naive
           Wilcoxon when between-model variance is low (ICC ~ 0).

ICC (fraction of variance between models) reports where the effective N sits
between n_models and N.
"""
import itertools
import numpy as np
import pandas as pd


def _stat(v, signs):
    return float(np.sum(signs * v))


def sign_permutation_p(values, groups, null="block",
                       max_exact=200000, n_mc=100000, seed=0):
    """
    Two-sided model-stratified sign-permutation p-value for location != 0.
    Returns (p_perm, method_str). Returns (nan, 'n<1') if empty/all-zero.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    m = ~np.isnan(v)
    v, g = v[m], g[m]
    n = len(v)
    if n == 0 or not np.any(v != 0):
        return np.nan, "n<1"

    obs = abs(_stat(v, np.ones(n)))
    models = list(pd.unique(g))
    k = len(models)

    if null == "block":
        idx = {mm: i for i, mm in enumerate(models)}
        unit_ix = np.array([idx[x] for x in g])
        npat = 2 ** k
        if npat <= max_exact:
            stats = np.fromiter(
                (abs(_stat(v, np.array(c, float)[unit_ix]))
                 for c in itertools.product([1, -1], repeat=k)),
                dtype=float, count=npat)
            method = f"exact_block_{npat}pat_{k}models"
        else:
            stats = np.fromiter(
                (abs(_stat(v, rng.choice([1., -1.], size=k)[unit_ix]))
                 for _ in range(n_mc)), dtype=float, count=n_mc)
            method = f"mc_block_{n_mc}_{k}models"
    elif null == "within":
        npat = 2 ** n
        if npat <= max_exact:
            stats = np.fromiter(
                (abs(_stat(v, np.array(c, float)))
                 for c in itertools.product([1, -1], repeat=n)),
                dtype=float, count=npat)
            method = f"exact_within_{npat}pat_N{n}"
        else:
            stats = np.fromiter(
                (abs(_stat(v, rng.choice([1., -1.], size=n)))
                 for _ in range(n_mc)), dtype=float, count=n_mc)
            method = f"mc_within_{n_mc}_N{n}"
    else:
        raise ValueError("null must be 'block' or 'within'")

    p = float(np.mean(stats >= obs - 1e-12))
    return p, method


def icc_and_neff(values, groups):
    """
    One-way ICC (between-model variance fraction) and implied effective N.
    Returns dict(icc, n_eff, n_models). NaNs if not computable.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    m = ~np.isnan(v)
    v, g = v[m], g[m]
    n = len(v)
    models = pd.unique(g)
    k = len(models)
    if n < 2 or k < 1:
        return {"icc": np.nan, "n_eff": np.nan, "n_models": k}
    grand = v.mean()
    ssb = ssw = 0.0
    sizes = []
    for mm in models:
        vm = v[g == mm]
        sizes.append(len(vm))
        ssb += len(vm) * (vm.mean() - grand) ** 2
        ssw += np.sum((vm - vm.mean()) ** 2)
    dfb, dfw = k - 1, n - k
    msb = ssb / dfb if dfb > 0 else np.nan
    msw = ssw / dfw if dfw > 0 else np.nan
    m0 = float(np.mean(sizes))
    denom = msb + (m0 - 1) * msw
    if not np.isfinite(denom) or denom == 0:
        icc = np.nan
    else:
        icc = (msb - msw) / denom
        icc = float(min(1.0, max(0.0, icc)))
    n_eff = n / (1 + (m0 - 1) * icc) if np.isfinite(icc) else np.nan
    return {"icc": icc, "n_eff": n_eff, "n_models": k}
