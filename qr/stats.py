"""Hypothesis tests for "when X happens, is Y different from normal?" questions.

Typical use:
    fwd = forward_returns(df.close, h)
    event_table(df.close, event_mask, horizons=[1, 3, 7])
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests


def forward_returns(close: pd.Series, h: int = 1) -> pd.Series:
    """Return from close t to close t+h (aligned at t). Uses future data -> only for analysis, never signals."""
    return close.shift(-h) / close - 1


def _default_lags(n: int, h: int = 1) -> int:
    """Newey-West rule of thumb, raised to 3h for h-bar overlapping samples (lags = h over-rejects: 9-10% at nominal 5%)."""
    return max(int(4 * (n / 100) ** (2 / 9)), 3 * h if h > 1 else 0)


def hac_mean_test(x: pd.Series, lags: int | None = None, h: int = 1) -> tuple[float, float]:
    """t-test of mean != 0 with Newey-West (HAC) errors. Needed when samples overlap or are autocorrelated.
    h = overlap of the samples in bars, used only when lags is None."""
    x = pd.Series(x).dropna()
    lags = lags if lags is not None else _default_lags(len(x), h)
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def robust_mean_test(x: pd.Series, h: int = 1, block: int | None = None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean != 0 by HAC and by centred moving-block bootstrap; p is the less favourable of the two (two-sided)."""
    x = pd.Series(x).dropna()
    t, _ = hac_mean_test(x, h=h)
    v = x.values
    p_hac = float(2 * stats.t.sf(abs(t), max(len(v) - 1, 1)))  # t(n-1) reference: the normal one over-rejects at small n
    L = min(block or max(3 * h, 8), max(len(v) // 5, 1))       # at least 5 blocks
    nb = int(np.ceil(len(v) / L))
    starts = np.random.default_rng(seed).integers(0, len(v), (n_boot, nb))  # circular blocks, truncated to n
    idx = ((starts[:, :, None] + np.arange(L)) % len(v)).reshape(n_boot, -1)[:, :len(v)]
    m = (v - v.mean())[idx].mean(1)
    p_boot = float((1 + (np.abs(m) >= abs(v.mean())).sum()) / (n_boot + 1))
    return {"mean": float(v.mean()), "n": len(v), "t": t, "p_hac": p_hac, "p_boot": p_boot, "p": max(p_hac, p_boot)}


def hac_diff_test(a: pd.Series, b: pd.Series, lags: int | None = None) -> tuple[float, float]:
    """HAC t-test of mean(a) != mean(b): regress pooled values on [1, is_a] in time order.

    Accounts for sampling error in BOTH means (testing a - mean(b) treats mean(b) as known,
    which is overconfident when b is not much larger than a) and for autocorrelation.
    """
    a, b = pd.Series(a).dropna(), pd.Series(b).dropna()
    y = pd.concat([a, b])
    d = pd.Series(np.r_[np.ones(len(a)), np.zeros(len(b))], index=y.index)
    try:
        order = np.argsort(y.index.values, kind="stable")
        y, d = y.iloc[order], d.iloc[order]
    except TypeError:
        pass
    lags = lags if lags is not None else int(4 * (len(y) / 100) ** (2 / 9))
    X = np.column_stack([np.ones(len(y)), d.values])
    fit = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[1]), float(fit.pvalues[1])


def bootstrap_diff_ci(a, b, n: int = 5000, ci: float = 0.95, seed: int = 0, block: int = 1) -> tuple[float, float]:
    """Bootstrap CI for mean(a) - mean(b). block > 1 = moving-block bootstrap (each series resampled in contiguous
    blocks of that length, in time order), needed when samples overlap (forward returns with h > 1) or cluster;
    block = 1 is the iid bootstrap, which is too narrow for dependent samples. Blocks are circular and the resample is
    truncated to the original length (the non-wrapping version covered 60% at n=18, block 8)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)

    def means(x, k):
        L = min(max(int(block), 1), max(len(x) // 5, 1))  # at least 5 blocks, else the CI is far too narrow
        if L == 1:
            return rng.choice(x, (k, len(x))).mean(1)
        n_ = len(x)
        nb = int(np.ceil(n_ / L))
        starts = rng.integers(0, n_, (k, nb))  # circular blocks, truncated to n (2026-11-07 fix: no-wrap + nb*L > n under-covered at small n)
        idx = ((starts[:, :, None] + np.arange(L)) % n_).reshape(k, -1)[:, :n_]
        return x[idx].mean(1)

    chunk = max(1, int(2e7 // (max(len(a), len(b)) + max(int(block), 1))))  # bound memory (~160MB per draw block)
    diffs = np.concatenate([means(a, k) - means(b, k)
                            for k in [chunk] * (n // chunk) + ([n % chunk] if n % chunk else [])])
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def compare(cond: pd.Series, base: pd.Series, hac_lags: int | None = None, n_boot: int = 5000) -> dict:
    """Compare conditional sample vs baseline with a battery of tests."""
    a, b = pd.Series(cond).dropna(), pd.Series(base).dropna()
    if len(a) < 30:
        warnings.warn(f"only {len(a)} event samples - results unreliable")
    pooled = np.sqrt((a.var() * (len(a) - 1) + b.var() * (len(b) - 1)) / (len(a) + len(b) - 2))
    lo, hi = bootstrap_diff_ci(a, b, n_boot, block=hac_lags or 1)
    t_hac, p_hac = hac_diff_test(a, b, hac_lags)
    return {
        "n_event": len(a), "n_base": len(b),
        "mean_event": a.mean(), "mean_base": b.mean(), "diff": a.mean() - b.mean(),
        "median_event": a.median(), "median_base": b.median(),
        "hit_rate_event": (a > 0).mean(), "hit_rate_base": (b > 0).mean(),
        "cohen_d": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
        "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
        "hac_t": t_hac, "hac_p": p_hac,
        "mannwhitney_p": stats.mannwhitneyu(a, b).pvalue,
        "ks_p": stats.ks_2samp(a, b).pvalue,
        "boot_ci_lo": lo, "boot_ci_hi": hi,
    }


def event_table(close: pd.Series, event: pd.Series, horizons=(1, 3, 5, 10),
                p_col: str = "hac_p", method: str = "fdr_bh") -> pd.DataFrame:
    """Event study across horizons. Baseline = all non-event bars.

    Overlapping forward windows (h > 1) inflate naive t-stats; hac_p corrects with lags = 3h (and block = 3h in the CI).
    p-values are adjusted for testing multiple horizons (Benjamini-Hochberg by default).
    """
    event = event.reindex(close.index).fillna(False).astype(bool)
    rows = {}
    for h in horizons:
        fwd = forward_returns(close, h)
        rows[h] = compare(fwd[event], fwd[~event], hac_lags=3 * h)
    out = pd.DataFrame(rows).T
    out.index.name = "horizon"
    out[f"{p_col}_adj"] = adjust_pvalues(out[p_col].astype(float), method)
    return out


def adjust_pvalues(pvals, method: str = "fdr_bh"):
    """Multiple-testing correction: 'bonferroni', 'holm', 'fdr_bh'."""
    return multipletests(np.asarray(pvals, float), method=method)[1]


def decile_table(feature: pd.Series, fwd: pd.Series, q: int = 10) -> pd.DataFrame:
    """Mean forward return by feature quantile - checks for monotonic relationships."""
    d = pd.DataFrame({"f": feature, "y": fwd}).dropna()
    d["bucket"] = pd.qcut(d["f"], q, labels=False, duplicates="drop")
    g = d.groupby("bucket")["y"]
    out = pd.DataFrame({"mean": g.mean(), "count": g.size(), "t": g.mean() / (g.std() / np.sqrt(g.size()))})
    out.attrs["spearman"] = stats.spearmanr(d["f"], d["y"])
    return out
