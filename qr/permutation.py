"""Monte Carlo permutation test (Masters, "Permutation and Randomization Tests for Trading System Development").

Each permutation rebuilds a synthetic price series from shuffled bar returns: overnight gaps
(open vs prior close) and intrabar shape (high/low/close relative to open) are shuffled separately,
so the synthetic market has the same return distribution and volatility but any real temporal
pattern is destroyed. The strategy is rerun on each one.

p-value = share of permutations whose metric >= real metric. Small p -> edge unlikely to be luck.
With optimize=True each permutation is re-optimized too, which also tests for data-mining bias.
Other columns (volume, funding, ...) are left in place, so their link to price is broken as well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from qr.backtest import run
from qr.optimize import grid_search


def permute_bars(df: pd.DataFrame, seed: int | None = None, start_index: int = 0) -> pd.DataFrame:
    """Synthetic OHLC with shuffled bars. Bars before start_index are kept intact (warm-up)."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (np.log(df[k].values) for k in ("open", "high", "low", "close"))
    gap = o[1:] - c[:-1]
    hi, lo, cl = h - o, l - o, c - o
    s = start_index
    n = len(df) - 1 - s
    p_gap = rng.permutation(n)
    p_bar = rng.permutation(n)
    new_gap = gap[s:][p_gap]
    new_hi, new_lo, new_cl = hi[s + 1:][p_bar], lo[s + 1:][p_bar], cl[s + 1:][p_bar]

    O, H, L, C = o.copy(), h.copy(), l.copy(), c.copy()
    C[s + 1:] = c[s] + np.cumsum(new_gap + new_cl)
    O[s + 1:] = C[s:-1] + new_gap
    H[s + 1:], L[s + 1:] = O[s + 1:] + new_hi, O[s + 1:] + new_lo
    out = df.copy()
    out["open"], out["high"], out["low"], out["close"] = np.exp(O), np.exp(H), np.exp(L), np.exp(C)
    return out


@dataclass
class PermResult:
    real: float
    perms: np.ndarray
    p_value: float
    metric: str
    perm_total_returns: np.ndarray

    def summary(self) -> pd.Series:
        return pd.Series({
            f"real {self.metric}": self.real,
            f"perm mean {self.metric}": self.perms.mean(),
            f"perm 95th pct {self.metric}": np.quantile(self.perms, 0.95),
            "p_value": self.p_value,
            "EV total return (null)": self.perm_total_returns.mean(),
            "n_permutations": len(self.perms),
        })

    def plot(self, title="Permutation test"):
        from qr.plots import plot_perm
        return plot_perm(self, title)


def _one(df, signal_fn, params, param_grid, metric, costs, seed, run_kw):
    p = permute_bars(df, seed)
    if param_grid is not None:
        t = grid_search(p, signal_fn, param_grid, metric=metric, costs=costs, **run_kw)
        return t.iloc[0]["score"], t.iloc[0]["total_return"]
    r = run(p, signal_fn, params, costs=costs, **run_kw)
    return r.metrics[metric], r.metrics["total_return"]


def permutation_test(df, signal_fn, params: dict | None = None, param_grid: dict | None = None,
                     n: int = 5000, metric: str = "sharpe", costs=None, n_jobs: int = -1,
                     seed: int = 0, **run_kw) -> PermResult:
    """Pass params for a fixed-parameter test, or param_grid to re-optimize on every permutation (slow)."""
    if param_grid is not None:
        real = grid_search(df, signal_fn, param_grid, metric=metric, costs=costs, **run_kw).iloc[0]["score"]
    else:
        real = run(df, signal_fn, params, costs=costs, **run_kw).metrics[metric]
    out = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_one)(df, signal_fn, params, param_grid, metric, costs, seed + i + 1, run_kw) for i in range(n))
    perms = np.array([o[0] for o in out], float)
    tot = np.array([o[1] for o in out], float)
    p = (np.sum(perms >= real) + 1) / (n + 1)
    return PermResult(real, perms, p, metric, tot)
