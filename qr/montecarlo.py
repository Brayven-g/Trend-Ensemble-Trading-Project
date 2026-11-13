"""Monte Carlo resampling of strategy returns -> VaR / CVaR / drawdown distributions.

method:
  "shuffle"   - reorder the same returns (final return identical, tests path/drawdown luck)
  "bootstrap" - sample with replacement (varies final return and drawdown)
  "block"     - stationary-ish block bootstrap; keeps volatility clustering / autocorrelation
Run on OOS (walk-forward) returns, not in-sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MCResult:
    final_returns: np.ndarray
    max_dds: np.ndarray
    sample_paths: np.ndarray  # (n_paths, points) downsampled equity curves for plotting
    actual_equity: pd.Series
    table: pd.DataFrame

    def plot(self, title="Monte Carlo"):
        from qr.plots import plot_mc
        return plot_mc(self, title)


def _resample(r: np.ndarray, n: int, method: str, block: int, rng) -> np.ndarray:
    T = len(r)
    if method == "shuffle":
        return np.array([rng.permutation(r) for _ in range(n)])
    if method == "bootstrap":
        return r[rng.integers(0, T, (n, T))]
    if method == "block":
        n_blocks = int(np.ceil(T / block))
        starts = rng.integers(0, T - block + 1, (n, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
        return r[idx]
    raise ValueError(method)


def monte_carlo(returns: pd.Series, n: int = 5000, method: str = "bootstrap", block: int = 20,
                ruin_dd: float = 0.5, seed: int = 0, chunk: int = 250, n_paths: int = 200) -> MCResult:
    r = returns.fillna(0).values
    rng = np.random.default_rng(seed)
    finals, dds, paths = [], [], []
    step = max(len(r) // 500, 1)
    for start in range(0, n, chunk):
        sims = _resample(r, min(chunk, n - start), method, block, rng)
        eq = np.cumprod(1 + sims, axis=1)
        finals.append(eq[:, -1] - 1)
        dds.append((eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1))
        if sum(len(p) for p in paths) < n_paths:
            paths.append(eq[:, ::step])
    finals, dds = np.concatenate(finals), np.concatenate(dds)
    paths = np.concatenate(paths)[:n_paths]

    def var(x, a):
        return -np.quantile(x, 1 - a)

    def cvar(x, a):
        q = np.quantile(x, 1 - a)
        return -x[x <= q].mean()

    rows = {
        "VaR 95% (loss, <0 = gain)": var(finals, 0.95),
        "CVaR 95%": cvar(finals, 0.95),
        "VaR 99%": var(finals, 0.99),
        "CVaR 99%": cvar(finals, 0.99),
        "median total return": np.median(finals),
        "mean total return": finals.mean(),
        "P(loss)": (finals < 0).mean(),
        "median max DD": np.median(dds),
        "95th pct max DD": np.quantile(dds, 0.05),
        "99th pct max DD": np.quantile(dds, 0.01),
        f"P(max DD worse than -{ruin_dd:.0%})": (dds <= -ruin_dd).mean(),
        "per-bar hist VaR 95%": var(r, 0.95),
        "per-bar hist VaR 99%": var(r, 0.99),
    }
    table = pd.DataFrame({"value": rows})
    return MCResult(finals, dds, paths, (1 + returns.fillna(0)).cumprod(), table)


def mc_dd_horizon(returns: pd.Series, horizon: int = 365, n: int = 2000, block: int = 20, q: float = 0.95,
                  seed: int = 0) -> float:
    """Max drawdown not exceeded with probability q over `horizon` bars (block bootstrap), as a negative number.
    Paper-trade gate: mc_dd_horizon(daily net OOS returns, 365) >= -0.25.
    Fixed horizon so strategies with different backtest lengths are comparable (full-path DD deepens with length)."""
    r = returns.dropna().values
    sims = _resample(r, n, "block", block, np.random.default_rng(seed))[:, :horizon] if len(r) >= horizon else None
    if sims is None:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(r) - block + 1, (n, int(np.ceil(horizon / block))))
        sims = r[(starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :horizon]]
    eq = np.cumprod(1 + sims, axis=1)
    return float(np.quantile((eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1), 1 - q))
