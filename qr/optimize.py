"""Parameter optimization with overfitting guards.

- Search a grid (or random subset) on in-sample data only.
- Pick the best *plateau*, not the best peak: each combo is scored by the average metric of itself
  and its grid neighbours, so isolated lucky spikes lose to robust regions.
- Deflated Sharpe Ratio corrects the best Sharpe for the number of trials tried.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from qr.backtest import run, periods_per_year


def grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]


def _eval(df, signal_fn, params, costs, run_kw):
    r = run(df, signal_fn, params, costs=costs, **run_kw)
    return {**params, **r.metrics}


def grid_search(df, signal_fn, param_grid: dict, metric: str = "sharpe", costs=None,
                random_n: int | None = None, seed: int = 0, n_jobs: int = 1,
                min_trades: int = 10, **run_kw) -> pd.DataFrame:
    """Evaluate every combo (or random_n of them). Returns table sorted by plateau-smoothed metric."""
    combos = grid(param_grid)
    if random_n and random_n < len(combos):
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), random_n, replace=False)]
    if n_jobs == 1:
        rows = [_eval(df, signal_fn, p, costs, run_kw) for p in combos]
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(_eval)(df, signal_fn, p, costs, run_kw) for p in combos)
    table = pd.DataFrame(rows)
    score = table[metric].astype(float).where(table["trades"] >= min_trades, np.nan)
    table["score"] = score
    table["smoothed"] = _plateau(table, list(param_grid), param_grid, "score")
    return table.sort_values(["smoothed", "score"], ascending=False, na_position="last", kind="stable").reset_index(drop=True)   # ties in the plateau score (coarse grids) -> own score decides


def _plateau(table, keys, param_grid, col) -> pd.Series:
    """Average of col over each combo and its +-1 grid-step neighbours."""
    idx = np.column_stack([table[k].map({v: i for i, v in enumerate(param_grid[k])}).values for k in keys])
    vals = table[col].values.astype(float)
    out = np.empty(len(table))
    for i in range(len(table)):
        nb = np.all(np.abs(idx - idx[i]) <= 1, axis=1)
        out[i] = np.nanmean(vals[nb]) if np.isfinite(vals[i]) else np.nan
    return pd.Series(out, index=table.index)


def best_params(table: pd.DataFrame, param_grid: dict) -> dict:
    row = table.iloc[0]
    return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in param_grid}


def deflated_sharpe(best_returns: pd.Series, trial_sharpes, ppy: float | None = None,
                    n_trials: int | None = None) -> dict:
    """Bailey & Lopez de Prado (2014). Probability the best Sharpe is real given N trials.

    best_returns: per-bar returns of the selected strategy; trial_sharpes: annualized Sharpe of all trials.
    n_trials: total number of configs tried across the whole research effort; overrides len(trial_sharpes)
        when larger, so the hurdle reflects everything searched, not just this notebook's grid.
    """
    r = best_returns.dropna()
    ppy = ppy or periods_per_year(r.index)
    sr = r.mean() / r.std()                                   # per-bar Sharpe
    srs = np.asarray(trial_sharpes, float)
    srs = srs[np.isfinite(srs)] / np.sqrt(ppy)
    n = max(len(srs), int(n_trials or 0), 1)
    emc = 0.5772156649
    v = srs.var() if n > 1 else 0.0
    sr0 = np.sqrt(v) * ((1 - emc) * stats.norm.ppf(1 - 1 / n) + emc * stats.norm.ppf(1 - 1 / (n * np.e))) if n > 1 else 0.0
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom)
    return {"sharpe_ann": sr * np.sqrt(ppy), "sr0_ann": sr0 * np.sqrt(ppy), "n_trials": n, "dsr": dsr}
