"""Rolling walk-forward out-of-sample testing.

Data is split into n_chunks equal slices (default 10 -> 10% each). Each fold optimizes on
`train_chunks` slices, then trades the next slice out-of-sample. The window then slides forward one
slice: 10% new data added, 10% oldest dropped (anchored=True keeps all old data instead).
OOS slices are stitched into one equity curve that never saw its own data during optimization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qr.backtest import metrics, run
from qr.optimize import best_params, grid_search


@dataclass
class WFResult:
    returns: pd.Series      # stitched OOS net returns
    bh_returns: pd.Series   # buy & hold over same OOS span
    position: pd.Series
    folds: pd.DataFrame
    metrics: dict
    bh_metrics: dict

    @property
    def equity(self):
        return (1 + self.returns).cumprod()

    @property
    def bh_equity(self):
        return (1 + self.bh_returns).cumprod()

    def plot(self, title="Walk-forward OOS"):
        from qr.plots import plot_result
        return plot_result(self, title=title)


def walk_forward(df, signal_fn, param_grid: dict, n_chunks: int = 10, train_chunks: int = 3,
                 anchored: bool = False, metric: str = "sharpe", costs=None, leverage: float = 1.0,
                 n_jobs: int = 1, delay: int = 0, **gs_kw) -> WFResult:
    b = np.linspace(0, len(df), n_chunks + 1).astype(int)
    rows, rets, bh, pos = [], [], [], []
    for i in range(train_chunks, n_chunks):
        lo = 0 if anchored else b[i - train_chunks]
        tr_start, tr_end = df.index[lo], df.index[b[i] - 1]
        te_start, te_end = df.index[b[i]], df.index[b[i + 1] - 1]
        hist = df.iloc[: b[i]]  # optimization never sees anything past the train window
        table = grid_search(hist, signal_fn, param_grid, metric=metric, costs=costs,
                            n_jobs=n_jobs, leverage=leverage, start=tr_start, delay=delay, **gs_kw)
        params = best_params(table, param_grid)
        oos = run(df.iloc[: b[i + 1]], signal_fn, params, costs=costs, leverage=leverage, start=te_start, delay=delay)
        rows.append({"fold": i - train_chunks + 1, "train_start": tr_start, "train_end": tr_end,
                     "test_start": te_start, "test_end": te_end, **params,
                     f"is_{metric}": table.iloc[0]["score"], f"oos_{metric}": oos.metrics[metric],
                     "oos_return": oos.metrics["total_return"], "bh_return": oos.bh_metrics["total_return"]})
        rets.append(oos.returns)
        bh.append(oos.bh_returns)
        pos.append(oos.position)
    folds = pd.DataFrame(rows)
    r, bhr, p = pd.concat(rets), pd.concat(bh), pd.concat(pos)
    m = metrics(r, p)
    is_mean = folds[f"is_{metric}"].mean()
    m["wf_efficiency"] = folds[f"oos_{metric}"].mean() / is_mean if is_mean else np.nan  # OOS/IS, >0.5 is decent
    m["pct_folds_profitable"] = (folds["oos_return"] > 0).mean()
    return WFResult(r, bhr, p, folds, m, metrics(bhr))
