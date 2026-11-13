"""Vectorized backtest engine.

Contract for strategy code in notebooks:
    def signal(df: pd.DataFrame, params: dict) -> pd.Series
        Target position per bar in [-1, 1], computed using data up to and including that bar's close.

The engine shifts the signal by one bar (decide at close t, hold during bar t+1), so a signal that
only uses past/current data can never look ahead. Leverage is modeled as constant leverage
rebalanced every bar: bar return = position * leverage * asset return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

SignalFn = Callable[[pd.DataFrame, dict], pd.Series]


def periods_per_year(index: pd.Index) -> float:
    """Bars per year from median bar spacing (crypto trades 365 days)."""
    secs = pd.Series(index).diff().dt.total_seconds().median()
    return 365.25 * 86400 / secs


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def trade_returns(returns: pd.Series, position: pd.Series) -> pd.Series:
    """Compounded return of each trade = a run of same-sign non-zero position. Size changes inside a run
    (vol-scaled sizing) do not start a new trade; flat -> long and long -> short do (audit_S0006 finding 1)."""
    side = np.sign(position.fillna(0))
    trade_id = (side != side.shift()).cumsum()
    mask = side != 0
    if not mask.any():
        return pd.Series(dtype=float)
    return (1 + returns[mask]).groupby(trade_id[mask]).prod() - 1


def metrics(returns: pd.Series, position: pd.Series | None = None, ppy: float | None = None) -> dict:
    r = returns.fillna(0)
    ppy = ppy or periods_per_year(r.index)
    equity = (1 + r).cumprod()
    years = len(r) / ppy
    final = equity.iloc[-1]
    downside = r[r < 0].std() * np.sqrt(ppy)
    mdd = drawdown(equity).min()
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    m = {
        "total_return": final - 1,
        "cagr": cagr,
        "ann_vol": r.std() * np.sqrt(ppy),
        "sharpe": r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else 0.0,
        "sortino": r.mean() * ppy / downside if downside > 0 else 0.0,
        "max_dd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "bars": len(r),
    }
    if position is not None:
        tr = trade_returns(r, position)
        wins, losses = tr[tr > 0].sum(), -tr[tr < 0].sum()
        m.update({
            "trades": len(tr),
            "win_rate": (tr > 0).mean() if len(tr) else np.nan,
            "profit_factor": wins / losses if losses > 0 else np.nan,
            "avg_trade": tr.mean() if len(tr) else np.nan,
            "exposure": (position != 0).mean(),
            "turnover_per_year": position.diff().abs().sum() / years,
        })
    return m


@dataclass
class Result:
    returns: pd.Series          # net strategy returns per bar
    gross: pd.Series            # before costs
    position: pd.Series         # position held during each bar (already shifted), before leverage
    equity: pd.Series
    bh_returns: pd.Series
    bh_equity: pd.Series
    leverage: float = 1.0
    liquidated_at: pd.Timestamp | None = None
    metrics: dict = field(default_factory=dict)
    bh_metrics: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"strategy": self.metrics, "buy_hold": self.bh_metrics})

    def plot(self, title: str = "", log: bool = True):
        from qr.plots import plot_result
        return plot_result(self, title=title, log=log)


def run(df: pd.DataFrame, signal_fn: SignalFn, params: dict | None = None, costs=None,
        leverage: float = 1.0, mmr: float = 0.004, start=None, end=None, delay: int = 0) -> Result:
    """Backtest signal_fn on df (needs 'close'; 'high'/'low' used for liquidation checks).

    start/end: evaluate only this window, but compute signals on all of df so indicators are warm.
    mmr: maintenance margin ratio for the liquidation check (OKX BTC perp tier 1 ~0.4%).
    delay: extra bars between decision and fill (default 0 = fill at the signal bar's close; optimistic).
        Report a delay=1 row for every strategy - the edge must survive it.
    """
    params = params or {}
    target = signal_fn(df, params).reindex(df.index).fillna(0).clip(-1, 1)
    held = target.shift(1 + delay).fillna(0)
    ret = df["close"].pct_change().fillna(0)

    sl = slice(start, end)
    held, ret, df_w = held.loc[sl].copy(), ret.loc[sl].copy(), df.loc[sl]
    held.iloc[0] = 0.0  # always start flat at window start
    ret.iloc[0] = 0.0
    exp = held * leverage
    gross = exp * ret
    cost = costs.bar_costs(df_w, held, leverage) if costs is not None else 0.0
    net = gross - cost

    # Liquidation: worst intrabar move against the position vs maintenance margin
    liquidated_at = None
    if {"high", "low"} <= set(df_w.columns):
        prev = df_w["close"].shift(1)
        adverse = np.where(held > 0, df_w["low"] / prev - 1,
                           np.where(held < 0, -(df_w["high"] / prev - 1), 0.0))
        adverse = pd.Series(adverse, index=held.index).fillna(0)
        eq_frac = 1 + exp.abs() * adverse  # equity left / equity at bar open
        liq = (exp.abs() > 0) & (eq_frac <= mmr * exp.abs() * (1 + adverse))
        if liq.any():
            liquidated_at = liq.idxmax()
            net = net.copy()
            net.loc[liquidated_at] = -1.0
            net.loc[net.index > liquidated_at] = 0.0
    net = net.clip(lower=-1.0)

    ppy = periods_per_year(df_w.index)
    res = Result(returns=net, gross=gross, position=held, equity=(1 + net).cumprod(),
                 bh_returns=ret, bh_equity=(1 + ret).cumprod(), leverage=leverage,
                 liquidated_at=liquidated_at)
    res.metrics = metrics(net, held, ppy)
    res.metrics["liquidated"] = liquidated_at is not None
    res.bh_metrics = metrics(ret, None, ppy)
    return res
