"""Realistic trading frictions: fees, spread, slippage, funding/borrow, leverage + liquidation.

Defaults approximate OKX regular-tier USDT perpetuals (taker 0.05%, maker 0.02%).
OKX spot regular tier is taker 0.10% / maker 0.08% -> use CostModel.okx_spot().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    fee: float = 0.0005               # per side, fraction of notional traded
    half_spread_bps: float = 0.5      # paid on every trade
    slippage_bps: float = 1.0         # extra market impact per trade
    instrument: str = "perp"          # "perp" (funding) or "spot" (borrow for shorts / leverage)
    funding: pd.Series | None = None  # perp funding rates indexed by funding timestamp
    funding_rate_8h: float = 0.0001   # used when no funding series given (~11%/yr, long pays)
    borrow_annual: float = 0.10       # spot margin borrow rate

    @classmethod
    def okx_perp(cls, **kw):
        return cls(fee=0.0005, instrument="perp", **kw)

    @classmethod
    def okx_spot(cls, **kw):
        return cls(fee=0.0010, instrument="spot", **kw)

    @classmethod
    def zero(cls):
        return cls(fee=0, half_spread_bps=0, slippage_bps=0, funding_rate_8h=0, borrow_annual=0)

    def bar_costs(self, df: pd.DataFrame, held: pd.Series, leverage: float = 1.0) -> pd.Series:
        """Cost per bar as a fraction of equity."""
        exp = held * leverage
        traded = exp.diff().abs()
        traded.iloc[0] = abs(exp.iloc[0])
        trading = traded * (self.fee + (self.half_spread_bps + self.slippage_bps) / 1e4)

        bar_hours = pd.Series(df.index).diff().dt.total_seconds().median() / 3600
        if self.instrument == "perp":
            if self.funding is not None:
                # assign each funding print to the bar that contains it
                f = self.funding.dropna()
                f = f[(f.index >= df.index[0]) & (f.index <= df.index[-1])]
                pos = df.index.searchsorted(f.index, side="right") - 1
                rate = pd.Series(f.values, index=df.index[pos]).groupby(level=0).sum()
                rate = rate.reindex(df.index).fillna(0)
            else:
                rate = pd.Series(self.funding_rate_8h * bar_hours / 8, index=df.index)
            carry = exp * rate  # long pays positive funding, short receives
        else:
            per_bar = self.borrow_annual * bar_hours / (365.25 * 24)
            borrowed = np.where(exp > 0, np.maximum(exp - 1, 0), exp.abs())
            carry = pd.Series(borrowed * per_bar, index=df.index)
        return trading + carry


def leverage_sweep(df, signal_fn, params, costs: CostModel | None = None,
                   levels=(1, 2, 3, 5, 10, 20, 30, 50), mmr: float = 0.004, **run_kw) -> pd.DataFrame:
    """Rerun the backtest at each leverage level. Returns one row of metrics per level;
    equity curves are in result.attrs['equity']."""
    from qr.backtest import run
    rows, curves = {}, {}
    for lev in levels:
        r = run(df, signal_fn, params, costs=costs, leverage=lev, mmr=mmr, **run_kw)
        rows[lev] = {**r.metrics, "liquidated_at": r.liquidated_at}
        curves[lev] = r.equity
    out = pd.DataFrame(rows).T
    out.index.name = "leverage"
    out.attrs["equity"] = pd.DataFrame(curves)
    return out


def okx_funding(coin: str, start: str = "2017-01-01", fill_8h: float = 0.0001) -> pd.Series:
    """Real OKX perp funding prints for coin (archive starts 2021-08), back-filled with fill_8h
    every 8h before the first real print so pre-2021 backtests still pay a realistic carry."""
    from qr.db import q
    real = q(f"SELECT ts, funding_rate FROM funding WHERE symbol = '{coin}-USDT-USDT'")["funding_rate"]
    synth_idx = pd.date_range(pd.Timestamp(start, tz=real.index.tz), real.index[0], freq="8h", inclusive="left")
    return pd.concat([pd.Series(fill_8h, index=synth_idx), real]).rename("funding_rate")


def stress(costs: CostModel, mult: float = 2.0) -> CostModel:
    """Same cost model with fees, spread and slippage x mult (funding/borrow unchanged) - report a x2 row."""
    from dataclasses import replace
    return replace(costs, fee=costs.fee * mult, half_spread_bps=costs.half_spread_bps * mult,
                   slippage_bps=costs.slippage_bps * mult)


def capacity(df: pd.DataFrame, position: pd.Series, aum: float, max_pct: float = 0.01) -> dict:
    """Share of trades whose size (|dposition| x aum, in quote currency) exceeds max_pct of that bar's
    dollar volume. > 5% of trades over the cap -> size is too big for the market at this AUM."""
    dollar_vol = (df["close"] * df["volume"]).reindex(position.index)
    trade = position.diff().abs() * aum
    traded = trade > 0
    over = (trade > max_pct * dollar_vol) & traded
    return {"aum": aum, "max_pct_of_bar_volume": max_pct, "trades": int(traded.sum()),
            "share_over_cap": float(over.sum() / max(traded.sum(), 1)),
            "median_trade_pct_of_volume": float((trade[traded] / dollar_vol[traded]).median() * 100)}
