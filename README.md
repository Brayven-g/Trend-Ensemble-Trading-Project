# Crypto trend ensemble (long/flat, 5 coins, spot)

A research project testing whether simple trend-following rules work on crypto, from hypothesis to a
walk-forward-tested strategy. Two hypotheses are tested first; the strategy combines what survived.

| Notebook | Question | Verdict |
|---|---|---|
| [`hypotheses/H0002`](hypotheses/H0002_vol_scaled_time_series_momentum_5_coins.ipynb) | Does a coin that has risen over the last few weeks keep rising (time-series momentum), and does holding it only while it trends beat buy & hold? | Weak-confirmed: right sign on 5/5 coins, long/flat portfolio net Sharpe 1.27 vs 0.97 buy & hold, edge fades after 2022 |
| [`hypotheses/H0012`](hypotheses/H0012_n_day_high_anchoring_and_donchian_breakouts.ipynb) | Do N-day highs and Donchian channel breakouts predict returns beyond H0002's momentum? | Rejected as a new signal (corr 0.87 with H0002), but the Donchian ensemble has ~1/4 the turnover and a smaller drawdown |
| [`strategies/S0004`](strategies/S0004_multi_horizon_trend_ensemble_long_flat_5_coins_spot.ipynb) | Fixed ensemble of trend horizons, vol-sized, long/flat on OKX spot BTC ETH XRP LTC SOL | Viable at 1x; marginal fail on the 1-year drawdown limit |

## The strategy (S0004)

Once a day at the 00:00 UTC close, for each coin:

1. $\sigma$ = standard deviation of the last 60 daily log returns.
2. Trend score $ts$ = share of the lookbacks $L \in \{14, 28, 56\}$ days with a positive return.
3. Donchian score $don$ = average over channel lengths $N$ of a state that turns on above the prior $N$-day high and off below the prior $N/2$-day low.
4. $P = \alpha \cdot don + (1-\alpha) \cdot ts$, target weight $w = P \cdot \min(tv / \sigma, 1)$, resized only when it moves by more than a band.
5. Equal-weight the five coins, long only, never above 1x.

Parameters ($\alpha$, $N$ set, vol target $tv$, band) are chosen on past data only, inside a walk-forward.

## Results (net of OKX spot costs, walk-forward out-of-sample 2020-05 to 2025-09)

| | 5-coin portfolio | Equal-weight buy & hold | BTC alone | BTC buy & hold |
|---|---|---|---|---|
| Sharpe | 1.41 | 1.09 | 1.53 | 1.12 |
| CAGR | 35.9% | 69.4% | 46.2% | 63.0% |
| Max drawdown | -26.9% | -79.9% | -30.0% | -75.6% |
| Annual volatility | 24% | 74% | 27% | 59% |

The edge is risk reduction, not higher return: it keeps much of the upside with a fraction of the drawdown,
and lags buy & hold in strong bull runs.

## How it was tested

- **No lookahead:** the signal on truncated history must equal the full-history signal at every cut (asserted in the notebook).
- **Optimisation on the first half only**, with a plateau-smoothed grid and the deflated Sharpe ratio (Bailey & Lopez de Prado) for multiple testing.
- **Walk-forward:** 10 slices, optimise on 3, trade the next, roll forward.
- **Permutation tests:** fixed parameters on 2,000 shuffled markets, and a stricter version that re-optimises the grid on every permutation.
- **Monte Carlo:** 5,000 block-bootstrap resamples of the out-of-sample returns for drawdown risk.
- **Costs and leverage:** OKX spot fees, spread and slippage; perp costs with real funding for comparison; leverage sweep to liquidation.
- **Holdout:** data from 2025-10-01 is locked out of all research (`qr/db.py`), kept for a single test of the frozen strategy.

## Limitations

- The five coins are today's survivors, and they are highly correlated: effectively one BTC-driven bet.
- The 2020-21 bull market drives much of the total return; 2025+ Sharpe is 0.54 vs 0.79 for buy & hold.
- LTC does not work (net Sharpe 0.30 vs 0.63 buy & hold); ETH's second-half permutation test fails (p = 0.073).
- The optimiser almost always picks $\alpha = 0$ for BTC, so the Donchian leg adds little.
- The 1-year 95th-percentile Monte Carlo drawdown is -26.0%, just past a -25% limit.
- S0004 is 0.96 correlated with an earlier single-lookback version of the strategy (S0002, not included; its
  out-of-sample returns are in `results/S0002_oos.parquet` for the comparison in the notebook).

## Repository layout

```
hypotheses/   H0002, H0012 hypothesis tests
strategies/   S0004 strategy notebook
qr/           backtest engine, costs, walk-forward, Monte Carlo, permutation, statistics, plots
scripts/      download_data.py
results/      saved out-of-sample returns and p-value tables
data/         downloaded market data (created by the script, not committed)
```

## Reproduce

```
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
jupyter lab
```

Data: OKX public API (spot daily candles, perpetual funding) and Coinbase daily candles, via `ccxt`.
No API keys needed.
