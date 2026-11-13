"""Standard plots used by every notebook (seaborn styled).

Importing this module applies the house theme. Colors:
  strategy = blue, buy & hold = gray, losses/VaR/liquidation = red, extra series follow PALETTE order.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from qr.backtest import drawdown

# fixed categorical order - assign in this order, never cycle
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STRAT, BH, NEG, POS = "#2a78d6", "#8a8984", "#e34948", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# diverging: red (bad) <- neutral gray -> blue (good)
DIVERGING = sns.blend_palette([NEG, "#f0efec", STRAT], as_cmap=True)
SEQUENTIAL = sns.blend_palette(["#cde2fb", "#2a78d6", "#0d366b"], as_cmap=True)


def set_theme():
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=0.95, rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.8, "legend.frameon": False, "figure.dpi": 110,
    })


set_theme()


def _pct(ax, axis="y", decimals=0):
    fmt = mtick.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _end_label(ax, series: pd.Series, text: str, color: str):
    """Direct label at the last point of a line."""
    ax.annotate(f" {text}", (series.index[-1], series.iloc[-1]), color=color,
                fontsize=9, fontweight="bold", va="center", annotation_clip=False)


def plot_result(res, title: str = "", log: bool = True):
    """Strategy equity vs buy & hold, drawdowns, and position."""
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 0.7], "hspace": 0.08})
    sns.lineplot(x=res.bh_equity.index, y=res.bh_equity.values, ax=ax[0], color=BH, lw=1.3, label="buy & hold")
    sns.lineplot(x=res.equity.index, y=res.equity.values, ax=ax[0], color=STRAT, lw=2, label="strategy")
    _end_label(ax[0], res.equity, f"{res.equity.iloc[-1]:.1f}x", STRAT)
    _end_label(ax[0], res.bh_equity, f"{res.bh_equity.iloc[-1]:.1f}x", BH)
    if log:
        ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax[0].set_ylabel("equity")
    ax[0].set_title(title)
    ax[0].legend(loc="upper left")

    dd, bdd = drawdown(res.equity), drawdown(res.bh_equity)
    ax[1].fill_between(dd.index, dd, 0, color=STRAT, alpha=0.35, lw=0)
    ax[1].plot(dd, color=STRAT, lw=1)
    ax[1].plot(bdd, color=BH, lw=1)
    ax[1].set_ylabel("drawdown")
    _pct(ax[1])

    ax[2].fill_between(res.position.index, res.position, 0, step="post", color=STRAT, alpha=0.5, lw=0)
    ax[2].set_ylabel("position")
    ax[2].set_ylim(min(-1.05, res.position.min() - 0.05) if (res.position < 0).any() else -0.05, 1.05)
    sns.despine(fig)
    return fig


def plot_heatmap(table: pd.DataFrame, x: str, y: str, value: str = "score", title: str = ""):
    """2-parameter optimization surface. Look for broad blue plateaus, not single bright cells."""
    pv = table.pivot_table(index=y, columns=x, values=value).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pv, annot=True, fmt=".2f", cmap=DIVERGING, center=np.nanmedian(pv.values),
                linewidths=2, linecolor=SURFACE, cbar_kws={"label": value, "shrink": 0.8},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title(title or value)
    return fig


def plot_mc(mc, title=""):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"wspace": 0.25})
    for p in mc.sample_paths:
        ax[0].plot(p, color=BH, alpha=0.12, lw=0.6)
    step = max(len(mc.actual_equity) // mc.sample_paths.shape[1], 1)
    lo, hi = np.quantile(mc.sample_paths, [0.05, 0.95], axis=0)
    ax[0].fill_between(range(len(lo)), lo, hi, color=STRAT, alpha=0.12, label="5-95% band")
    ax[0].plot(mc.actual_equity.values[::step], color=STRAT, lw=2, label="actual")
    ax[0].set_yscale("log")
    ax[0].set_title("Simulated equity paths")
    ax[0].set_xlabel("time")
    ax[0].legend(loc="upper left")

    var_key = [k for k in mc.table.index if k.startswith("VaR 95%")][0]
    sns.histplot(mc.final_returns, bins=80, color=STRAT, ax=ax[1], edgecolor=SURFACE, linewidth=0.5)
    ax[1].axvline(-mc.table.loc[var_key, "value"], color=NEG, ls="--", lw=1.5, label="VaR 95%")
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_title("Total return distribution")
    _pct(ax[1], "x")
    ax[1].legend()

    sns.histplot(mc.max_dds, bins=80, color=NEG, ax=ax[2], edgecolor=SURFACE, linewidth=0.5)
    ax[2].axvline(np.quantile(mc.max_dds, 0.05), color=INK, ls="--", lw=1.2, label="95th pct")
    ax[2].set_title("Max drawdown distribution")
    _pct(ax[2], "x")
    ax[2].legend()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def plot_perm(pr, title=""):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(pr.perms, bins=80, color=BH, ax=ax, edgecolor=SURFACE, linewidth=0.5, label="shuffled markets")
    ax.axvline(pr.real, color=STRAT, lw=2.5, label=f"real strategy ({pr.real:.2f})")
    ax.axvline(np.quantile(pr.perms, 0.95), color=INK2, ls="--", lw=1, label="95th pct of shuffled")
    ax.set_title(f"{title}   p = {pr.p_value:.4f}")
    ax.set_xlabel(pr.metric)
    ax.legend()
    return fig


def plot_leverage(sweep: pd.DataFrame, title="Leverage sweep (net of costs)"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"wspace": 0.2})
    eq = sweep.attrs["equity"]
    colors = sns.color_palette(SEQUENTIAL(np.linspace(0.25, 1, len(eq.columns))))
    for c, lev in zip(colors, eq.columns):
        ax[0].plot(eq[lev].clip(lower=1e-3), color=c, lw=1.3, label=f"{lev}x")
    ax[0].set_yscale("log")
    ax[0].set_title("Equity by leverage")
    ax[0].legend(ncol=2, fontsize=8, title="leverage")

    d = pd.DataFrame({"leverage": sweep.index.astype(str), "cagr": sweep["cagr"].astype(float),
                      "outcome": np.where(sweep["liquidated"].astype(bool), "liquidated", "survived")})
    sns.barplot(d, x="leverage", y="cagr", hue="outcome", palette={"survived": STRAT, "liquidated": NEG},
                dodge=False, ax=ax[1], edgecolor=SURFACE, linewidth=2)
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_title("CAGR by leverage")
    _pct(ax[1])
    ax[1].legend(title="")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def plot_event(close: pd.Series, event: pd.Series, horizon: int = 1, title=""):
    """Distribution of forward returns after events vs baseline."""
    from qr.stats import forward_returns
    fwd = forward_returns(close, horizon)
    ev = event.reindex(close.index).fillna(False).astype(bool)
    d = pd.DataFrame({"fwd": fwd, "group": np.where(ev, "after event", "baseline")}).dropna()
    lo, hi = d["fwd"].quantile([0.005, 0.995])
    d = d[d["fwd"].between(lo, hi)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.2})
    pal = {"baseline": BH, "after event": STRAT}
    sns.kdeplot(d, x="fwd", hue="group", palette=pal, fill=True, common_norm=False, alpha=0.3, lw=1.8, ax=ax[0])
    for g, c in pal.items():
        ax[0].axvline(d.loc[d.group == g, "fwd"].mean(), color=c, ls="--", lw=1.2)
    ax[0].set_title(title or f"{horizon}-bar forward returns (dashed = mean)")
    _pct(ax[0], "x", 1)
    sns.boxplot(d, x="group", y="fwd", hue="group", palette=pal, ax=ax[1], width=0.5,
                showfliers=False, linewidth=1.2)
    ax[1].set_title("Spread")
    ax[1].set_xlabel("")
    _pct(ax[1], "y", 1)
    return fig


def plot_corr(returns: pd.DataFrame, title="Strategy return correlation"):
    fig, ax = plt.subplots(figsize=(1.2 * len(returns.columns) + 3, 1 * len(returns.columns) + 2))
    c = returns.corr()
    sns.heatmap(c, annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                linewidths=2, linecolor=SURFACE, mask=np.triu(np.ones_like(c, bool), 1), ax=ax)
    ax.set_title(title)
    return fig
