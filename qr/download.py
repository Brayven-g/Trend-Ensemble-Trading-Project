"""Download crypto data from OKX via ccxt into data/ as partitioned Parquet.

Incremental: re-running only fetches bars after the last stored timestamp.
Public market data needs no keys; keys from .env (OKX_API_KEY, OKX_API_SECRET,
OKX_API_PASSPHRASE) are used if present.
"""
from __future__ import annotations

import os
import time

import ccxt
import pandas as pd
from dotenv import load_dotenv

from qr.db import DATA_DIR

load_dotenv()


def exchange() -> ccxt.okx:
    cfg = {"enableRateLimit": True}
    if os.getenv("OKX_API_KEY"):
        cfg.update(apiKey=os.getenv("OKX_API_KEY"), secret=os.getenv("OKX_API_SECRET"),
                   password=os.getenv("OKX_API_PASSPHRASE"))
    return ccxt.okx(cfg)


def _sym_dir(symbol: str) -> str:
    return symbol.replace("/", "-").replace(":", "-")


def _path(kind: str, symbol: str, tf: str | None = None):
    p = DATA_DIR / kind
    if tf:
        p = p / f"tf={tf}"
    return p / f"symbol={_sym_dir(symbol)}" / "data.parquet"


def _merge_write(path, new: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new])
    new = new.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    tmp = path.with_suffix(".tmp")  # write-then-rename: readers never see a half-written file
    new.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return new


def _last_ts(path) -> int | None:
    if not path.exists():
        return None
    return int(pd.read_parquet(path, columns=["ts"])["ts"].max().timestamp() * 1000)


def ohlcv(symbol: str, tf: str, since: str = "2017-01-01", ex=None, kind: str = "ohlcv",
          limit: int = 100) -> pd.DataFrame:
    """Fetch OHLCV bars for symbol (e.g. 'BTC/USDT' spot, 'BTC/USDT:USDT' perp).

    Non-OKX sources: pass ex (any ccxt exchange) and kind, e.g. kind="ohlcv_coinbase".
    """
    ex = ex or exchange()
    path = _path(kind, symbol, tf)
    start = _last_ts(path) or int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    step = ex.parse_timeframe(tf) * 1000
    now = ex.milliseconds()
    rows = []

    def flush():
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[df["ts"] < pd.Timestamp(now - step, unit="ms", tz="UTC")]  # drop unfinished bar
        rows.clear()
        return _merge_write(path, df)

    while start < now - step:
        for attempt in range(10):
            try:
                batch = ex.fetch_ohlcv(symbol, tf, since=start, limit=limit)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError):  # timeouts, 429s, dropped responses: back off
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"{symbol} {tf}: fetch failed 10x at {start}")
        if not batch:
            # nothing before listing date / during delisting gaps; jump forward
            start += step * limit
            continue
        rows += batch
        nxt = batch[-1][0] + step
        if nxt <= start:
            break
        start = nxt
        if len(rows) >= 50_000:  # checkpoint so long runs survive crashes
            flush()
    if not rows:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return flush()


def funding(symbol: str, since: str = "2017-01-01", ex=None) -> pd.DataFrame:
    """Fetch perp funding history (OKX API only serves ~3 months back; vendor needed for more)."""
    ex = ex or exchange()
    path = _path("funding", symbol)
    start = _last_ts(path) or int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=start, limit=100)
        batch = [b for b in batch if b["timestamp"] >= start]
        if not batch:
            break
        rows += [(b["timestamp"], b["fundingRate"]) for b in batch]
        start = batch[-1]["timestamp"] + 1
        time.sleep(ex.rateLimit / 1000)
    if not rows:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "funding_rate"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return _merge_write(path, df)


def funding_archive(inst: str, since: str = "2018-01") -> pd.DataFrame:
    """Full funding history from OKX's monthly CSV archive (e.g. inst='BTC-USDT-SWAP').

    Writes the same data/funding/symbol=<BTC-USDT-USDT> Parquet as funding(), so the
    two merge. Months before listing / before the archive starts are skipped.
    """
    path = _path("funding", inst.replace("-SWAP", "/USDT:USDT").replace("-USDT/", "/"))
    frames = []
    for m in pd.period_range(since, pd.Timestamp.now(tz="UTC").strftime("%Y-%m"), freq="M"):
        url = (f"https://static.okx.com/cdn/okex/traderecords/swaprates/monthly/{m.strftime('%Y%m')}/"
               f"{inst}-fundingrates-{m}.zip")
        try:
            frames.append(pd.read_csv(url, usecols=["funding_rate", "funding_time"]))
        except Exception:  # 404 for months not archived
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).rename(columns={"funding_time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return _merge_write(path, df[["ts", "funding_rate"]])
