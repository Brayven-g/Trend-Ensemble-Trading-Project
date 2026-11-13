"""SQL access to local data via DuckDB.

Every sub-folder of data/ becomes a SQL view:
  - folders of Parquet files (hive partitioned, e.g. data/ohlcv/tf=1h/symbol=BTC-USDT/data.parquet)
    -> view named after the folder, partition keys (tf, symbol) become columns
  - CSV files anywhere under data/alt/ -> one view per file, named after the file stem

Usage in a notebook:
    from qr import q
    df = q("SELECT * FROM ohlcv WHERE symbol='BTC-USDT' AND tf='1d'")
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Final holdout: every SQL view ends before this date unless unlocked. No research or tuning looks at it.
# Unlock only to evaluate a frozen strategy once: set env QR_UNLOCK_HOLDOUT=1 before importing qr,
# or call qr.db.unlock_holdout().
HOLDOUT_START = "2025-10-01"
HOLDOUT_LOCKED = os.getenv("QR_UNLOCK_HOLDOUT") != "1"

_con: duckdb.DuckDBPyConnection | None = None


def connect(refresh: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a shared in-memory DuckDB connection with views over data/."""
    global _con
    if _con is not None and not refresh:
        return _con
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    if DATA_DIR.exists():
        for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
            if any(folder.rglob("*.parquet")):
                glob = (folder / "**" / "*.parquet").as_posix()
                src = f"read_parquet('{glob}', hive_partitioning=true, union_by_name=true)"
                cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
                cut = f" WHERE ts < TIMESTAMPTZ '{HOLDOUT_START} 00:00:00+00'" if HOLDOUT_LOCKED and "ts" in cols else ""
                con.execute(f"CREATE OR REPLACE VIEW {folder.name} AS SELECT * FROM {src}{cut}")
        alt = DATA_DIR / "alt"
        if alt.exists():
            for csv in alt.rglob("*.csv"):
                con.execute(
                    f"CREATE OR REPLACE VIEW {csv.stem} AS "
                    f"SELECT * FROM read_csv_auto('{csv.as_posix()}')"
                )
    _con = con
    return con


def unlock_holdout() -> None:
    """Open the final holdout for a one-time evaluation of a frozen strategy (rebuilds the views)."""
    global HOLDOUT_LOCKED
    HOLDOUT_LOCKED = False
    connect(refresh=True)


def q(sql: str, index: str | None = "ts") -> pd.DataFrame:
    """Run SQL, return DataFrame. Sets `index` column as index if present (default 'ts')."""
    df = connect().execute(sql).df()
    if index and index in df.columns:
        df = df.set_index(index).sort_index()
    return df


def tables() -> pd.DataFrame:
    """List available views and their columns."""
    return connect().execute(
        "SELECT table_name, string_agg(column_name, ', ') AS columns "
        "FROM information_schema.columns GROUP BY table_name ORDER BY table_name"
    ).df()


def ohlcv(coin: str, tf: str = "1h", perp: bool = False, source: str = "okx") -> pd.DataFrame:
    """OHLCV for one coin, e.g. ohlcv('BTC', '1h') -> OKX BTC-USDT spot; perp=True -> BTC-USDT-USDT swap.

    source='coinbase' -> Coinbase X-USD spot (longer history, 2015+; robustness checks only).
    """
    if source == "coinbase":
        view, sym = "ohlcv_coinbase", f"{coin}-USD"
    else:
        view, sym = "ohlcv", f"{coin}-USDT-USDT" if perp else f"{coin}-USDT"
    return q(f"SELECT ts, open, high, low, close, volume FROM {view} WHERE symbol = '{sym}' AND tf = '{tf}'")
