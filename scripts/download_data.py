"""Download every dataset the notebooks use into data/ (Parquet, incremental).

    python scripts/download_data.py

- OKX spot daily candles for BTC, ETH, XRP, LTC, SOL  -> data/ohlcv/tf=1d/...
- OKX perpetual funding history for the same coins   -> data/funding/...
- Coinbase daily candles for BTC, ETH, LTC (2015+)   -> data/ohlcv_coinbase/tf=1d/...

Public market data only, no API keys needed. Re-running fetches only new bars.
"""
import sys
from pathlib import Path

import ccxt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qr import download  # noqa: E402

COINS = ["BTC", "ETH", "XRP", "LTC", "SOL"]
COINBASE_COINS = ["BTC", "ETH", "LTC"]

okx = download.exchange()
for c in COINS:
    df = download.ohlcv(f"{c}/USDT", "1d", "2017-01-01", okx)
    print(f"OKX {c}/USDT 1d: {len(df)} bars, {df.ts.min().date()} -> {df.ts.max().date()}")

for c in COINS:
    try:
        download.funding_archive(f"{c}-USDT-SWAP")      # long history from OKX's monthly archive
    except Exception as e:
        print(f"OKX {c} funding archive skipped: {e}")
    f = download.funding(f"{c}/USDT:USDT", "2017-01-01", okx)  # recent prints from the API
    print(f"OKX {c} funding: {len(f)} rows")

cb = ccxt.coinbaseexchange({"enableRateLimit": True})
for c in COINBASE_COINS:
    df = download.ohlcv(f"{c}/USD", "1d", "2015-01-01", cb, kind="ohlcv_coinbase", limit=300)
    print(f"Coinbase {c}/USD 1d: {len(df)} bars, {df.ts.min().date()} -> {df.ts.max().date()}")
