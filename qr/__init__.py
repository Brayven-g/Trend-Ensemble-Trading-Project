"""Shared research library.

Notebooks hold idea-specific code (event definitions, signal functions, conclusions).
Everything generic (data access, backtest engine, validation stages) lives here so it is
written and tested once and results stay comparable across strategies.
"""
from qr.db import q, connect, tables, ohlcv
from qr.backtest import run, metrics, Result
from qr.costs import CostModel, leverage_sweep, okx_funding

__all__ = ["q", "connect", "tables", "ohlcv", "okx_funding", "run", "metrics", "Result", "CostModel", "leverage_sweep"]
