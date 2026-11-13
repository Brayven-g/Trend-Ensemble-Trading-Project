"""Research bookkeeping.

Every notebook's first code cell defines a META dict, e.g.
    META = {"id": "S0004", "title": "...", "status": "viable", "hypotheses": ["hypotheses/H0002_x.ipynb"]}
save_oos() stores a strategy's final net OOS daily returns in results/ so other notebooks can load them.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from qr.db import ROOT

RESULTS_DIR = ROOT / "results"


def read_meta(nb_path: Path) -> dict:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "META":
                return ast.literal_eval(node.value)
        break
    return {}


def save_oos(strategy_id: str, returns: pd.Series) -> Path:
    """Save net OOS returns, compounded to daily, to results/<id>_oos.parquet."""
    RESULTS_DIR.mkdir(exist_ok=True)
    daily = (1 + returns.fillna(0)).resample("D").prod() - 1
    path = RESULTS_DIR / f"{strategy_id}_oos.parquet"
    daily.rename("ret").to_frame().to_parquet(path)
    return path


def load_oos(strategy_id: str) -> pd.Series:
    """Saved OOS daily returns, clipped to before qr.db.HOLDOUT_START while the holdout is locked.
    Prefer this to pd.read_parquet so a saved file never leaks holdout data into research."""
    from qr import db
    r = pd.read_parquet(RESULTS_DIR / f"{strategy_id}_oos.parquet")["ret"]
    if db.HOLDOUT_LOCKED:
        r = r[r.index < pd.Timestamp(db.HOLDOUT_START, tz=r.index.tz)]
    return r.rename(strategy_id)
