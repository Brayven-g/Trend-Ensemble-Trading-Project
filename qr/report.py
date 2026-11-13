"""Notebook presentation helpers: coloured callouts and pass/fail tables.

    from qr.report import callout, verdict_table, color_p
    callout("amber", "Verdict: weak", "<b>L=28</b> passes on 1/5 coins ...")
    verdict_table([("HAC p_adj < 0.05 on >= 3/5 coins", "1/5", False), ...])
    color_p(df, ["hac_p_adj"])            # green < 0.05, amber < 0.10, red otherwise
"""
from __future__ import annotations

import pandas as pd
from IPython.display import HTML

COLORS = {"green": ("#1a7f37", "#e6f4ea"), "red": ("#cf222e", "#fdecea"), "amber": ("#9a6700", "#fff4d6"),
          "grey": ("#57606a", "#f2f3f5")}
CELL = {k: f"background-color: {v[1]}; color: {v[0]}" for k, v in COLORS.items()}


def callout(kind: str, title: str, body_html: str) -> HTML:
    """Box with a coloured left border: green = pass, red = fail, amber = borderline / weak, grey = info."""
    fg, bg = COLORS[kind]
    return HTML(f'<div style="border-left: 6px solid {fg}; background: {bg}; color: #1f2328; padding: 10px 14px; '
                f'margin: 6px 0; border-radius: 4px;"><div style="font-weight: 700; color: {fg}; margin-bottom: 4px;">'
                f'{title}</div>{body_html}</div>')


def verdict_table(rows: list[tuple]) -> "pd.io.formats.style.Styler":
    """rows = (criterion, value, status) with status True (pass) / False (fail) / None (n/a) / 'borderline'."""
    df = pd.DataFrame(rows, columns=["criterion", "value", "status"])
    label = {True: "pass", False: "fail", None: "n/a", "borderline": "borderline"}
    kind = {True: "green", False: "red", None: "grey", "borderline": "amber"}
    df["result"] = df["status"].map(label)
    colors = df["status"].map(kind)
    out = df[["criterion", "value", "result"]]
    return out.style.apply(lambda col: [CELL[c] for c in colors] if col.name == "result" else [""] * len(col), axis=0) \
        .hide(axis="index")


def color_p(df: pd.DataFrame, cols, alpha: float = 0.05, border: float = 0.10, fmt: dict | None = None):
    """Colour p-value columns: green < alpha, amber < border, red otherwise."""
    def one(v):
        if pd.isna(v):
            return CELL["grey"]
        return CELL["green"] if v < alpha else CELL["amber"] if v < border else CELL["red"]
    return df.style.map(one, subset=list(cols)).format(fmt or {}, na_rep="-")


def color_sign(styler_or_df, cols):
    """Colour effect columns: green > 0, red < 0."""
    s = styler_or_df if hasattr(styler_or_df, "map") and not isinstance(styler_or_df, pd.DataFrame) else styler_or_df.style
    return s.map(lambda v: "" if pd.isna(v) else CELL["green"] if v > 0 else CELL["red"] if v < 0 else "", subset=list(cols))
