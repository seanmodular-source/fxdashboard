"""
Pre-trade impact: the book summary before and after a proposed trade (engine.book_summary).
"""
from __future__ import annotations

import pandas as pd


def compare_books(before: dict, after: dict) -> pd.DataFrame:
    """Side-by-side of two book summaries (see engine.book_summary)."""
    df = pd.DataFrame({"Before": pd.Series(before), "After": pd.Series(after)})
    df["Change"] = df["After"] - df["Before"]
    return df
