import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FXDASH_OFFLINE", "1")  # unit tests never hit Yahoo (see test_network.py)
# the tests read and write their own book file, never your data/portfolio.csv
os.environ["FXDASH_BOOK_FILE"] = str(Path(tempfile.mkdtemp(prefix="fxdash_test_")) / "portfolio.csv")

import positions  # noqa: E402
import trades as T  # noqa: E402


def make_trades(rows):
    """rows: (pair, side, notional, price, trade_date) -> normalised live trades with base/quote/signed."""
    df = pd.DataFrame(rows, columns=["pair", "side", "notional", "price", "trade_date"])
    clean, errors = T.normalize(df)
    assert not errors, errors
    return positions.live_trades(clean)


def q_row(**kwargs):
    """Q row (USD per unit) e.g. q_row(EUR=1.12, JPY=1/150)."""
    return pd.Series(kwargs, dtype=float)


@pytest.fixture
def mk():
    return make_trades


@pytest.fixture
def q():
    return q_row
