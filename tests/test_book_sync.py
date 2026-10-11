"""Two-way sync between the dashboard and the book file (data/portfolio.csv; a temp file in tests)."""
import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import trades as T
from conftest import ROOT


@pytest.fixture
def fresh_book():
    """Start every test from the sample book and give the test its file path."""
    T.save_book(T.load_seed())
    return T.book_path()


def _app():
    os.environ["FXDASH_OFFLINE"] = "1"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


# ----------------------------------------------------------------------------- file helpers
def test_save_and_read_round_trip(fresh_book):
    df, errors = T.read_book()
    assert not errors and len(df) == len(T.load_seed())
    sig = T.book_signature()
    T.save_book(T.cancel_trade(df, "T001"))
    assert T.book_signature() != sig
    again, _ = T.read_book()
    assert again.set_index("trade_id").loc["T001", "status"] == "CANCELLED"


def test_row_without_id_gets_next_id_and_others_keep_theirs(fresh_book):
    raw = pd.read_csv(fresh_book)
    raw.loc[len(raw)] = [None, "", "2026-10-09", "USDJPY", "SELL", 1_000_000, 158.0, "LIVE"]
    df, errors = T.normalize(raw)
    assert not errors
    assert list(df["trade_id"][:-1]) == list(T.load_seed()["trade_id"])   # existing ids unchanged
    assert df["trade_id"].iloc[-1] == "T010"


def test_backup_before_replacing_the_whole_book(fresh_book):
    T.save_book(T.load_seed().iloc[:3], backup=True)
    backup = fresh_book.with_name(fresh_book.stem + ".backup.csv")
    assert len(pd.read_csv(backup)) == len(T.load_seed()) and len(pd.read_csv(fresh_book)) == 3


# ----------------------------------------------------------------------------- the running app
def test_outside_edit_to_the_csv_appears_in_the_dashboard(fresh_book):
    at = _app()
    n = len(at.session_state["trades"])
    raw = pd.read_csv(fresh_book)
    raw.loc[len(raw)] = ["T010", "", "2026-10-09", "USDJPY", "SELL", 1_000_000, 158.0, "LIVE"]
    raw.to_csv(fresh_book, index=False)                       # e.g. edited and saved in Excel
    at.run()
    assert len(at.session_state["trades"]) == n + 1
    assert "T010" in set(at.session_state["trades"]["trade_id"])


def test_dashboard_change_is_written_to_the_csv(fresh_book):
    at = _app()
    at.selectbox(key="cancel_id").set_value("T002")
    next(b for b in at.button if b.label == "Cancel trade").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    on_disk = pd.read_csv(fresh_book).set_index("trade_id")
    assert on_disk.loc["T002", "status"] == "CANCELLED"
    at.run()                                                   # the app's own write must not trigger a reload
    assert at.session_state["trades"].set_index("trade_id").loc["T002", "status"] == "CANCELLED"


def test_broken_csv_is_not_loaded_and_is_flagged(fresh_book):
    at = _app()
    good = at.session_state["trades"].copy()
    raw = pd.read_csv(fresh_book)
    raw.loc[0, "side"] = "HOLD"                                # invalid side
    raw.to_csv(fresh_book, index=False)
    at.run()
    assert at.session_state["book_errors"]
    pd.testing.assert_frame_equal(at.session_state["trades"].reset_index(drop=True), good.reset_index(drop=True))
    assert any("was not loaded" in e.value for e in at.error)
