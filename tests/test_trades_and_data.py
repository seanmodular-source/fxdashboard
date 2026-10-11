import logging

import numpy as np
import pandas as pd
import pytest

import logger as L
import market_data as M
import trades as T


# ----------------------------------------------------------------------------- trades
def test_seed_loads_cleanly():
    df = T.load_seed()
    assert len(df) == 9 and (df["status"] == "LIVE").all()


def test_normalize_rejects_bad_rows():
    raw = pd.DataFrame({"trade_date": ["2026-01-05", "not a date"], "pair": ["EURUSD", "EURXXX"],
                        "side": ["BUY", "HOLD"], "notional": [1e6, -5], "price": [1.1, 0]})
    _, errors = T.normalize(raw)
    joined = " ".join(errors)
    for word in ("EURXXX", "side", "notional", "price", "trade_date"):
        assert word in joined


def test_add_cancel_and_ids():
    df = T.load_seed()
    trade = T.make_trade(df, "USDJPY", "SELL", 1e6, 150.0, "2026-10-09")
    assert trade["trade_id"] == "T010"
    df2 = T.add_trade(df, trade)
    assert len(df2) == 10
    df3 = T.cancel_trade(df2, "T010")
    assert df3.loc[df3["trade_id"] == "T010", "status"].iloc[0] == "CANCELLED"


def test_ticket_validation():
    today = pd.Timestamp("2026-10-09")
    errors, _ = T.validate_ticket("EURUSD", "BUY", 1e6, 1.12, "2026-12-01", 1.12, today, True)
    assert any("future" in e for e in errors)
    errors, warnings = T.validate_ticket("EURUSD", "BUY", 1e6, 1.30, today, 1.12, today, True)
    assert not errors and warnings  # fat-finger warning, not a block
    errors, _ = T.validate_ticket("EURUSD", "BUY", 0, 1.12, today, 1.12, today, True)
    assert errors


def test_csv_round_trip():
    df = T.load_seed()
    back, errors = T.read_upload(T.to_csv_bytes(df))
    assert not errors
    pd.testing.assert_frame_equal(back, df)


# ----------------------------------------------------------------------------- market data
def test_restamp_drops_weekends_and_live_bar():
    idx = pd.to_datetime(["2026-10-02", "2026-10-03", "2026-10-05", "2026-10-06", "2026-10-07"])  # Fri Sat Mon Tue Wed
    close = pd.DataFrame({"EURUSD=X": [1.0, 9.9, 1.1, 1.2, 1.3]}, index=idx)
    out = M.restamp_daily(close, drop_last=True)
    # Saturday dropped, Wednesday (live) dropped, Mon -> Fri, Tue -> Mon
    assert list(out.index) == list(pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"]))
    assert list(out["EURUSD=X"]) == [1.0, 1.1, 1.2]


def test_hourly_close_assigns_sunday_to_monday():
    idx = pd.to_datetime(["2026-10-09 20:00", "2026-10-11 22:00", "2026-10-12 05:00"])  # Fri, Sun, Mon
    h = pd.DataFrame({"X": [1.0, 2.0, 3.0]}, index=idx)
    d = M.hourly_to_daily_close(h)
    assert list(d.index) == list(pd.to_datetime(["2026-10-09", "2026-10-12"]))
    assert list(d["X"]) == [1.0, 3.0]


def test_fallback_used_when_live_fails(monkeypatch):
    monkeypatch.delenv("FXDASH_OFFLINE", raising=False)

    def boom():
        raise M.DataError("simulated Yahoo outage")

    monkeypatch.setattr(M, "_load_live", boom)
    M.load_market_data.clear()
    md = M.load_market_data()
    assert md.source == "fallback" and not md.live_marks and len(md.q) > 500
    M.load_market_data.clear()


def test_market_data_basics():
    md = M.load_market_data(offline=True)
    assert md.now["USD"] == 1.0
    assert md.prev_date < md.last_date
    assert md.pair_price("EURUSD") == pytest.approx(md.now["EUR"])
    assert md.pair_price("USDJPY") == pytest.approx(1 / md.now["JPY"])


# ----------------------------------------------------------------------------- logger
def test_setup_logging_is_idempotent():
    root = L.setup_logging()
    n = len(root.handlers)
    L.setup_logging()
    L.setup_logging()
    assert len(logging.getLogger(L.ROOT_NAME).handlers) == n


def test_log_step_attaches_error_id_and_logs_traceback():
    @L.log_step
    def broken():
        raise ValueError("kaboom")

    with pytest.raises(ValueError) as info:
        broken()
    err_id = info.value.err_id
    assert len(err_id) == 6
    for h in logging.getLogger(L.ROOT_NAME).handlers:
        h.flush()
    found = L.find_in_log(err_id)
    assert found and any("kaboom" in line for line in found)
    assert any("Traceback" in line for line in found)
