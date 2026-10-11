"""Hand-checked P&L cases from the blueprint (Section 9 test checklist)."""
import pandas as pd
import pytest

import pnl
import positions

OLD = "2026-01-05"
PREV = pd.Timestamp("2026-10-08")


def test_eurusd_long_hand_check(mk, q):
    t = mk([("EURUSD", "BUY", 1_000_000, 1.10, OLD)])
    assert pnl.mtm_usd(t, q(EUR=1.12)).iloc[0] == pytest.approx(20_000)


def test_usdjpy_short_hand_check(mk, q):
    t = mk([("USDJPY", "SELL", 3_000_000, 152.40, OLD)])
    # P&L in JPY = -3m x (150 - 152.40) = +7.2m JPY -> / 150 = +$48,000
    assert pnl.mtm_usd(t, q(JPY=1 / 150)).iloc[0] == pytest.approx(48_000)


def test_cross_converts_via_quote_ccy(mk, q):
    t = mk([("EURGBP", "BUY", 1_000_000, 0.85, OLD)])
    # EURGBP 0.86 with GBPUSD 1.30 -> Q_EUR = 0.86 * 1.30; P&L = 1m x 0.01 GBP x 1.30
    assert pnl.mtm_usd(t, q(EUR=0.86 * 1.30, GBP=1.30)).iloc[0] == pytest.approx(13_000)


def test_leg_mtm_equals_pair_formula(mk, q):
    t = mk([("EURUSD", "BUY", 5e6, 1.1697, OLD), ("USDJPY", "SELL", 3e6, 152.4, OLD),
            ("EURGBP", "SELL", 1.5e6, 0.8542, OLD), ("AUDNZD", "BUY", 2e6, 1.10, OLD)])
    row = q(EUR=1.1206, JPY=1 / 158.25, GBP=1.3233, AUD=0.6988, NZD=0.5612)
    pd.testing.assert_series_equal(pnl.mtm_usd(t, row), pnl.mtm_usd_pair_formula(t, row), check_names=False)


def test_offsetting_trade_locks_in_pnl(mk, q):
    t = mk([("EURUSD", "BUY", 1e6, 1.10, OLD), ("EURUSD", "SELL", 1e6, 1.12, OLD)])
    for spot in (1.05, 1.12, 1.20):
        assert pnl.mtm_usd(t, q(EUR=spot)).sum() == pytest.approx(20_000)
    exp = positions.exposure_by_currency(t, q(EUR=1.15))
    assert exp.get("EUR", 0.0) == pytest.approx(0.0)


def test_same_day_trade_daily_equals_inception(mk, q):
    t = mk([("NZDUSD", "BUY", 3e6, 0.5600, "2026-10-09")])
    out = pnl.trade_pnl(t, q(NZD=0.5612), q(NZD=0.5590), PREV)
    assert out["daily_pnl"].iloc[0] == pytest.approx(out["inception_pnl"].iloc[0])


def test_daily_pnl_uses_previous_close_rates(mk, q):
    t = mk([("USDJPY", "BUY", 1e6, 150.0, OLD)])
    now, prev = q(JPY=1 / 152.0), q(JPY=1 / 151.0)
    out = pnl.trade_pnl(t, now, prev, PREV)
    expected = pnl.mtm_usd(t, now).iloc[0] - pnl.mtm_usd(t, prev).iloc[0]
    assert out["daily_pnl"].iloc[0] == pytest.approx(expected)


def test_short_gains_when_price_falls(mk, q):
    t = mk([("GBPUSD", "SELL", 2e6, 1.34, OLD)])
    assert pnl.mtm_usd(t, q(GBP=1.30)).iloc[0] > 0
    assert pnl.mtm_usd(t, q(GBP=1.38)).iloc[0] < 0


def test_break_even_entry_of_net_position(mk, q):
    t = mk([("EURUSD", "BUY", 5e6, 1.16973, OLD), ("EURUSD", "SELL", 2e6, 1.14479, OLD)])
    pos = positions.positions_table(t, q(EUR=1.12))
    expected = (5e6 * 1.16973 - 2e6 * 1.14479) / 3e6
    assert pos.loc["EURUSD", "avg_entry"] == pytest.approx(expected)
    # P&L of the net position at the break-even price is zero
    assert pnl.mtm_usd(t, q(EUR=expected)).sum() == pytest.approx(0.0, abs=1e-6)


def test_exposure_legs(mk, q):
    t = mk([("EURGBP", "BUY", 1e6, 0.85, OLD)])
    exp = positions.exposure_by_currency(t, q(EUR=1.10, GBP=1.30))
    assert exp["EUR"] == pytest.approx(1e6 * 1.10)
    assert exp["GBP"] == pytest.approx(-1e6 * 0.85 * 1.30)
