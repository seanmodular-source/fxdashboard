"""Bad-print filter (market_data.clean_spikes) and the quoted-cross data check (crosscheck.py)."""
import numpy as np
import pandas as pd
import pytest

import crosscheck as XC
import market_data as M


def walk(n=400, seed=1, start=0.055, vol=0.006):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0, vol, n))), index=idx)


def eur_zar():
    return pd.DataFrame({"EUR": walk(seed=5, start=1.10, vol=0.004), "ZAR": walk(seed=6)})


def make_md(q: pd.DataFrame) -> M.MarketData:
    return M._finalise(q, "live", False, q.index[-1], [])


# ----------------------------------------------------------------------------- bad-print filter
def test_clean_spikes_removes_a_reversing_print():
    s = walk()
    bad_day = s.index[200]
    raw = s.copy()
    raw[bad_day] *= 1.22                            # +22% for one day, back the next (USDZAR Nov 2024 style)
    clean, corr = M.clean_spikes(raw.to_frame("ZAR"))
    assert list(corr["date"]) == [bad_day]
    assert clean.loc[bad_day, "ZAR"] == pytest.approx(np.sqrt(s.iloc[199] * s.iloc[201]))  # neighbours' midpoint
    assert corr.loc[0, "ticker"] == "USDZAR" and corr.loc[0, "bad_price"] == pytest.approx(1 / raw[bad_day])
    others = clean.index != bad_day
    pd.testing.assert_series_equal(clean.loc[others, "ZAR"], raw[others], check_names=False)


def test_clean_spikes_keeps_a_real_jump_that_holds():
    s = walk()
    s.iloc[200:] *= 1.20                            # SNB-style: jumps and stays there
    clean, corr = M.clean_spikes(s.to_frame("CHF"))
    assert corr.empty
    pd.testing.assert_frame_equal(clean, s.to_frame("CHF"))


def test_clean_spikes_handles_back_to_back_prints():
    s = walk(seed=2)
    raw = s.copy()
    raw.iloc[200] *= 1.15                           # ZAR Jan 2025 pattern: bad, good, bad
    raw.iloc[202] *= 1.21
    clean, corr = M.clean_spikes(raw.to_frame("ZAR"))
    assert set(corr["date"]) == {s.index[200], s.index[202]}
    assert (clean["ZAR"] / s - 1).abs().max() < 0.03


def test_market_data_records_corrections():
    md = M.load_market_data(offline=True)
    assert {"date", "ticker", "bad_price", "clean_price"} <= set(md.corrections.columns)
    assert "bad_prints" in md.status.columns


# ----------------------------------------------------------------------------- cross check
def test_cross_check_flags_a_wrong_price_today_and_in_history():
    q = eur_zar()
    quoted = (q["EUR"] / q["ZAR"]).to_frame("EURZAR")   # Yahoo's quoted cross = the true cross
    q_bad = q.copy()
    q_bad.iloc[-1, 1] *= 1.01                           # today's ZAR 1% off (the latest price is never cleaned)
    q_bad.iloc[-100:-98, 1] *= 1.03                     # a two-day error: not a one-day spike, so not cleaned
    xc = XC.compare(make_md(q_bad), quoted, ["ZAR"], lookback=250)
    row = xc.table.loc["ZAR"]
    assert row["status"].startswith("❌") and row["gap_now_bps"] == pytest.approx(100, abs=1)
    assert len(xc.flagged) >= 2 and (xc.flagged["gap_bps"].abs() > 200).sum() == 2
    levels = [lvl for lvl, _ in xc.alerts]
    assert levels.count("warning") == 2              # today's price + the VaR-window history
    assert xc.checks and not any(xc.checks.values())


def test_cross_check_confirms_a_removed_bad_print():
    q = eur_zar()
    quoted = (q["EUR"] / q["ZAR"]).to_frame("EURZAR")
    q_bad = q.copy()
    q_bad.iloc[-50, 1] *= 1.22
    md = make_md(q_bad)
    assert len(md.corrections) == 1
    xc = XC.compare(md, quoted, ["ZAR"], lookback=250)
    assert xc.corrections.loc[0, "cross_check"].startswith("confirmed")
    assert xc.table.loc["ZAR", "status"] == "✅ OK" and all(xc.checks.values())


def test_cross_check_skips_offline_but_still_reports_cleaning(monkeypatch):
    monkeypatch.setenv("FXDASH_OFFLINE", "1")
    md = M.load_market_data(offline=True)
    xc = XC.run(md, ["EUR", "ZAR"], 250)
    assert xc.status == "skipped"
    assert any("skipped" in m for _, m in xc.alerts)
    if len(md.corrections):
        assert any("Data cleaning removed" in m for _, m in xc.alerts)


def test_cross_check_download_failure_is_a_warning(monkeypatch):
    monkeypatch.delenv("FXDASH_OFFLINE", raising=False)

    def boom(pairs, md):
        raise M.DataError("simulated Yahoo outage")

    monkeypatch.setattr(XC, "load_quoted", boom)
    xc = XC.run(make_md(eur_zar()), ["ZAR"], 250)
    assert xc.status == "unavailable"
    assert any(lvl == "warning" and "unavailable" in m for lvl, m in xc.alerts)


def test_cross_check_never_raises(monkeypatch):
    monkeypatch.delenv("FXDASH_OFFLINE", raising=False)
    q = eur_zar()
    monkeypatch.setattr(XC, "load_quoted", lambda pairs, md: ((q["EUR"] / q["ZAR"]).to_frame("EURZAR"), []))

    def broken(*args, **kwargs):
        raise ValueError("kaboom")

    monkeypatch.setattr(XC, "compare", broken)
    xc = XC.run(make_md(q), ["ZAR"], 250)
    assert xc.status == "error" and any("error ID" in m for _, m in xc.alerts)


def test_cross_check_ignores_an_unreliable_reference():
    q = eur_zar()
    rng = np.random.default_rng(9)
    noisy = (q["EUR"] / q["ZAR"]) * np.exp(rng.normal(0, 0.01, len(q)))   # a quoted cross that is ~100 bps off daily
    xc = XC.compare(make_md(q), noisy.to_frame("EURZAR"), ["ZAR"], lookback=250)
    assert xc.table.loc["ZAR", "status"].startswith("⚪")
    assert not any(lvl == "warning" for lvl, _ in xc.alerts)
    assert any("too noisy" in m for _, m in xc.alerts)
    assert xc.flagged.empty and all(xc.checks.values())


def test_check_cross_overrides():
    assert XC.check_pair("ZAR") == "EURZAR" and XC.check_pair("EUR") == "EURGBP"
    assert XC.check_pair("IDR") == "SGDIDR"
