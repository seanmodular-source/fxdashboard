"""SGT clock, calendar events, seasonality, P&L attribution and the extended rich/cheap windows."""
import numpy as np
import pandas as pd
import pytest

import attribution as A
import clock
import engine as E
import market_data as M
import trades as T
from analytics import calendar_events as CE
from analytics import richcheap as RC
from analytics import seasonality as SE
from conftest import make_trades


# ----------------------------------------------------------------------------- clock
def test_sgt_is_utc_plus_8_and_daily_close_is_0800_sgt_next_day():
    assert clock.to_sgt(pd.Timestamp("2026-10-09 22:00")) == pd.Timestamp("2026-10-10 06:00")
    assert clock.to_sgt(pd.Timestamp("2026-10-09 22:00", tz="UTC")) == pd.Timestamp("2026-10-10 06:00")
    assert clock.close_time("2026-10-08") == "09 Oct 08:00 SGT"
    hours = [h for z in ("Asia", "Europe", "US") for h in range(*clock.SESSION_HOURS_UTC[z])]
    assert hours == list(range(24))                      # the three sessions cover the day exactly once


# ----------------------------------------------------------------------------- calendar
def test_calendar_codes_and_weekend_events_move_to_next_trading_day():
    ev = CE.events_between("USDJPY", "2026-01-01", "2026-02-28")
    codes = set(ev["event_code"])
    assert {"ME", "MM", "HD"} <= codes
    assert ((ev["event_code"] == "HD") & ev["event_name"].str.startswith("JP")).any()
    assert ((ev["event_code"] == "HD") & ev["event_name"].str.startswith("US")).any()
    days = pd.bdate_range("2026-01-01", "2026-03-06")
    mapped = CE.map_to_trading_days(ev, days)
    jan31 = mapped[(mapped["event_code"] == "ME") & (mapped["event_date"] == "2026-01-31")]  # a Saturday
    assert list(jan31["plotted_date"]) == [pd.Timestamp("2026-02-02")]                    # -> Monday
    jp_fy = CE.events_between("USDJPY", "2026-03-30", "2026-04-02")
    assert {"FE", "FS"} <= set(jp_fy["event_code"])                                     # JP fiscal year Apr-Mar


# ----------------------------------------------------------------------------- seasonality
def _monthly_price():
    idx = pd.bdate_range("2015-01-01", "2026-10-09")
    rng = np.random.default_rng(11)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx)))), index=idx)


def test_monthly_returns_run_month_end_to_month_end_and_skip_the_open_month():
    p = _monthly_price()
    r, mtd = SE.monthly_returns(p)
    assert r.index[-1] == pd.Timestamp("2026-09-30")    # October 2026 is not finished
    sep_end, aug_end = p.loc[:"2026-09-30"].iloc[-1], p.loc[:"2026-08-31"].iloc[-1]
    assert r.iloc[-1] == pytest.approx((sep_end / aug_end - 1) * 100)
    assert mtd == pytest.approx((p.iloc[-1] / sep_end - 1) * 100)


def test_demeaning_uses_only_past_months():
    r, _ = SE.monthly_returns(_monthly_price())
    d1 = SE.demean_trailing(r)
    r2 = r.copy()
    r2.iloc[-1] += 50.0                                  # change the latest month only
    d2 = SE.demean_trailing(r2)
    pd.testing.assert_series_equal(d1.iloc[:-1], d2.iloc[:-1])
    stats = SE.month_stats(r)
    assert list(stats.index) == SE.MONTHS and stats["years"].between(10, 12).all()
    jan = r[r.index.month == 1]
    assert stats.loc["Jan", "hit_rate_%"] == pytest.approx((jan > 0).mean() * 100)


# ----------------------------------------------------------------------------- attribution
def test_pnl_by_currency_adds_up_to_the_headline_exactly():
    md = M.load_market_data(offline=True)
    book = E.build_book(T.load_seed(), md, pd.Series(dtype=float), E.Settings())
    by = A.pnl_by_currency(book.live, md)
    assert by["daily_pnl"].sum() == pytest.approx(book.daily_pnl, abs=0.01)
    assert by["inception_pnl"].sum() == pytest.approx(book.inception_pnl, abs=0.01)
    assert "USD" not in by.index
    assert not any("xecution" in str(c) for c in by.index)          # no left-over line


def test_currency_pnl_matches_the_positions_table():
    """A currency traded only against USD carries exactly that pair's P&L (Overview = Positions)."""
    md = M.load_market_data(offline=True)
    book = E.build_book(T.load_seed(), md, pd.Series(dtype=float), E.Settings())
    by = A.pnl_by_currency(book.live, md)
    pos = book.positions
    pos = pos.set_index("pair") if "pair" in pos else pos
    for ccy, pair in [("NZD", "NZDUSD"), ("AUD", "AUDUSD"), ("JPY", "USDJPY"), ("CAD", "USDCAD"), ("CHF", "USDCHF")]:
        for col in ("daily_pnl", "inception_pnl"):
            assert by.loc[ccy, col] == pytest.approx(pos.loc[pair, col], abs=0.01), (ccy, col)
    # the cross is split between its two currencies
    for col in ("daily_pnl", "inception_pnl"):
        assert by.loc[["EUR", "GBP"], col].sum() == pytest.approx(
            pos.loc[["EURUSD", "GBPUSD", "EURGBP"], col].sum(), abs=0.01)


def test_entry_prices_are_worth_zero_at_entry():
    live = make_trades([("EURGBP", "SELL", 1_500_000, 0.85, "2026-08-12"), ("USDJPY", "BUY", 3_000_000, 150.0,
                                                                            "2026-08-12")])
    q0 = pd.Series({"EUR": 1.15, "GBP": 1.35, "JPY": 1 / 149, "USD": 1.0})
    for t in live.itertuples():
        legs = A.entry_legs(t, q0)
        assert sum(a * q for _, a, q in legs) == pytest.approx(0.0, abs=1e-6)
        (b, _, qb), (q, _, qq) = legs
        assert qb / qq == pytest.approx(t.price)


def _hourly_eur():
    idx = pd.date_range("2026-09-07 00:00", "2026-09-18 23:00", freq="h")
    idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(4)
    eur = pd.Series(1.10 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx)))), index=idx)
    hourly = eur.to_frame("EUR")
    daily = M.hourly_to_daily_close(hourly)
    md = M.MarketData(q=daily, source="live", live_marks=True, as_of=idx[-1], fetched_at=idx[-1],
                      status=pd.DataFrame())
    return eur, hourly, md


def test_session_and_weekday_pnl_add_up_to_inception():
    eur, hourly, md = _hourly_eur()
    live = make_trades([("EURUSD", "BUY", 1_000_000, 1.10, "2026-09-09")])      # no time: at the 9 Sep close
    sa = A.session_attribution(live, md, hourly)
    expected = 1_000_000 * (eur.iloc[-1] - 1.10)                                 # = inception P&L
    assert sa.daily.sum() == pytest.approx(expected, rel=1e-9)
    assert sa.by_session_ccy.values.sum() == pytest.approx(expected, rel=1e-9)
    assert sa.by_weekday["total"].sum() == pytest.approx(expected, rel=1e-9)
    assert sa.before_window == 0 and sa.mark_gap == pytest.approx(0.0, abs=1e-6)
    assert sa.start == pd.Timestamp("2026-09-09")
    # entered in the 9 Sep 23:00 UTC bar (US session): entry price -> that bar's price
    first = sa.daily.loc["2026-09-09"]
    assert first == pytest.approx(1_000_000 * (eur.loc["2026-09-09 23:00"] - 1.10), rel=1e-9)


def test_trade_enters_in_the_hour_it_was_booked():
    eur, hourly, md = _hourly_eur()
    live = make_trades([("EURUSD", "BUY", 1_000_000, 1.10, "2026-09-09")])
    live["timestamp"] = "2026-09-09 10:15:00 SGT"                                # 02:15 UTC: Asia session
    sa = A.session_attribution(live, md, hourly)
    entry = 1_000_000 * (eur.loc["2026-09-09 02:00"] - 1.10)
    later_asia = 1_000_000 * (eur.loc["2026-09-09 07:00"] - eur.loc["2026-09-09 02:00"])
    assert sa.daily.sum() == pytest.approx(1_000_000 * (eur.iloc[-1] - 1.10), rel=1e-9)
    day1 = sa.daily.loc["2026-09-09"]
    assert day1 == pytest.approx(1_000_000 * (eur.loc["2026-09-09 23:00"] - 1.10), rel=1e-9)
    assert entry + later_asia == pytest.approx(
        1_000_000 * (eur.loc["2026-09-09 07:00"] - 1.10), rel=1e-9)


def test_trade_time_uses_booked_time_only_on_the_trade_date():
    assert A.trade_time_utc("2026-10-11 21:35:32 SGT", "2026-10-11") == pd.Timestamp("2026-10-11 13:35:32")
    assert A.trade_time_utc("2026-03-02 12:00:00 UTC", "2026-03-02") == pd.Timestamp("2026-03-02 12:00:00")
    close = pd.Timestamp("2026-03-02 23:59:59")
    assert A.trade_time_utc("2026-10-11 21:35:32 SGT", "2026-03-02") == close      # back-dated booking
    assert A.trade_time_utc("", "2026-03-02") == close


# ----------------------------------------------------------------------------- rich / cheap windows
def test_rich_cheap_has_every_window():
    md = M.load_market_data(offline=True)
    rc = RC.rich_cheap_table(md.q[["EUR", "JPY"]])
    for n in RC.LEVEL_WINDOWS.values():
        assert f"level_z_{n}" in rc
    for h in RC.MOVE_HORIZONS.values():
        assert f"stretch_z_{h}" in rc and f"ret_{h}d" in rc
    assert np.isfinite(rc["stretch_z_252"]).all()
    assert rc.loc["EUR", "ret_21d"] == pytest.approx((md.q["EUR"].iloc[-1] / md.q["EUR"].iloc[-22] - 1) * 100)


def test_season_paths_start_from_the_previous_close_on_calendar_days():
    p = _monthly_price()
    years = list(range(2016, 2027))
    mp = SE.month_paths(p, 3, years)
    assert (mp.loc[0] == 0).all()                                          # day 0 = previous month-end close
    feb_end = p.loc[:"2020-02-29"].iloc[-1]
    assert mp.loc[31, 2020] == pytest.approx((p.loc[:"2020-03-31"].iloc[-1] / feb_end - 1) * 100)
    yp = SE.year_paths(p, years)
    assert yp[2026].dropna().index.max() == (pd.Timestamp("2026-10-09") - pd.Timestamp("2026-01-01")).days
    fp = SE.forward_paths(p, 8, 24, 60, years)
    assert fp.loc[0].abs().max() < 1e-12 and len(fp) == 61
    s = SE.summary(fp, years[:-1])
    assert s.loc[60, "mean"] == pytest.approx(fp[years[:-1]].loc[60].mean())
    t = SE.year_month_table(SE.monthly_returns(p)[0])
    assert list(t.columns) == SE.MONTHS and 2020 in t.index
