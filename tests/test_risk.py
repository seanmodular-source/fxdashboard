import numpy as np
import pandas as pd
import pytest

import risk


@pytest.fixture
def returns():
    rng = np.random.default_rng(7)
    n = 600
    common = rng.normal(0, 0.005, n)
    data = {
        "EUR": common + rng.normal(0, 0.002, n),
        "GBP": 0.8 * common + rng.normal(0, 0.003, n),
        "JPY": -0.3 * common + rng.normal(0, 0.006, n),
    }
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n))


@pytest.fixture
def exposure():
    # rows = positions, cols = currencies (USD exposure); GBPUSD short partly hedges EURUSD long
    return pd.DataFrame({"EUR": [3e6, 0, 0], "GBP": [0, -2e6, 0], "JPY": [0, 0, -3e6]},
                        index=["EURUSD", "GBPUSD", "USDJPY"])


def test_hs_var_and_es_definitions():
    pnl = np.arange(-100, 100, dtype=float)  # 200 equally likely outcomes
    assert risk.hs_var(pnl, 0.95) == pytest.approx(-np.percentile(pnl, 5))
    assert risk.tail_count(500) == 25 and risk.tail_count(200) == 10
    assert risk.expected_shortfall(pnl, 0.95) == pytest.approx(-np.mean(np.arange(-100, -90)))   # 10 worst
    assert risk.expected_shortfall(pnl, 0.95) >= risk.hs_var(pnl, 0.95)


def test_contributions_add_up_to_es(exposure, returns):
    r = risk.compute_risk(exposure, returns, pd.Series([3e6, -2e6, 3e6], index=exposure.index), 0.95, 500)
    assert r.component.sum() == pytest.approx(r.es, rel=1e-9)
    assert r.pct_risk.sum() == pytest.approx(1.0)
    worst = r.scenario_pnl.nsmallest(25).index
    assert r.component["EURUSD"] == pytest.approx(-r.position_pnl.loc[worst, "EURUSD"].mean())


def test_hedge_has_negative_contribution(exposure, returns):
    r = risk.compute_risk(exposure, returns, pd.Series([3e6, -2e6, 3e6], index=exposure.index), 0.95, 500)
    assert r.component["GBPUSD"] < 0  # short GBP against long EUR (positively correlated)


def test_marginal_is_contribution_per_dollar_and_matches_a_small_increase(exposure, returns):
    size = pd.Series([3e6, -2e6, 3e6], index=exposure.index)
    r = risk.compute_risk(exposure, returns, size, 0.95, 500)
    assert r.marginal_per_mm["EURUSD"] == pytest.approx(r.component["EURUSD"] / 3e6 * 1e6)
    bumped = exposure.copy()
    bumped.loc["EURUSD"] *= 1 + 1e-4                                   # $300 more EURUSD
    r2 = risk.compute_risk(bumped, returns, size, 0.95, 500)
    assert (r2.es - r.es) / 300 * 1e6 == pytest.approx(r.marginal_per_mm["EURUSD"], rel=1e-6)


def test_component_var_adds_up_to_var_and_matches_a_hand_calculation(exposure, returns):
    size = pd.Series([3e6, -2e6, 3e6], index=exposure.index)
    r = risk.compute_risk(exposure, returns, size, 0.95, 500)
    assert r.component_var.sum() == pytest.approx(r.var_hs, rel=1e-9)
    p, x = r.scenario_pnl, r.position_pnl
    band = p.sort_values(kind="stable").index[19:30]                   # 20th-30th worst of 500
    assert set(band) == set(r.var_band_days)
    raw = -x.loc[band].mean()
    pd.testing.assert_series_equal(r.component_var, raw * r.var_hs / raw.sum())
    assert r.marginal_var_per_mm["EURUSD"] == pytest.approx(r.component_var["EURUSD"] / 3e6 * 1e6)


def test_backtest_shape(exposure, returns):
    r = risk.compute_risk(exposure, returns, pd.Series([1.0] * 3, index=exposure.index), 0.95, 300)
    assert r.backtest is not None and len(r.backtest) == 250
    assert set(r.backtest.columns) == {"pnl", "var", "breach"}


def test_no_exposure_returns_none(returns):
    assert risk.compute_risk(pd.DataFrame(), returns, pd.Series(dtype=float)) is None


def test_risk_uses_completed_days_only():
    """On a weekday the latest row is today's unfinished day: VaR and the backtest must leave it out."""
    import market_data as M
    import trades as T
    import engine as E
    md = M.load_market_data(offline=True)
    running = M.MarketData(q=md.q, source="live", live_marks=True, as_of=md.last_date + pd.Timedelta(hours=9),
                           fetched_at=md.last_date + pd.Timedelta(hours=10), status=md.status)
    assert running.last_complete_date == md.prev_date
    closed = M.MarketData(q=md.q, source="live", live_marks=True, as_of=md.last_date + pd.Timedelta(hours=21),
                          fetched_at=md.last_date + pd.Timedelta(days=2), status=md.status)  # e.g. Sunday
    assert closed.last_complete_date == md.last_date
    book = E.build_book(T.load_seed(), running, pd.Series(dtype=float), E.Settings())
    assert book.risk.scenario_pnl.index[-1] == md.prev_date
    assert book.risk.backtest.index[-1] == md.prev_date
