import numpy as np
import pandas as pd
import pytest

import universe as U
from analytics import compute_market_analytics
from analytics import beta as B
from analytics import richcheap as RC
from analytics import sessions as SS


def synthetic_q(n=600, seed=3):
    """Random-walk Q panel (USD per unit) for every universe currency."""
    rng = np.random.default_rng(seed)
    dollar = rng.normal(0, 0.004, n)
    cols = {}
    for i, c in enumerate(U.CURRENCIES):
        start = 1.0 / (1 + i)
        r = -(0.6 + 0.05 * (i % 8)) * dollar + rng.normal(0, 0.003, n)
        cols[c] = start * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=pd.bdate_range("2024-01-01", periods=n))


def test_synthetic_dxy_matches_official_formula():
    p = {"EURUSD": 1.12, "USDJPY": 158.25, "GBPUSD": 1.3233, "USDCAD": 1.4254, "USDSEK": 9.994, "USDCHF": 0.83}
    official = (50.14348112 * p["EURUSD"] ** -0.576 * p["USDJPY"] ** 0.136 * p["GBPUSD"] ** -0.119
                * p["USDCAD"] ** 0.091 * p["USDSEK"] ** 0.042 * p["USDCHF"] ** 0.036)
    q = pd.DataFrame([{"EUR": p["EURUSD"], "JPY": 1 / p["USDJPY"], "GBP": p["GBPUSD"], "CAD": 1 / p["USDCAD"],
                       "SEK": 1 / p["USDSEK"], "CHF": 1 / p["USDCHF"]}])
    assert B.synthetic_dxy(q).iloc[0] == pytest.approx(official, rel=1e-12)


def test_level_z_is_symmetric_under_inversion():
    q = synthetic_q()[["JPY"]]
    z_ccy = RC.level_z(np.log(q), 252)
    z_usd = RC.level_z(np.log(1 / q), 252)
    assert z_ccy["JPY"] == pytest.approx(-z_usd["JPY"])


def test_rolling_beta_recovers_true_beta():
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0, 0.004, 400))
    u = pd.DataFrame({"X": 1.3 * d + rng.normal(0, 0.001, 400)})
    beta, r2 = B.rolling_beta(u, d, 252, 200)
    assert beta["X"].iloc[-1] == pytest.approx(1.3, abs=0.05)
    assert r2["X"].iloc[-1] > 0.9


def test_layered_breakdown_adds_up_exactly():
    ma = compute_market_analytics(synthetic_q())
    for h, d in ma.decomposition.items():
        np.testing.assert_allclose(d["total"], d[["dollar", "bloc", "own"]].sum(axis=1), atol=1e-10)


def test_dxy_equivalent_sign():
    # long $1m EUR with USD-beta 1 behaves like short $1m DXY
    assert B.dxy_equivalent(pd.Series({"EUR": 1e6}), pd.Series({"EUR": 1.0})) == pytest.approx(-1e6)


def test_sessions_add_up_to_daily_log_return():
    idx = pd.date_range("2026-09-01 00:00", "2026-09-12 21:00", freq="h", tz=None)
    idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(5)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx)))), index=idx)
    zr, daily = SS.session_returns(s)
    np.testing.assert_allclose(zr.sum(axis=1).iloc[1:], daily.iloc[1:], atol=1e-10)  # log returns add up exactly
    summ = SS.session_summary(zr)
    total = np.exp(zr.sum().sum() / 100) - 1               # whole period's move, all sessions together
    assert np.prod(1 + summ["cumulative_%"] / 100) - 1 == pytest.approx(total, rel=1e-9)


def test_session_zscore_lookback_excludes_today():
    zr = pd.DataFrame({"Asia": np.r_[np.zeros(30), 5.0], "Europe": 0.0, "US": 0.0})
    zr.iloc[:30, 0] = np.random.default_rng(2).normal(0, 1, 30)
    z20 = SS.session_zscores(zr, lookback=20)
    prior = zr["Asia"].iloc[10:30]
    assert z20["Asia"].iloc[-1] == pytest.approx((5.0 - prior.mean()) / prior.std())


def test_session_hit_rate_ignores_missing():
    zr = pd.DataFrame({"Asia": [1.0, -1.0, np.nan], "Europe": [1.0, 1.0, 1.0], "US": [-1.0, -1.0, -1.0]})
    summ = SS.session_summary(zr)
    assert summ.loc["Asia", "hit_rate_%"] == pytest.approx(50.0)


def test_cross_prices_and_z_are_symmetric():
    q = synthetic_q()
    p = RC.pair_prices(q, ["AUDNZD", "NZDAUD", "EURUSD"])
    assert np.allclose(p["AUDNZD"] * p["NZDAUD"], 1.0)
    pd.testing.assert_series_equal(p["EURUSD"], q["EUR"], check_names=False)
    rc = RC.rich_cheap_table(p)
    assert rc.loc["AUDNZD", "level_z_252"] == pytest.approx(-rc.loc["NZDAUD", "level_z_252"])


def test_g10_crosses_follow_market_convention():
    crosses = U.g10_crosses()
    assert len(crosses) == len(set(crosses)) == 36
    assert {"EURGBP", "AUDNZD", "NOKSEK", "CADJPY"} <= set(crosses) and not any("USD" in c for c in crosses)
