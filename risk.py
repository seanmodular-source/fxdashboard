"""
Risk engine. ONE methodology: historical simulation, 1-day horizon, 95%, last 500 completed trading days.

Why: no bell-curve assumption (FX has fat tails and jumps, historical days keep them at their real size);
500 days (~2 years) covers more than one calm year; at 95% there are 25 tail days, enough for a stable VaR and
ES (99% would rest on 5 days); ES says how bad the bad days are; the same tail days give an exact risk split.

    X[s,i] = sum_c E[i,c] * r[s,c]     what position i would make / lose if past day s happened again today
                                       (E = today's USD exposure of the position's currency legs,
                                        r = that day's % move of each currency vs USD)
    P[s]   = sum_i X[s,i]              the whole book's P&L on replayed day s
    VaR    = -percentile(P, 5%)                     loss exceeded on 1 day in 20 (numpy linear interpolation)
    ES     = -mean(P over the k worst days)         k = ceil(5% * n) -> 25 of 500: average loss on the bad days
    contribution_i = -mean(X[.,i] over the same k days)   -> sum_i contribution_i = ES exactly
    % of risk_i    = contribution_i / ES
    marginal_i     = contribution_i / |size_i| * $1m      ES change per $1m more of position i (tail days fixed)
    component VaR_i = -mean(X[.,i] over the 11 days ranked around the VaR day, 20th-30th worst of 500),
                     scaled so that sum_i component VaR_i = VaR exactly. VaR is set by one or two days, so their
                     split alone is noise; the band of nearby days gives a stable split (kernel / neighbourhood
                     estimate of the Euler allocation).
    marginal VaR_i = component VaR_i / |size_i| * $1m     VaR change per $1m more of position i (small changes)
    position VaR_i = -percentile(X[.,i], 5%)              position on its own
    diversification = sum_i position VaR_i - VaR
    backtest: for each of the last 250 days, VaR from the 500 days before it vs that day's P
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

CONF = 0.95          # VaR / ES confidence
LOOKBACK = 500       # completed trading days replayed
BACKTEST_DAYS = 250  # days tested in the backtest
VAR_BAND = 5         # component VaR averages the days ranked within 5 of the VaR day (11 days: 20th-30th worst)


@dataclass
class RiskReport:
    conf: float
    lookback: int
    n_obs: int
    scenario_pnl: pd.Series          # P: book P&L per replayed day (lookback window)
    position_pnl: pd.DataFrame       # X: per-position P&L per replayed day
    var_hs: float                    # historical VaR
    es: float                        # expected shortfall (average of the k worst days)
    tail_days: pd.DatetimeIndex      # the k worst days of P (used for ES and the contributions)
    position_var: pd.Series          # stand-alone historical VaR per position
    component: pd.Series             # contribution to ES per position (sums to ES)
    marginal_per_mm: pd.Series       # ES change per +$1m of the same position
    pct_risk: pd.Series              # contribution / ES
    var_band_days: pd.DatetimeIndex  # the days ranked around the VaR day (used for component VaR)
    component_var: pd.Series         # component VaR per position (sums to VaR)
    marginal_var_per_mm: pd.Series   # VaR change per +$1m of the same position
    diversification: float
    backtest: pd.DataFrame | None    # date, pnl, var, breach


def hs_var(pnl: pd.Series | np.ndarray, conf: float = CONF) -> float:
    arr = np.asarray(pnl, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(-np.percentile(arr, (1 - conf) * 100))


def tail_count(n: int, conf: float = CONF) -> int:
    """Number of worst days averaged by ES: ceil((1 - conf) * n), at least 1 (25 of 500 at 95%)."""
    return max(1, math.ceil(round((1 - conf) * n, 9)))


def expected_shortfall(pnl: pd.Series | np.ndarray, conf: float = CONF) -> float:
    """Average loss over the k worst days."""
    arr = np.sort(np.asarray(pnl, dtype=float))
    if arr.size == 0:
        return 0.0
    return float(-arr[:tail_count(arr.size, conf)].mean())


def scenario_pnl_matrix(exposure: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """X = returns @ exposure.T  (dates x positions)."""
    ccys = list(exposure.columns)
    r = returns[ccys]
    return pd.DataFrame(r.values @ exposure[ccys].values.T, index=r.index, columns=exposure.index)


def backtest_var(full_pnl: pd.Series, conf: float, lookback: int, days: int = BACKTEST_DAYS) -> pd.DataFrame | None:
    """Rolling out-of-sample check: VaR from the prior ``lookback`` days vs the next day's scenario P&L."""
    n = len(full_pnl)
    days = min(days, n - lookback)
    if days < 20:
        return None
    vals = full_pnl.values
    rows = []
    for i in range(n - days, n):
        var_i = hs_var(vals[i - lookback:i], conf)
        rows.append((full_pnl.index[i], vals[i], var_i, vals[i] < -var_i))
    return pd.DataFrame(rows, columns=["date", "pnl", "var", "breach"]).set_index("date")


def compute_risk(exposure: pd.DataFrame, returns: pd.DataFrame, size_usd: pd.Series,
                 conf: float = CONF, lookback: int = LOOKBACK) -> RiskReport | None:
    """Full risk report for the current positions. Returns None when there is no FX exposure."""
    if exposure.empty or exposure.abs().values.sum() == 0:
        return None
    x_full = scenario_pnl_matrix(exposure, returns)
    p_full = x_full.sum(axis=1)
    x = x_full.iloc[-lookback:]
    p = p_full.iloc[-lookback:]
    if len(p) < lookback:
        log.warning("Only %d return days available for a %d-day lookback", len(p), lookback)

    var_hs = hs_var(p, conf)
    k = tail_count(len(p), conf)
    tail = p.nsmallest(k).index
    es = float(-p.loc[tail].mean())
    component = -x.loc[tail].mean()                      # each position's average P&L on the book's worst days
    position_var = x.apply(lambda col: hs_var(col, conf))
    size = size_usd.reindex(component.index).abs()
    marginal = component / size.replace(0, np.nan) * 1e6
    pct = component / es if es else component * 0

    order = np.argsort(p.values, kind="stable")                 # replayed days, worst first
    mid = int(math.floor((1 - conf) * (len(p) - 1)))           # rank of the VaR day (24 = 25th worst of 500)
    band = p.index[order[max(0, mid - VAR_BAND): mid + VAR_BAND + 1]]   # the 11 days ranked 20th-30th worst
    raw = -x.loc[band].mean()
    component_var = raw * (var_hs / raw.sum()) if abs(raw.sum()) > 1e-9 else raw * 0
    marginal_var = component_var / size.replace(0, np.nan) * 1e6

    gap = abs(component.sum() - es)
    if gap > max(1.0, 1e-6 * abs(es)):
        log.warning("ES contributions %.2f do not add up to ES %.2f", component.sum(), es)

    report = RiskReport(
        conf=conf, lookback=lookback, n_obs=len(p), scenario_pnl=p, position_pnl=x, var_hs=var_hs, es=es,
        tail_days=pd.DatetimeIndex(tail), position_var=position_var, component=component,
        marginal_per_mm=marginal, pct_risk=pct, var_band_days=pd.DatetimeIndex(band), component_var=component_var,
        marginal_var_per_mm=marginal_var, diversification=float(position_var.sum() - var_hs),
        backtest=backtest_var(p_full, conf, lookback),
    )
    log.debug("Risk: VaR=%.0f ES=%.0f (%d tail days of %d)", var_hs, es, k, len(p))
    return report
