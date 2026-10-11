"""
Synthetic DXY, rolling beta to DXY, and the dollar / residual split of each currency's move.

Why a synthetic DXY: Yahoo's DX-Y.NYB is an ICE futures-hours series; mixing it with Yahoo FX
closes misaligns timestamps. Rebuilding DXY from the same FX closes guarantees identical timing.

Conventions (log returns, so signs flip exactly and multi-day moves add up):
    l_c   = ln Q_c,t - ln Q_c,t-1          currency vs USD   (+ = currency stronger)
    u_c   = -l_c                           USD vs currency   (+ = USD stronger)  <- notebook convention
    d     = ln DXY_t - ln DXY_t-1
    beta  = cov(u_c, d) / var(d)           ~ +1 for EUR; displayed as "USD beta"
    dollar part (currency terms) = -beta_{t-1} * d_t
    residual                     = l_c - dollar part
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import universe as U


def synthetic_dxy(q: pd.DataFrame) -> pd.Series:
    """ln DXY = ln K - sum_c w_c ln Q_c  (requires the six DXY currencies)."""
    w = pd.Series(U.DXY_WEIGHTS)
    missing = [c for c in w.index if c not in q.columns]
    if missing:
        raise ValueError(f"Cannot build synthetic DXY, missing {missing}")
    log_q = np.log(q[w.index].dropna(how="any"))
    return np.exp(np.log(U.DXY_CONSTANT) - log_q.mul(w, axis=1).sum(axis=1)).rename("DXY")


def log_returns(q: pd.DataFrame) -> pd.DataFrame:
    return np.log(q).diff()


def rolling_beta(usd_ret: pd.DataFrame, dxy_ret: pd.Series, window: int, min_periods: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling OLS slope (cov/var) and R^2 of each column of ``usd_ret`` on ``dxy_ret``."""
    betas, r2 = {}, {}
    for c in usd_ret.columns:
        # variance over the same days as the covariance (days where the currency has a price)
        var_d = dxy_ret.where(usd_ret[c].notna()).rolling(window, min_periods=min_periods).var()
        cov = usd_ret[c].rolling(window, min_periods=min_periods).cov(dxy_ret)
        corr = usd_ret[c].rolling(window, min_periods=min_periods).corr(dxy_ret)
        betas[c] = cov / var_d
        r2[c] = corr ** 2
    return pd.DataFrame(betas), pd.DataFrame(r2)


def beta_analytics(q: pd.DataFrame, short: int = 63, long: int = 252) -> dict:
    """Everything beta-related, computed once per market-data load."""
    dxy = synthetic_dxy(q)
    lr = log_returns(q).loc[dxy.index]
    d = np.log(dxy).diff()
    usd = -lr

    beta_s, r2_s = rolling_beta(usd, d, short, min_periods=int(short * 0.8))
    beta_l, r2_l = rolling_beta(usd, d, long, min_periods=int(long * 0.8))

    table = pd.DataFrame({
        "beta_63": beta_s.iloc[-1], "r2_63": r2_s.iloc[-1],
        "beta_252": beta_l.iloc[-1], "r2_252": r2_l.iloc[-1],
    })

    # Daily decomposition in currency terms, using yesterday's beta (no look-ahead).
    # Masked to the days the currency itself has a return, so the parts always add up.
    dollar = (-(beta_s.shift(1)).mul(d, axis=0)).fillna(0.0).where(lr.notna())
    resid = lr - dollar
    return {"dxy": dxy, "dxy_ret": d, "lr": lr, "table": table, "dollar": dollar, "resid": resid,
            "beta_ts": beta_s}


def dxy_equivalent(exposure_nonusd: pd.Series, beta: pd.Series) -> float:
    """USD notional of DXY that behaves like the book:  -sum_c E_c * beta_c  (negative = short DXY)."""
    b = beta.reindex(exposure_nonusd.index).fillna(1.0)
    return float(-(exposure_nonusd * b).sum())
