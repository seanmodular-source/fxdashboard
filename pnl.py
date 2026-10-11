"""
USD P&L, per trade and per pair.

MTM_USD(trade) = N * Q_base  -  N * P_trade * Q_quote          (value of both legs in USD)
which equals the textbook pair formulas:
    xxxUSD : N (S - P)                  e.g. long 1m EURUSD 1.10 -> 1.12 = +$20,000
    USDxxx : N (S - P) / S              e.g. short 3m USDJPY 152.40 -> 150 = +$48,000
    cross  : N (S - P) * Q_quote        e.g. EURGBP P&L in GBP converted at GBPUSD

Inception P&L = MTM_USD at the latest mark.
Daily P&L     = MTM_USD(latest) - MTM_USD(previous close)  (previous-close FX rates, so it is exact),
                or the full MTM_USD for trades done after the previous close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import universe as U


def _q(q_row: pd.Series, ccys: pd.Series) -> pd.Series:
    q = q_row.copy()
    q[U.BASE_CCY] = 1.0
    return ccys.map(q).astype(float)


def mtm_usd(live: pd.DataFrame, q_row: pd.Series) -> pd.Series:
    """Leg-based mark-to-market in USD for each live trade."""
    if live.empty:
        return pd.Series(dtype=float)
    price = live["price"].astype(float)
    return live["signed"] * _q(q_row, live["base"]) - live["signed"] * price * _q(q_row, live["quote"])


def mtm_usd_pair_formula(live: pd.DataFrame, q_row: pd.Series) -> pd.Series:
    """Independent check: N * (S - P) converted from quote currency to USD."""
    if live.empty:
        return pd.Series(dtype=float)
    qb, qq = _q(q_row, live["base"]), _q(q_row, live["quote"])
    spot = qb / qq
    return live["signed"] * (spot - live["price"].astype(float)) * qq


def trade_pnl(live: pd.DataFrame, q_now: pd.Series, q_prev: pd.Series, prev_date: pd.Timestamp) -> pd.DataFrame:
    """Per-trade inception and daily USD P&L, plus the current pair price."""
    if live.empty:
        return live.assign(price_now=pd.Series(dtype=float), inception_pnl=pd.Series(dtype=float),
                           daily_pnl=pd.Series(dtype=float))
    out = live.copy()
    out["price_now"] = _q(q_now, out["base"]) / _q(q_now, out["quote"])
    out["inception_pnl"] = mtm_usd(out, q_now)
    prev_mtm = mtm_usd(out, q_prev)
    new_today = pd.to_datetime(out["trade_date"]) > pd.Timestamp(prev_date)
    out["daily_pnl"] = np.where(new_today, out["inception_pnl"], out["inception_pnl"] - prev_mtm)
    return out


def pnl_by_pair(trade_pnl_df: pd.DataFrame) -> pd.DataFrame:
    if trade_pnl_df.empty:
        return pd.DataFrame(columns=["daily_pnl", "inception_pnl"])
    return trade_pnl_df.groupby("pair")[["daily_pnl", "inception_pnl"]].sum()
