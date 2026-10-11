"""
Rich / cheap versus each currency's own history (currency terms: + = currency strong / expensive).

    Level z    = (ln Q_t - mean(ln Q over the prior N days)) / std(ln Q over the prior N days)
                 N = 21, 63, 126, 252 trading days (1M, 3M, 6M, 1Y); today is never in its own yardstick
    Move z     = h-day log move / (daily sigma x sqrt(h)), sigma from the 63 days BEFORE the move
                 h = 5, 21, 63, 126, 252 (1W, 1M, 3M, 6M, 1Y)
    % change   = (Q_t / Q_t-h - 1) x 100
A level z for 1W is not shown: an average of the last few days is almost today's price, so the measure is
noise. The 1W move is covered by move z and % change.

Log prices make the measure symmetric: z(USDJPY) = -z(JPYUSD) exactly.

pair_prices() builds any pair from the USD legs (AUDNZD = Q[AUD] / Q[NZD]); used by Seasonality and Markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import universe as U


def pair_prices(q: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    """Price history of any pair from the USD legs: base/quote = Q[base] / Q[quote] (Q[USD] = 1)."""
    full = q.assign(**{U.BASE_CCY: 1.0})
    cols = {}
    for p in pairs:
        base, quote = U.parse_pair(p)
        if base in full and quote in full:
            cols[p] = full[base] / full[quote]
    return pd.DataFrame(cols, index=q.index)


def level_z(log_q: pd.DataFrame, window: int) -> pd.Series:
    prior = log_q.shift(1).rolling(window, min_periods=int(window * 0.8))
    z = (log_q - prior.mean()) / prior.std()
    return z.iloc[-1]


def stretch_z(log_q: pd.DataFrame, h: int, vol_window: int = 63) -> pd.Series:
    if len(log_q) < h + vol_window + 1:
        return pd.Series(np.nan, index=log_q.columns)
    lr = log_q.diff()
    move = log_q.iloc[-1] - log_q.iloc[-1 - h]
    vol = lr.iloc[-(h + vol_window):-h].std()
    return move / (vol * np.sqrt(h))


LEVEL_WINDOWS: dict[str, int] = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
MOVE_HORIZONS: dict[str, int] = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def rich_cheap_table(q: pd.DataFrame) -> pd.DataFrame:
    """level_z_{N}, stretch_z_{h} (move z) and ret_{h}d (% change) for every column of a price panel."""
    log_q = np.log(q)
    cols = {f"level_z_{n}": level_z(log_q, n) for n in LEVEL_WINDOWS.values()}
    cols.update({f"stretch_z_{h}": stretch_z(log_q, h) for h in MOVE_HORIZONS.values()})
    for h in MOVE_HORIZONS.values():
        cols[f"ret_{h}d"] = (q.iloc[-1] / q.iloc[-1 - h] - 1) * 100 if len(q) > h else pd.Series(np.nan, index=q.columns)
    return pd.DataFrame(cols)
