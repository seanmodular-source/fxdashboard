"""
Session strength (Asia / Europe / US) - port of the FX Seasonality notebook, in currency terms.

    Sessions (fixed clock windows, see clock.py):
        Asia 08:00-16:00 SGT (00-08 UTC) · Europe 16:00-21:00 SGT (08-13 UTC) · US 21:00-08:00 SGT (13-24 UTC)
    Trading day = 08:00 SGT to 08:00 SGT next day; the Sunday-evening (UTC) open, Monday 05:00 SGT, counts
    as Monday's Asia session.

    Session return  = sum of the hourly LOG returns inside the window = ln(price at end / price at start)
                      Log returns add up exactly: Asia + Europe + US = the day's close-to-close log move,
                      so the stacked bars are an exact split of the day.
    Cumulative      = sum of the session's log returns over the period, shown as a % move: exp(sum) - 1
    Session z       = (today - mean of the prior N days) / std of the prior N days   (N typed by the user,
                      default 90; shift(1) so today never sits in its own yardstick; needs min(N, 20) days)
    Hit rate        = share of sessions that went up, ignoring sessions with no data

Differences vs the notebook (deliberate):
  * Log returns instead of compounded % returns, so sessions add up exactly (for small moves the two are
    almost identical: a 0.50% move is a 0.499% log move).
  * Weekend bars (Sunday-evening open) are assigned to Monday's Asia session instead of a 'Sunday' row.
  * Hit rate excludes missing sessions (the notebook counted them as down sessions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clock import SESSION_HOURS_UTC
from logger import get_logger

log = get_logger(__name__)

ZONES = ["Asia", "Europe", "US"]
LOOKBACK = 90
MIN_PERIODS = 20
Z_FLAG = 1.0


def _zone(hours: np.ndarray) -> np.ndarray:
    return np.select([hours < SESSION_HOURS_UTC["Asia"][1], hours < SESSION_HOURS_UTC["Europe"][1]],
                     ["Asia", "Europe"], "US")


def trading_day(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC date of each bar; Saturday / Sunday bars (the Sunday-evening open) belong to Monday."""
    shift = np.where(idx.dayofweek == 5, 2, np.where(idx.dayofweek == 6, 1, 0))
    return idx.normalize() + pd.to_timedelta(shift, unit="D")


def bar_zone(idx: pd.DatetimeIndex) -> np.ndarray:
    """Session of each hourly bar (UTC hour of the bar's start); weekend bars are Monday's Asia."""
    return np.where(idx.dayofweek >= 5, "Asia", _zone(idx.hour))


def session_returns(hourly_q: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """(session log returns per day in %, daily close-to-close log return in %) for one currency."""
    s = hourly_q.dropna()
    idx = pd.DatetimeIndex(s.index)
    lr = np.log(s).diff()
    df = pd.DataFrame({"ret": lr.values, "day": trading_day(idx), "zone": bar_zone(idx)}, index=idx)
    df = df.dropna(subset=["ret"])
    zr = df.groupby(["day", "zone"])["ret"].sum().unstack("zone").reindex(columns=ZONES)
    daily_close = s.groupby(trading_day(idx)).last()
    daily_ret = np.log(daily_close).diff()
    implied = zr.sum(axis=1, min_count=1)
    gap_bps = ((implied - daily_ret).abs() * 1e4).iloc[1:].max()
    if pd.notna(gap_bps) and gap_bps > 0.5:
        log.warning("Session returns do not reconcile to daily (max gap %.2f bps)", gap_bps)
    return zr * 100, daily_ret * 100


def session_zscores(zone_ret_pct: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    prior = zone_ret_pct.shift(1).rolling(lookback, min_periods=min(MIN_PERIODS, lookback))
    return (zone_ret_pct - prior.mean()) / prior.std()


def session_summary(zone_ret_pct: pd.DataFrame) -> pd.DataFrame:
    r = zone_ret_pct / 100
    return pd.DataFrame({
        "avg_%": zone_ret_pct.mean(),
        "cumulative_%": (np.exp(r.sum()) - 1) * 100,
        "hit_rate_%": r.apply(lambda col: (col.dropna() > 0).mean() * 100),
        "days": r.count(),
    }).reindex(ZONES)


def book_session_pnl(hourly_q: pd.DataFrame, exposure_nonusd: pd.Series, days: int = 20) -> pd.DataFrame:
    """USD P&L of TODAY's exposures by session for the last ``days`` days: sum_c E_c * (exp(r_c,session) - 1)."""
    frames = {}
    for c in exposure_nonusd.index:
        if c in hourly_q.columns:
            zr, _ = session_returns(hourly_q[c])
            frames[c] = (np.exp(zr / 100) - 1) * exposure_nonusd[c]
    if not frames:
        return pd.DataFrame(columns=ZONES)
    total = pd.concat(frames.values()).groupby(level=0).sum(min_count=1).sort_index()
    return total.iloc[-days:]
