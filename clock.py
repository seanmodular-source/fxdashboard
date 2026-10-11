"""
One clock for the whole app: Singapore time (SGT = UTC+8, no daylight saving).

Data arrives from Yahoo in UTC and is stored and computed in UTC; everything shown to the user is in SGT.
Neither UTC nor SGT has daylight saving, so converting is always exactly +8 hours: no number changes,
only the labels.

Trading-day convention (from Yahoo's data, see market_data.py):
    trading day D = 00:00 UTC on D -> 00:00 UTC on D+1 = 08:00 SGT on D -> 08:00 SGT on D+1
    the daily 'close' of D is the price at about 08:00 SGT on D+1
    (not the 5pm New York close, which is 05:00 SGT in the US summer and 06:00 SGT in winter)

FX sessions: fixed clock windows, identical all year, so every day is measured the same way.
    Asia    08:00-16:00 SGT  (00-08 UTC)   Tokyo, Singapore, Sydney
    Europe  16:00-21:00 SGT  (08-13 UTC)   London morning
    US      21:00-08:00 SGT  (13-24 UTC)   London afternoon and New York, until the daily close
London and New York change their clocks in summer, so their real opens move by one hour in SGT
(London 15:00 SGT in summer / 16:00 in winter; New York 20:00 / 21:00). With fixed windows, in summer
the first London hour counts as Asia and the first New York hour as Europe.
"""
from __future__ import annotations

import pandas as pd

TZ = "SGT"
OFFSET = pd.Timedelta(hours=8)

SESSION_HOURS_UTC: dict[str, tuple[int, int]] = {"Asia": (0, 8), "Europe": (8, 13), "US": (13, 24)}
SESSION_SGT: dict[str, str] = {"Asia": "08:00–16:00", "Europe": "16:00–21:00", "US": "21:00–08:00 (+1 day)"}
DAY_SGT = "08:00 SGT to 08:00 SGT the next day"


def _naive_utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def now_sgt() -> pd.Timestamp:
    return now_utc() + OFFSET


def today_sgt() -> pd.Timestamp:
    return now_sgt().normalize()


def trading_day_now() -> pd.Timestamp:
    """The trading day running now. A day runs 08:00 SGT to 08:00 SGT, i.e. it is the UTC date: a trade at
    03:00 SGT on Tuesday belongs to Monday's trading day."""
    return now_utc().normalize()


def to_sgt(ts_utc) -> pd.Timestamp:
    """UTC timestamp (naive = UTC, or tz-aware) -> naive SGT timestamp."""
    return _naive_utc(ts_utc) + OFFSET


def fmt(ts_utc, pattern: str = "%d %b %Y %H:%M") -> str:
    """Format a UTC timestamp in SGT, e.g. '10 Oct 2026 06:00 SGT'."""
    return f"{to_sgt(ts_utc):{pattern}} {TZ}"


def close_time(day) -> str:
    """When the close of trading day ``day`` happens in SGT, e.g. '10 Oct 08:00 SGT'."""
    return f"{pd.Timestamp(day).normalize() + pd.Timedelta(days=1) + pd.Timedelta(hours=8):%d %b %H:%M} {TZ}"
