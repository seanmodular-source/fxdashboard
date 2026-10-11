"""
P&L attribution: where is the book making or losing money? Both views add up exactly to the headline P&L;
there is no left-over 'execution' line (no fees or slippage are modelled).

Entry price of each currency. A trade fixes only the pair price P = Q_base / Q_quote, so the USD price each
currency was bought or sold at is implied by P:
        xxxUSD (e.g. AUDUSD): Q_base,entry  = P                    (USD = 1)
        USDxxx (e.g. USDJPY): Q_quote,entry = 1 / P
        cross  (e.g. EURGBP): Q_quote,entry = Q_quote at the trade-date close, Q_base,entry = P x Q_quote,entry
These always satisfy Q_base,entry / Q_quote,entry = P, so the two legs are worth exactly 0 at entry.

1) By currency - every trade is two currency amounts a_c (e.g. +4,000,000 AUD and -2,797,680 USD):
        inception P&L of a leg = a_c x (Q_c,now - Q_c,entry)
        daily P&L of a leg     = a_c x (Q_c,now - Q_c,previous close)   (a trade done since then: from entry)
   Summed over the legs this is exactly the trade's P&L. USD legs never have FX P&L (Q_USD = 1).
   A pair against USD puts all of its P&L in its currency (NZD row = NZDUSD position); a cross is split.

2) By FX session and weekday - from hourly prices.
   A trade enters in the hourly bar it was done in: its booked-at time if that falls on its trade date, otherwise
   (back-dated or no time) the trade date's close. In that bar it makes a_c x (Q_c,bar close - Q_c,entry); from the
   next bar on it makes a_c x (change in Q_c over each hour). Bars are grouped by session (Asia / Europe / US, fixed
   SGT windows) and by weekday, so sessions + P&L before the hourly history = inception P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import clock
import universe as U
from analytics import sessions as SS
from logger import get_logger
from market_data import MarketData

log = get_logger(__name__)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def entry_legs(t, q0: pd.Series) -> list[tuple[str, float, float]]:
    """(currency, amount, USD price at entry) for both legs of one trade; q0 = Q row at the trade-date close."""
    price = float(t.price)
    if t.quote == U.BASE_CCY:
        qb, qq = price, 1.0
    elif t.base == U.BASE_CCY:
        qb, qq = 1.0, 1.0 / price
    else:
        qq = float(q0[t.quote])
        qb = price * qq
    return [(t.base, t.signed, qb), (t.quote, -t.signed * price, qq)]


def pnl_by_currency(live: pd.DataFrame, md: MarketData) -> pd.DataFrame:
    """Rows = currencies, columns = daily_pnl, inception_pnl (USD). Adds up exactly to the headline."""
    if live.empty:
        return pd.DataFrame(columns=["daily_pnl", "inception_pnl"])
    q_now, q_prev = md.now, md.prev
    rows: list[tuple[str, float, float]] = []
    for t in live.itertuples():
        new = pd.Timestamp(t.trade_date) > md.prev_date
        for c, a, q_entry in entry_legs(t, md.q_on(t.trade_date)):
            start = q_entry if new else q_prev[c]
            rows.append((c, a * (q_now[c] - start), a * (q_now[c] - q_entry)))
    df = pd.DataFrame(rows, columns=["ccy", "daily_pnl", "inception_pnl"]).groupby("ccy").sum()
    return df.drop(U.BASE_CCY, errors="ignore").sort_values("inception_pnl")


def trade_time_utc(timestamp, trade_date) -> pd.Timestamp:
    """When the trade was done (UTC): its booked-at time if that falls on its trade date (the trading day is the
    UTC date, 08:00 to 08:00 SGT), otherwise the trade date's close (a back-dated trade, or no time given)."""
    day = pd.Timestamp(trade_date).normalize()
    close = day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    s = str(timestamp or "").strip()
    try:
        if s.endswith("SGT"):
            t = pd.Timestamp(s[:-3].strip()) - clock.OFFSET
        else:
            t = pd.Timestamp(s.replace("UTC", "").strip())
            t = t.tz_convert(None) if t.tzinfo is not None else t
    except (ValueError, TypeError):
        return close
    return t if pd.notna(t) and t.normalize() == day else close


@dataclass
class SessionAttribution:
    start: pd.Timestamp                 # first trading day covered
    end: pd.Timestamp                   # last trading day covered
    by_session_ccy: pd.DataFrame        # sessions x currencies (USD)
    by_weekday: pd.DataFrame            # Mon..Fri: total, avg per day, % up days, days
    daily: pd.Series                    # total P&L per trading day (hourly-based)
    before_window: float                # P&L of trades done before the hourly history starts, up to its start
    mark_gap: float                     # inception P&L at the dashboard mark minus at the last hourly bar


def session_attribution(live: pd.DataFrame, md: MarketData, hourly_q: pd.DataFrame) -> SessionAttribution | None:
    """Actual P&L of the book by session and weekday (see module docstring). None when no data overlaps."""
    if live.empty or hourly_q.empty:
        return None
    legs = [(t.trade_id, c, a, q_entry, trade_time_utc(getattr(t, "timestamp", ""), t.trade_date))
            for t in live.itertuples() for c, a, q_entry in entry_legs(t, md.q_on(t.trade_date))
            if c != U.BASE_CCY]
    ccys = sorted({c for _, c, *_ in legs if c in hourly_q.columns})
    if not ccys:
        return None
    h = hourly_q[ccys].sort_index().ffill().bfill()
    idx = pd.DatetimeIndex(h.index)
    n = len(idx)

    start_amt = np.zeros((n, len(ccys)))        # amount that starts being held at each bar
    entry_pnl = np.zeros((n, len(ccys)))        # entry bar: amount x (bar close - entry price)
    before_window = 0.0
    first_bar = n
    for _, c, a, q_entry, when in legs:
        if c not in ccys:
            continue
        j = ccys.index(c)
        pos = int(idx.searchsorted(when, side="right")) - 1     # bar the trade was done in
        if pos < 0:                                             # before the hourly history: held from bar 0
            start_amt[0, j] += a
            before_window += a * (h.iloc[0, j] - q_entry)
            first_bar = 0
        else:
            entry_pnl[pos, j] += a * (h.iloc[pos, j] - q_entry)
            if pos + 1 < n:
                start_amt[pos + 1, j] += a
            first_bar = min(first_bar, pos)
    if first_bar >= n:
        return None
    hold = np.cumsum(start_amt, axis=0)
    moves = h.diff().fillna(0.0).values
    bar_pnl = pd.DataFrame(hold * moves + entry_pnl, index=idx, columns=ccys)

    day, zone = SS.trading_day(idx), SS.bar_zone(idx)
    keep = day >= day[first_bar]
    bar_pnl, k_day, k_zone = bar_pnl[keep], day[keep], zone[keep]
    by_session_ccy = bar_pnl.groupby(k_zone).sum().reindex(SS.ZONES).fillna(0.0)
    daily = bar_pnl.sum(axis=1).groupby(k_day).sum()
    wd = pd.Series(daily.index.dayofweek, index=daily.index).map(dict(enumerate(WEEKDAYS)))
    by_weekday = pd.DataFrame({
        "total": daily.groupby(wd).sum(), "avg_per_day": daily.groupby(wd).mean(),
        "up_days_%": daily.groupby(wd).apply(lambda x: (x > 0).mean() * 100), "days": daily.groupby(wd).count(),
    }).reindex(WEEKDAYS)

    # sessions + before_window = P&L at the last hourly bar; the gap to the headline is the dashboard mark vs that bar
    q_now = md.now
    mark_gap = float(sum(a * (q_now[c] - h[c].iloc[-1]) for _, c, a, _, _ in legs if c in ccys))
    log.info("Session attribution %s..%s: sessions %.0f + before window %.0f, mark gap %.2f",
             daily.index.min().date(), daily.index.max().date(), daily.sum(), before_window, mark_gap)
    return SessionAttribution(start=daily.index.min(), end=daily.index.max(), by_session_ccy=by_session_ccy,
                              by_weekday=by_weekday, daily=daily, before_window=before_window, mark_gap=mark_gap)
