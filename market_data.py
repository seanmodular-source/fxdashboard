"""
Market data: fetch from yfinance, clean, align, cache, fall back to a bundled snapshot.

Yahoo FX quirks handled here (verified against hourly data, see README):
  * Weekend rows are stray quotes -> dropped.
  * A daily FX bar dated D closes at ~00:00 UTC on D, i.e. it is the close of
    trading day D-1 -> every bar is re-stamped one business day earlier.
  * The most recent daily bar is a live, still-updating quote -> dropped and the
    latest closes are rebuilt from hourly bars (close = last hourly bar of each
    UTC weekday; Sunday-evening bars belong to Monday).
  * Occasional one-day bad prints (e.g. USDZAR +20% then -20% on 14-15 Nov 2024)
    -> removed by clean_spikes() and listed in Diagnostics (see crosscheck.py for
    the independent confirmation against quoted EUR crosses).

Everything is returned in Q terms: Q[c] = USD value of 1 unit of currency c.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import universe as U
from logger import get_logger, log_step

log = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
FALLBACK_DAILY = DATA_DIR / "fallback_prices.csv"
DAILY_PERIOD = "3y"        # 500-day lookback + 250-day backtest needs ~750 rows
LIVE_PERIOD = "7d"         # hourly bars used to rebuild the latest closes
STALE_BDAYS = 3            # a price older than this many business days is flagged
OVERLAP_WARN_BPS = 50.0    # daily-vs-hourly close mismatch that suggests Yahoo changed convention
MIN_ROWS = 60              # currencies with less history are dropped
SPIKE_SIGMA = 6.0          # a bad print jumps more than 6 typical daily moves away ...
SPIKE_REVERT = 0.5         # ... and snaps back: its two neighbours agree to within half that jump
CORRECTION_COLUMNS = ["date", "ccy", "ticker", "bad_price", "clean_price", "jump_in_%", "jump_out_%", "size_sigma",
                      "bad_q", "clean_q"]


class DataError(RuntimeError):
    """Raised when a download returns nothing usable."""


@dataclass
class MarketData:
    q: pd.DataFrame                      # daily closes (Q terms), index = trading dates
    source: str                          # "live" or "fallback"
    live_marks: bool                     # latest row built from hourly bars?
    as_of: pd.Timestamp                  # timestamp of the latest mark (UTC, naive)
    fetched_at: pd.Timestamp             # when the data was loaded (UTC, naive)
    status: pd.DataFrame                 # per-currency data quality table
    notes: list[str] = field(default_factory=list)
    corrections: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CORRECTION_COLUMNS))  # bad prints removed

    @property
    def now(self) -> pd.Series:
        """Latest mark per currency (Q terms) with USD = 1."""
        s = self.q.iloc[-1].copy()
        s[U.BASE_CCY] = 1.0
        return s

    @property
    def prev(self) -> pd.Series:
        """Previous trading-day close per currency (Q terms) with USD = 1."""
        s = self.q.iloc[-2].copy()
        s[U.BASE_CCY] = 1.0
        return s

    @property
    def last_date(self) -> pd.Timestamp:
        return self.q.index[-1]

    @property
    def prev_date(self) -> pd.Timestamp:
        return self.q.index[-2]

    @property
    def last_complete_date(self) -> pd.Timestamp:
        """Latest trading day whose close is final. A day still running (its UTC day, i.e. 08:00 SGT to 08:00 SGT,
        has not ended when the data was loaded) is excluded, so risk never treats a part-day as a full day."""
        running = self.live_marks and self.fetched_at < self.last_date + pd.Timedelta(days=1)
        return self.prev_date if running else self.last_date

    @property
    def currencies(self) -> list[str]:
        return list(self.q.columns)

    def has(self, ccy: str) -> bool:
        """True if the currency has a usable latest price."""
        return ccy == U.BASE_CCY or (ccy in self.q.columns and pd.notna(self.q[ccy].iloc[-1]))

    def q_on(self, date) -> pd.Series:
        """Q row on or before ``date`` (USD = 1)."""
        d = pd.Timestamp(date).normalize()
        sub = self.q.loc[:d]
        row = (sub.iloc[-1] if len(sub) else self.q.iloc[0]).copy()
        row[U.BASE_CCY] = 1.0
        return row

    def pair_price(self, pair: str, row: pd.Series | None = None) -> float:
        """Market price of a pair (quote currency per 1 base) from a Q row (default: latest)."""
        base, quote = U.parse_pair(pair)
        r = self.now if row is None else row
        return float(r[base] / r[quote])

    def simple_returns(self, ccys: list[str]) -> pd.DataFrame:
        """Daily simple returns of Q for ``ccys`` on dates where all of them have prices."""
        if not ccys:
            return pd.DataFrame(index=self.q.index[1:])
        return self.q[ccys].pct_change(fill_method=None).dropna(how="any")


# ----------------------------------------------------------------------------- helpers
def _to_q(close: pd.DataFrame) -> pd.DataFrame:
    """Ticker closes -> Q terms with currency-code columns."""
    out = {}
    for ccy, (ticker, invert, _name) in U.CURRENCIES.items():
        if ticker in close.columns:
            s = close[ticker].astype(float).replace(0.0, np.nan)
            out[ccy] = 1.0 / s if invert else s
    return pd.DataFrame(out)


def _download_close(tickers: list[str], **kwargs) -> pd.DataFrame:
    """yf.download wrapper returning a Close panel with one column per ticker."""
    data = yf.download(tickers=tickers, auto_adjust=False, progress=False, threads=True, **kwargs)
    if data is None or data.empty:
        raise DataError(f"yfinance returned no data for {len(tickers)} tickers ({kwargs})")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.reindex(columns=tickers)
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    close.index = idx
    missing = [t for t in tickers if close[t].dropna().empty]
    if missing:
        log.warning("No data returned for %s", ", ".join(missing))
    return close.sort_index()


def restamp_daily(close: pd.DataFrame, drop_last: bool) -> pd.DataFrame:
    """Drop weekend rows, optionally the live last bar, and re-stamp to the true trading day."""
    df = close[close.index.dayofweek < 5].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz is not None else df.index
    if drop_last and len(df):
        df = df.iloc[:-1]
    df.index = df.index.normalize() - pd.offsets.BDay(1)
    return df[~df.index.duplicated(keep="last")]


def hourly_to_daily_close(hourly: pd.DataFrame) -> pd.DataFrame:
    """Last hourly close per UTC weekday. Weekend (Sunday-evening open) bars count toward Monday."""
    h = hourly.dropna(how="all")
    idx = pd.DatetimeIndex(h.index)
    idx = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    day = idx.normalize()
    shift = np.where(idx.dayofweek == 5, 2, np.where(idx.dayofweek == 6, 1, 0))
    day = day + pd.to_timedelta(shift, unit="D")
    h = h.copy()
    h.index = idx
    return h.groupby(day).last()


def clean_spikes(q: pd.DataFrame, k: float = SPIKE_SIGMA, revert: float = SPIKE_REVERT
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove one-day bad prints. Returns (clean prices, one row per price that was replaced).

    A price on day t is a bad print when all of these hold (log prices; sigma = typical daily move):
        |jump in| > k * sigma  and  |jump out| > k * sigma        a huge move into the price and out of it
        the two jumps have opposite signs                          it reverses
        |price(t+1) - price(t-1)| < revert * smaller jump          the day before and the day after agree
    sigma is robust (median absolute daily move / 0.6745), so the bad prints cannot inflate it.
    The bad price is replaced by the geometric midpoint of its neighbours.

    A real jump that holds (SNB 2015, a devaluation) is never touched: its neighbours disagree.
    The latest price has no 'next day', so it is never changed here; crosscheck.py covers it.
    Works on any panel of prices (also used on Yahoo's quoted crosses).
    """
    out = q.copy()
    rows = []
    for col in q.columns:
        s = q[col].dropna()
        if len(s) < 3:
            continue
        lp = np.log(s)
        sigma = float(lp.diff().abs().median() / 0.6745)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        for _ in range(5):  # repeat: back-to-back bad prints are uncovered one pass at a time
            jump_in = lp - lp.shift(1)
            jump_out = lp.shift(-1) - lp
            span = (lp.shift(-1) - lp.shift(1)).abs()
            bad = ((jump_in.abs() > k * sigma) & (jump_out.abs() > k * sigma)
                   & (np.sign(jump_in) != np.sign(jump_out))
                   & (span < revert * np.minimum(jump_in.abs(), jump_out.abs())))
            bad &= ~bad.shift(1, fill_value=False)  # never fix two neighbouring days in the same pass
            if not bad.any():
                break
            for t in bad.index[bad]:
                i = lp.index.get_loc(t)
                fixed = (lp.iloc[i - 1] + lp.iloc[i + 1]) / 2
                rows.append({"date": t, "ccy": col, "bad_q": float(np.exp(lp.iloc[i])), "clean_q": float(np.exp(fixed)),
                             "size_sigma": float(abs(jump_in[t]) / sigma)})
                lp.iloc[i] = fixed
                out.loc[t, col] = float(np.exp(fixed))
    corr = pd.DataFrame(rows, columns=["date", "ccy", "bad_q", "clean_q", "size_sigma"])
    if corr.empty:
        return out, pd.DataFrame(columns=CORRECTION_COLUMNS)

    # Show prices and moves the way Yahoo quotes them (USDZAR, not ZAR per USD) so they can be looked up.
    def quoted(ccy: str, value: float) -> float:
        return 1.0 / value if (ccy in U.CURRENCIES and U.CURRENCIES[ccy][1]) else value

    corr["ticker"] = [U.ticker_of(c).replace("=X", "") if c in U.CURRENCIES else c for c in corr["ccy"]]
    corr["bad_price"] = [quoted(c, v) for c, v in zip(corr["ccy"], corr["bad_q"])]
    corr["clean_price"] = [quoted(c, v) for c, v in zip(corr["ccy"], corr["clean_q"])]
    ins, outs = [], []
    for r in corr.itertuples():
        s = out[r.ccy].dropna()  # cleaned neighbours (matters only when two bad prints sit side by side)
        i = s.index.get_loc(r.date)
        prev_p, next_p = quoted(r.ccy, s.iloc[i - 1]), quoted(r.ccy, s.iloc[i + 1])
        ins.append((r.bad_price / prev_p - 1) * 100)
        outs.append((next_p / r.bad_price - 1) * 100)
    corr["jump_in_%"], corr["jump_out_%"] = ins, outs
    return out, corr[CORRECTION_COLUMNS].sort_values(["date", "ccy"]).reset_index(drop=True)


def _status_table(q_raw: pd.DataFrame, source: str) -> pd.DataFrame:
    last_date = q_raw.index.max()
    rows = []
    for ccy in q_raw.columns:
        s = q_raw[ccy].dropna()
        lv = s.index.max() if len(s) else pd.NaT
        lag = int(np.busday_count(lv.date(), last_date.date())) if pd.notna(lv) else 999
        rows.append({
            "ccy": ccy, "ticker": U.ticker_of(ccy), "rows": int(len(s)),
            "first": s.index.min().date() if len(s) else None, "last": lv.date() if pd.notna(lv) else None,
            "stale": lag > STALE_BDAYS, "source": source,
        })
    return pd.DataFrame(rows).set_index("ccy")


def _finalise(q_raw: pd.DataFrame, source: str, live_marks: bool, as_of: pd.Timestamp, notes: list[str]) -> MarketData:
    q_raw = q_raw.sort_index()
    keep = [c for c in q_raw.columns if q_raw[c].count() >= MIN_ROWS]
    dropped = sorted(set(q_raw.columns) - set(keep))
    if dropped:
        log.warning("Dropped currencies with < %d rows: %s", MIN_ROWS, ", ".join(dropped))
        notes.append(f"Not enough history for: {', '.join(dropped)}")
    q_raw = q_raw[keep]
    status = _status_table(q_raw, source)
    stale = status.index[status["stale"]].tolist()
    if stale:
        log.warning("Stale prices (> %d business days old): %s", STALE_BDAYS, ", ".join(stale))
    q_raw, corrections = clean_spikes(q_raw)
    status["bad_prints"] = corrections.groupby("ccy").size().reindex(status.index).fillna(0).astype(int)
    if len(corrections):
        log.warning("Removed %d bad price prints from %s history: %s", len(corrections), source,
                    "; ".join(f"{t} {d:%Y-%m-%d} ({a:+.1f}% then {b:+.1f}%)" for t, d, a, b in
                              zip(corrections["ticker"], corrections["date"], corrections["jump_in_%"],
                                  corrections["jump_out_%"])))
    q = q_raw.ffill(limit=STALE_BDAYS)
    q = q.dropna(how="all")
    md = MarketData(q=q, source=source, live_marks=live_marks, as_of=as_of,
                    fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None), status=status, notes=notes,
                    corrections=corrections)
    log.info("Market data ready: source=%s rows=%d ccys=%d last=%s as_of=%s",
             source, len(q), q.shape[1], md.last_date.date(), as_of)
    return md


def _load_fallback(reason: str) -> MarketData:
    log.warning("Using fallback snapshot %s (%s)", FALLBACK_DAILY.name, reason)
    if not FALLBACK_DAILY.exists():
        raise DataError(f"Live data failed ({reason}) and no fallback file at {FALLBACK_DAILY}")
    q = pd.read_csv(FALLBACK_DAILY, index_col=0, parse_dates=True)
    notes = [f"Live data unavailable ({reason}). Showing bundled snapshot."]
    return _finalise(q, "fallback", live_marks=False, as_of=q.index.max(), notes=notes)


@log_step
def _load_live() -> MarketData:
    tickers = [U.ticker_of(c) for c in U.all_currencies()]
    daily_close = _download_close(tickers, period=DAILY_PERIOD, interval="1d")
    q_daily = _to_q(restamp_daily(daily_close, drop_last=True))
    notes: list[str] = []

    try:
        hourly = _download_close(tickers, period=LIVE_PERIOD, interval="60m")
        q_hourly = _to_q(hourly_to_daily_close(hourly))
        as_of = pd.DatetimeIndex(hourly.dropna(how="all").index).tz_localize(None).max()
        # Sanity check of the re-stamping convention on liquid G10 pairs.
        common = q_daily.index.intersection(q_hourly.index)
        g10 = [c for c in ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF"] if c in q_daily and c in q_hourly]
        if len(common) and g10:
            diff_bps = ((q_hourly.loc[common, g10] / q_daily.loc[common, g10] - 1).abs() * 1e4).median().median()
            log.info("Daily-vs-hourly close check on %d dates: median |diff| = %.1f bps", len(common), diff_bps)
            if diff_bps > OVERLAP_WARN_BPS:
                log.warning("Daily vs hourly closes differ by %.0f bps - Yahoo bar convention may have changed",
                            diff_bps)
                notes.append("Daily and hourly Yahoo closes disagree; check Diagnostics.")
        q = q_hourly.combine_first(q_daily)
        live_marks = True
    except Exception as exc:  # hourly is a nice-to-have: degrade to daily closes
        log.warning("Hourly marks unavailable (%s); prices are as of the last completed daily close", exc)
        notes.append("Latest intraday marks unavailable - prices are the last completed daily close.")
        q = q_daily
        as_of = q.index.max()
        live_marks = False
    return _finalise(q, "live", live_marks, pd.Timestamp(as_of), notes)


@st.cache_data(ttl=900, show_spinner="Loading market data...")
def load_market_data(offline: bool = False) -> MarketData:
    """Cached entry point used by the app. ``offline`` (or env FXDASH_OFFLINE=1) forces the snapshot."""
    if offline or os.environ.get("FXDASH_OFFLINE") == "1":
        return _load_fallback("offline mode")
    try:
        return _load_live()
    except Exception as exc:
        return _load_fallback(f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------- long history (seasonality)
@st.cache_data(ttl=24 * 3600, show_spinner="Loading 10 years of history...")
def load_long_history(ccys: tuple[str, ...], period: str = "10y") -> tuple[pd.DataFrame, str]:
    """Daily Q closes for ``ccys`` over ``period``, same date fix and bad-print filter as the main load.
    Returns (prices, source note). Falls back to the bundled 3-year snapshot if Yahoo is unavailable."""
    ccys = tuple(c for c in ccys if c in U.CURRENCIES)
    if not ccys:
        return pd.DataFrame(), "no currencies"
    if os.environ.get("FXDASH_OFFLINE") != "1":
        try:
            close = _download_close([U.ticker_of(c) for c in ccys], period=period, interval="1d")
            q, corrections = clean_spikes(_to_q(restamp_daily(close, drop_last=True)))
            log.info("Long history %s for %s: %d rows, %d bad prints removed", period, ",".join(ccys), len(q),
                     len(corrections))
            return q, f"Yahoo daily closes, {period} ({q.index.min():%b %Y} – {q.index.max():%b %Y})"
        except Exception as exc:
            log.warning("Long history unavailable for %s (%s); using the bundled snapshot", ccys, exc)
    q, _ = clean_spikes(pd.read_csv(FALLBACK_DAILY, index_col=0, parse_dates=True))
    q = q[[c for c in ccys if c in q.columns]]
    return q, f"bundled snapshot only ({q.index.min():%b %Y} – {q.index.max():%b %Y}); 10-year download unavailable"


# ----------------------------------------------------------------------------- hourly (sessions)
@st.cache_data(ttl=900, show_spinner="Loading hourly data...")
def load_hourly_q(ccys: tuple[str, ...], period: str = "180d") -> pd.DataFrame:
    """Hourly closes in Q terms (UTC index) for the session analysis. Empty frame on failure."""
    ccys = tuple(c for c in ccys if c in U.CURRENCIES)
    if not ccys or os.environ.get("FXDASH_OFFLINE") == "1":
        return pd.DataFrame()
    try:
        close = _download_close([U.ticker_of(c) for c in ccys], period=period, interval="60m")
    except Exception as exc:
        log.warning("Hourly session data unavailable for %s: %s", ccys, exc)
        return pd.DataFrame()
    q = _to_q(close)
    q.index = pd.DatetimeIndex(q.index).tz_localize(None) if q.index.tz is not None else q.index
    log.info("Hourly data for sessions: %d bars x %d ccys", len(q), q.shape[1])
    return q
