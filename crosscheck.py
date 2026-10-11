"""
Data check against a second Yahoo price series: quoted crosses.

Every price in the app comes from the USD pairs (EURUSD, USDZAR, ...). A cross such as EURZAR can be
computed from them, or downloaded as its own Yahoo series. Arbitrage keeps the two within about 1 bp,
so a big gap means one of the prices is wrong.

    implied EURc = Q[EUR] / Q[c]          from EURUSD and USDc - the prices every number in the app uses
    quoted  EURc = Yahoo 'EURc=X'         downloaded only for this check
    gap (bps)    = (quoted / implied - 1) x 10,000

The check cross for currency c is EURc (EURGBP for EUR itself; SGD crosses for IDR, KRW, MYR - see
universe.CHECK_CROSS). Only held currencies are checked (one extra download: history cached for a day,
latest prices refreshed with the market data).

Reliability test first: a reference that disagrees on normal days cannot judge anything. If the typical
(median) gap over the VaR window is above 5 bps, or 1 day in 10 is above 25 bps, the quoted cross is
treated as unreliable and that currency is reported as 'not checked' instead of raising false alarms.

Flags (dashboard alerts + logs):
    gap today > 50 bps                                   warning  today's P&L / exposure in c may be wrong
    past gap > 200 bps in the VaR window, not explained  warning  a bad price may still be in the VaR history
    past gap > 50 bps                                    listed in Diagnostics (often holiday / fast-market snapshots)
    quoted cross unreliable / missing                    info     that currency is not independently verified
    download failed                                      warning  prices are not independently verified
    bad prints removed by market_data.clean_spikes        info     marked 'confirmed' when the quoted cross agrees
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

import market_data as M
import universe as U
from logger import get_logger, new_error_id

log = get_logger(__name__)

REF = "EUR"              # default reference currency of a check cross
GAP_BPS = 50.0           # gap that gets listed / flags today's price
SEVERE_BPS = 200.0       # past gap inside the VaR window that raises a dashboard warning
RELIABLE_MEDIAN_BPS = 5.0   # a usable reference agrees to within this on a typical day ...
RELIABLE_P90_BPS = 25.0     # ... and to within this on 9 days out of 10
RETRY_AFTER_S = 900      # after a failed download, wait 15 minutes before trying again

_FAILED: dict[tuple[str, ...], float] = {}  # tickers -> time of the last failed download (whole process)


@dataclass
class CrossCheck:
    status: str                                  # ok | flags | skipped | unavailable | error
    message: str                                 # one-line summary shown in Diagnostics
    table: pd.DataFrame                          # one row per check cross
    flagged: pd.DataFrame                        # past days with |gap| > GAP_BPS inside the VaR window
    corrections: pd.DataFrame                    # bad prints removed + whether the quoted cross confirms them
    alerts: list[tuple[str, str]] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


def check_pair(ccy: str) -> str:
    """The quoted cross used to check a currency: EURc, EURGBP for EUR itself, or an override (SGDIDR...)."""
    if ccy in U.CHECK_CROSS:
        return U.CHECK_CROSS[ccy]
    return "EURGBP" if ccy == REF else REF + ccy


def _other_leg(pair: str, ccy: str) -> str:
    base, quote = U.parse_pair(pair)
    return base if quote == ccy else quote


def _legs(pair: str) -> str:
    base, quote = U.parse_pair(pair)
    return " & ".join(U.ticker_of(c).replace("=X", "") for c in (base, quote) if c != U.BASE_CCY)


# ----------------------------------------------------------------------------- download
@st.cache_data(ttl=24 * 3600, show_spinner="Downloading quoted crosses for the data check...")
def _quoted_daily(tickers: tuple[str, ...]) -> pd.DataFrame:
    return M.restamp_daily(M._download_close(list(tickers), period=M.DAILY_PERIOD, interval="1d"), drop_last=True)


@st.cache_data(ttl=900, show_spinner=False)
def _quoted_hourly(tickers: tuple[str, ...], md_stamp: str) -> pd.DataFrame:
    """md_stamp ties this cache to the market-data load, so both latest prices are fetched together."""
    return M.hourly_to_daily_close(M._download_close(list(tickers), period=M.LIVE_PERIOD, interval="60m"))


def load_quoted(pairs: list[str], md: M.MarketData) -> tuple[pd.DataFrame, list[str]]:
    """Quoted cross closes (dates x pairs), on the same date convention as market_data. Returns (panel, notes)."""
    tickers = tuple(sorted(p + "=X" for p in pairs))
    failed_at = _FAILED.get(tickers)
    if failed_at and time.time() - failed_at < RETRY_AFTER_S:
        raise M.DataError(f"the last download failed {(time.time() - failed_at) / 60:.0f} min ago; "
                          "retrying after 15 min")
    try:
        panel = _quoted_daily(tickers)
    except Exception:
        _FAILED[tickers] = time.time()
        raise
    notes: list[str] = []
    if md.live_marks:
        try:
            panel = _quoted_hourly(tickers, f"{md.fetched_at}").combine_first(panel)
        except Exception as exc:
            log.warning("Latest quoted crosses unavailable (%s); only past days are checked", exc)
            notes.append("Latest quoted crosses unavailable - today's prices were not cross-checked.")
    panel = panel.rename(columns=lambda t: t.replace("=X", ""))
    return panel, notes


# ----------------------------------------------------------------------------- comparison
def _verdicts(md: M.MarketData, quoted: pd.DataFrame, ccys: list[str], unreliable: set[str]) -> pd.DataFrame:
    """md.corrections plus a 'cross_check' column: does the quoted cross confirm the removed price was wrong?"""
    corr = md.corrections.copy()
    verdicts = []
    for r in corr.itertuples():
        pair = check_pair(r.ccy)
        if r.ccy not in ccys:
            verdicts.append("not checked (not held)")
            continue
        if r.ccy in unreliable:
            verdicts.append(f"not checked (quoted {pair} unreliable)")
            continue
        quoted_px = quoted[pair].get(r.date, np.nan) if pair in quoted else np.nan
        if pd.isna(quoted_px):
            verdicts.append(f"not checked (no quoted {pair} that day)")
            continue
        row = md.q.loc[r.date].copy()
        row[U.BASE_CCY] = 1.0
        row[r.ccy] = r.bad_q
        base, quote = U.parse_pair(pair)
        raw_gap = (quoted_px / (row[base] / row[quote]) - 1) * 1e4
        verdicts.append(f"confirmed: quoted {pair} was {raw_gap:+,.0f} bps away" if abs(raw_gap) > GAP_BPS
                        else f"not confirmed: quoted {pair} showed the same price ({raw_gap:+,.0f} bps), probably "
                             "built from the same feed; removed because the move fully reversed next day")
    corr["cross_check"] = verdicts
    return corr


def compare(md: M.MarketData, quoted: pd.DataFrame, ccys: list[str], lookback: int) -> CrossCheck:
    """Pure comparison (no downloads) - see the module docstring for the rules."""
    q_full = md.q.assign(**{U.BASE_CCY: 1.0})
    start = md.q.index[max(0, len(md.q) - lookback - 1)]       # first price date of the VaR window
    fixed = set(zip(md.corrections["ccy"], md.corrections["date"]))
    _, quoted_bad = M.clean_spikes(quoted.loc[start:]) if not quoted.empty else (None, pd.DataFrame(columns=["ccy"]))
    quoted_fixed = set(zip(quoted_bad["ccy"], quoted_bad["date"])) if len(quoted_bad) else set()

    rows, flagged, alerts = [], [], []
    today_bad, history_bad, unreliable, unreliable_ccys = [], [], [], set()
    for c in ccys:
        pair = check_pair(c)
        base, quote = U.parse_pair(pair)
        if pair not in quoted or quoted[pair].dropna().empty:
            rows.append({"check": pair, "ccy": c, "legs": _legs(pair), "status": "No data"})
            log.warning("Cross check: Yahoo returned no data for %s=X", pair)
            continue
        implied = q_full[base] / q_full[quote]
        gap = ((quoted[pair] / implied - 1) * 1e4).dropna()
        hist = gap.loc[start:]
        gap_now = float(gap.get(md.last_date, np.nan))
        typical, p90 = float(hist.abs().median()), float(hist.abs().quantile(0.9))
        row = {"check": pair, "ccy": c, "legs": _legs(pair),
               "implied_now": float(implied.get(md.last_date, np.nan)),
               "quoted_now": float(quoted[pair].get(md.last_date, np.nan)), "gap_now_bps": gap_now,
               "typical_gap_bps": typical, "p90_gap_bps": p90, "days_checked": int(len(hist))}

        # 1) Reliability: a reference that disagrees on normal days cannot judge our prices.
        if not (typical <= RELIABLE_MEDIAN_BPS and p90 <= RELIABLE_P90_BPS):
            unreliable.append(f"{c} ({pair}: typical gap {typical:.0f} bps)")
            unreliable_ccys.add(c)
            log.info("Cross check: quoted %s is too noisy to check %s (median %.1f bps, 90th pct %.1f bps)",
                     pair, c, typical, p90)
            rows.append({**row, "status": "⚪ Not checked: reference unreliable"})
            continue

        # 2) Past days inside the VaR window where the two disagree.
        other = _other_leg(pair, c)
        big = hist[hist.abs() > GAP_BPS]
        severe = []
        for d, g in big.items():
            if (c, d) in fixed or (other, d) in fixed:
                note = "bad print removed from our price that day"
            elif (pair, d) in quoted_fixed:
                note = f"bad print in the quoted {pair} (our price looks fine)"
            else:
                note = "unexplained (often a holiday or fast-market snapshot)"
                if abs(g) > SEVERE_BPS:
                    severe.append((d, g))
            flagged.append({"date": d, "check": pair, "ccy": c, "gap_bps": float(g), "note": note})

        # 3) Today's price, then the status for the table.
        if severe:
            history_bad.append(c)
        if pd.notna(gap_now) and abs(gap_now) > GAP_BPS:
            today_bad.append(c)
            status = "❌ Today off"
            msg = (f"{c} price may be wrong now: the quoted {pair} differs from {_legs(pair)} by {gap_now:+,.0f} bps. "
                   f"Check {c} P&L and exposure before trading.")
            alerts.append(("warning", msg))
            log.warning("Cross check: %s", msg)
        elif severe:
            status = "⚠ Check history"
        elif pd.isna(gap_now):
            status = "✅ OK (today not checked)"
        else:
            status = "✅ OK"
        if severe:
            worst_d, worst_g = max(severe, key=lambda x: abs(x[1]))
            msg = (f"{pair}: {len(severe)} day(s) in the VaR window where the quoted cross and the USD legs disagree "
                   f"by more than {SEVERE_BPS:.0f} bps (worst {worst_d:%d %b %Y}, {worst_g:+,.0f} bps). "
                   "A bad price may still be in the VaR history - see Diagnostics.")
            alerts.append(("warning", msg))
            log.warning("Cross check: %s", msg)
        rows.append({**row, "days_over_50bps": int(len(big)),
                     "max_gap_bps": float(hist.abs().max()) if len(hist) else np.nan,
                     "max_gap_date": hist.abs().idxmax().date() if len(hist) else None, "status": status})

    corrections = _verdicts(md, quoted, ccys, unreliable_ccys)
    table = pd.DataFrame(rows).set_index("ccy") if rows else pd.DataFrame()  # EUR and GBP can share EURGBP
    flagged_df = (pd.DataFrame(flagged, columns=["date", "check", "ccy", "gap_bps", "note"])
                  .sort_values("date", ascending=False).reset_index(drop=True))
    today_ok, history_ok = not today_bad, not history_bad
    no_data = [r["check"] for r in rows if r.get("status") == "No data"]
    if no_data:
        alerts.append(("info", f"Cross check: Yahoo has no quoted {', '.join(no_data)} - those currencies were not "
                               "cross-checked."))
    if unreliable:
        alerts.append(("info", "Cross check: Yahoo's quoted cross is too noisy to verify " + "; ".join(unreliable)
                               + " - not independently verified."))
    n_checked = len(rows) - len(no_data) - len(unreliable)
    checked = table[~table["status"].astype(str).str.startswith(("No data", "⚪"))] if len(table) else table
    max_now = checked["gap_now_bps"].abs().max() if len(checked) else np.nan
    message = (f"{n_checked} of {len(rows)} held currencies verified against quoted crosses; largest gap today "
               f"{max_now:.1f} bps; {len(flagged_df)} past day(s) over {GAP_BPS:.0f} bps in the VaR window")
    log.info("Cross check: %s", message)
    return CrossCheck(status="ok" if today_ok and history_ok else "flags", message=message, table=table,
                      flagged=flagged_df, corrections=corrections, alerts=alerts,
                      checks={"Today's prices agree with quoted crosses": today_ok,
                              f"No unexplained > {SEVERE_BPS:.0f} bps gaps in the VaR window": history_ok})


# ----------------------------------------------------------------------------- entry point
def _cleaning_alert(md: M.MarketData, ccys: list[str]) -> list[tuple[str, str]]:
    corr = md.corrections
    if corr.empty:
        return []
    held = corr[corr["ccy"].isin(ccys)]
    held_txt = ("; in held currencies: " + ", ".join(f"{t} ({n})" for t, n in held["ticker"].value_counts().items())
                if len(held) else "; none in held currencies")
    return [("info", f"Data cleaning removed {len(corr)} bad Yahoo price print(s) from history{held_txt}. "
                     "Details in Diagnostics → Data checks.")]


def _empty(md: M.MarketData, status: str, message: str, alerts: list[tuple[str, str]]) -> CrossCheck:
    corr = md.corrections.assign(cross_check="not checked")
    return CrossCheck(status=status, message=message, table=pd.DataFrame(), flagged=pd.DataFrame(),
                      corrections=corr, alerts=alerts)


def run(md: M.MarketData, held: list[str], lookback: int) -> CrossCheck:
    """Cross-check the held currencies. Never raises: any failure becomes a dashboard warning with an error id."""
    ccys = [c for c in held if c in U.CURRENCIES]
    cleaning = _cleaning_alert(md, ccys)
    try:
        if not ccys:
            return _empty(md, "skipped", "No held currencies to cross-check.", cleaning)
        if os.environ.get("FXDASH_OFFLINE") == "1" or md.source != "live":
            reason = "offline mode" if os.environ.get("FXDASH_OFFLINE") == "1" else "live data unavailable"
            return _empty(md, "skipped", f"Cross check skipped ({reason}).",
                          cleaning + [("info", f"Cross check skipped ({reason}) - prices not independently verified.")])
        pairs = sorted({check_pair(c) for c in ccys})
        try:
            quoted, notes = load_quoted(pairs, md)
        except Exception as exc:
            log.warning("Cross check unavailable: could not download %s (%s)", ", ".join(pairs), exc)
            return _empty(md, "unavailable", f"Cross check unavailable: {exc}",
                          cleaning + [("warning", f"Cross check unavailable (download failed: {exc}) - prices "
                                                  "are not independently verified.")])
        result = compare(md, quoted, ccys, lookback)
        result.alerts = cleaning + result.alerts + [("info", n) for n in notes]
        return result
    except Exception as exc:
        err = getattr(exc, "err_id", None) or new_error_id()
        log.exception("[%s] Cross check failed: %s", err, exc)
        return _empty(md, "error", f"Cross check failed - error ID {err}",
                      cleaning + [("warning", f"Cross check failed - error ID {err} (search it in Diagnostics).")])
