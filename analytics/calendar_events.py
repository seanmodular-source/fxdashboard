"""
Calendar events for a pair - port of the FX Seasonality notebook.

    HD  public holiday in the country of either currency (holidays package: USD->US, JPY->JP, EUR->DE, ...)
    ME  month-end (last calendar day of the month)
    MM  mid-month (the 15th)
    FS / FE  fiscal-year start / end of either currency (JPY, GBP, INR, SGD: Apr-Mar; USD, THB: Oct-Sep;
             AUD, NZD: Jul-Jun; others the calendar year)

An event on a non-trading day is shown on the next trading day ('plotted_date'), as in the notebook:
a Saturday month-end shows up on Monday.

Source: public holidays come from the open-source Python package `holidays` (PyPI 'holidays', maintained by
vacanza; country calendars such as holidays.country_holidays("JP")). Month-end, mid-month and fiscal dates are
computed here from the rules above.
"""
from __future__ import annotations

import holidays
import pandas as pd

import universe as U
from logger import get_logger

log = get_logger(__name__)

CODES = {"HD": "Holiday", "ME": "Month-End", "MM": "Mid-Month", "FS": "Fiscal start", "FE": "Fiscal end"}

COUNTRY: dict[str, str] = {
    "USD": "US", "JPY": "JP", "THB": "TH", "EUR": "DE", "GBP": "GB", "AUD": "AU", "NZD": "NZ", "CAD": "CA",
    "CHF": "CH", "SGD": "SG", "KRW": "KR", "TWD": "TW", "INR": "IN", "MYR": "MY", "IDR": "ID", "PHP": "PH",
    "ZAR": "ZA", "MXN": "MX", "NOK": "NO", "SEK": "SE", "PLN": "PL", "CZK": "CZ", "HUF": "HU", "TRY": "TR",
    "BRL": "BR", "CLP": "CL", "COP": "CO",
}

# (start month, start day, end month, end day) of each currency's fiscal year; calendar year otherwise.
FISCAL: dict[str, tuple[int, int, int, int]] = {
    "USD": (10, 1, 9, 30), "JPY": (4, 1, 3, 31), "THB": (10, 1, 9, 30), "GBP": (4, 1, 3, 31),
    "AUD": (7, 1, 6, 30), "NZD": (7, 1, 6, 30), "INR": (4, 1, 3, 31), "SGD": (4, 1, 3, 31),
}


def _holidays(country: str, years: list[int]) -> dict:
    try:
        return holidays.country_holidays(country, years=years, language="en_US")
    except Exception:  # language not offered for this country -> local names
        try:
            return holidays.country_holidays(country, years=years)
        except Exception as exc:
            log.warning("No holiday calendar for %s: %s", country, exc)
            return {}


def events_between(pair: str, start, end) -> pd.DataFrame:
    """All events for ``pair`` with start <= event_date <= end (calendar days)."""
    base, quote = U.parse_pair(pair)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = pd.date_range(start, end, freq="D")
    rows = []
    for d in days:
        if d.is_month_end:
            rows.append((d, "ME", "Month-End"))
        if d.day == 15:
            rows.append((d, "MM", "Mid-Month"))
    for ccy in (base, quote):
        sm, sd, em, ed = FISCAL.get(ccy, (1, 1, 12, 31))
        for d in days:
            if d.month == sm and d.day == sd:
                rows.append((d, "FS", f"{ccy} fiscal year start"))
            if d.month == em and d.day == ed:
                rows.append((d, "FE", f"{ccy} fiscal year end"))
    years = sorted(set(days.year))
    for ccy in dict.fromkeys((base, quote)):
        country = COUNTRY.get(ccy)
        if not country:
            continue
        for dt, name in _holidays(country, years).items():
            ts = pd.Timestamp(dt)
            if start <= ts <= end:
                rows.append((ts, "HD", f"{country} holiday: {name}"))
    out = pd.DataFrame(rows, columns=["event_date", "event_code", "event_name"])
    return out.drop_duplicates().sort_values(["event_date", "event_code", "event_name"]).reset_index(drop=True)


def map_to_trading_days(events: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Add 'plotted_date' = first trading day on or after the event; events after the last day are dropped."""
    if events.empty:
        return events.assign(plotted_date=pd.Series(dtype="datetime64[ns]"))
    idx = pd.DatetimeIndex(trading_days).normalize().unique().sort_values()
    pos = idx.searchsorted(pd.DatetimeIndex(events["event_date"]))
    keep = pos < len(idx)
    out = events.loc[keep].copy()
    out["plotted_date"] = idx[pos[keep]]
    return out.sort_values(["plotted_date", "event_code", "event_name"]).reset_index(drop=True)


def upcoming(pair: str, today, days: int = 7) -> pd.DataFrame:
    today = pd.Timestamp(today).normalize()
    return events_between(pair, today, today + pd.Timedelta(days=days))


def title_line(up: pd.DataFrame, days: int = 7, max_items: int = 5) -> str:
    if up.empty:
        return f"Next {days}D: none"
    parts = [f"{d:%d-%b} {c}" for d, c in zip(up["event_date"], up["event_code"])][:max_items]
    return f"Next {days}D: " + " · ".join(parts) + (" …" if len(up) > max_items else "")
