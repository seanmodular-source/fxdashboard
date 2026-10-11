"""
Seasonality of a pair: how does it usually behave at this time of the year, or this time of the month?

Units: plain % changes, (price / starting price - 1) x 100. Each number is one change over one period, so %
is the most intuitive unit here. (Log moves are used elsewhere in the app, where moves are ADDED together.)

Apples to apples: every year's path starts from the same kind of close and is laid on the same calendar dates.
    month path     days 0..31 of a month; day 0 = the previous month-end close (last trading day of the prior month)
    year path      1 Jan .. 31 Dec, starting from the previous year-end close
    forward path   from a chosen calendar date (e.g. 24 Aug) for N calendar days, starting from the close on or
                   before that date in each year
    Weekends and holidays carry the last close forward, so every year has a value on every calendar day.
    Average / median / 30th-70th percentile are taken across the COMPLETED years at each calendar day.

Monthly returns: r_m = close at month-end m / close at month-end m-1 - 1   (x 100; last trading day of each month)
    demeaned:  r_m - average of the previous 12 monthly returns (removes the trend using only past data)
    per calendar month: mean, median, hit rate = share of years the month was up
"""
from __future__ import annotations

import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TREND_MONTHS = 12


def _calendar(price: pd.Series) -> pd.Series:
    """Daily calendar series: every calendar day, carrying the last close over weekends and holidays."""
    p = price.dropna()
    return p.asfreq("D").ffill()


def _path(cal: pd.Series, base: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    seg = cal.loc[start:end]
    return (seg / base - 1) * 100


def _base_before(price: pd.Series, day: pd.Timestamp) -> float | None:
    """Last close strictly before ``day`` (the previous period's close)."""
    before = price.dropna().loc[: day - pd.Timedelta(days=1)]
    return float(before.iloc[-1]) if len(before) else None


def month_paths(price: pd.Series, month: int, years: list[int]) -> pd.DataFrame:
    """Index = day of month 0..31 (0 = previous month-end close), columns = years, values = % since then."""
    cal, out = _calendar(price), {}
    for y in years:
        start = pd.Timestamp(y, month, 1)
        base = _base_before(price, start)
        if base is None or start > cal.index[-1]:
            continue
        path = _path(cal, base, start, start + pd.offsets.MonthEnd(0))
        out[y] = pd.concat([pd.Series([0.0], index=[0]), pd.Series(path.values, index=path.index.day)])
    return pd.DataFrame(out)


def year_paths(price: pd.Series, years: list[int]) -> pd.DataFrame:
    """Index = days since 1 Jan (0..365), columns = years, values = % since the previous year-end close."""
    cal, out = _calendar(price), {}
    for y in years:
        start = pd.Timestamp(y, 1, 1)
        base = _base_before(price, start)
        if base is None or start > cal.index[-1]:
            continue
        path = _path(cal, base, start, pd.Timestamp(y, 12, 31))
        out[y] = pd.Series(path.values, index=(path.index - start).days)
    return pd.DataFrame(out)


def forward_paths(price: pd.Series, month: int, day: int, horizon: int, years: list[int]) -> pd.DataFrame:
    """Index = calendar days after the start date (0..horizon), columns = years, values = % since the start close."""
    cal, out = _calendar(price), {}
    for y in years:
        start = pd.Timestamp(y, month, min(day, pd.Timestamp(y, month, 1).days_in_month))  # 29 Feb -> 28 Feb
        if start > cal.index[-1]:
            continue
        base = cal.loc[:start]
        if base.empty:
            continue
        path = _path(cal, float(base.iloc[-1]), start, start + pd.Timedelta(days=horizon))
        out[y] = pd.Series(path.values, index=(path.index - start).days)
    return pd.DataFrame(out)


def summary(paths: pd.DataFrame, completed: list[int]) -> pd.DataFrame:
    """Average, median and 30th / 70th percentile across the completed years at each point of the path."""
    done = paths[[y for y in completed if y in paths.columns]]
    return pd.DataFrame({"mean": done.mean(axis=1), "median": done.median(axis=1),
                         "p30": done.quantile(0.30, axis=1), "p70": done.quantile(0.70, axis=1)})


def monthly_returns(price: pd.Series, running: bool = False) -> tuple[pd.Series, float | None]:
    """(complete-month % returns, month-to-date % of the unfinished month or None).
    ``running``: the last price is a live, unfinished day (so a month ending today is not complete yet)."""
    p = price.dropna()
    if len(p) < 40:
        return pd.Series(dtype=float), None
    month_end = p.resample("ME").last()
    r = (month_end.pct_change() * 100).dropna()
    last_day = p.index[-1]
    mtd = None
    if running or last_day.normalize() < (last_day + pd.offsets.BMonthEnd(0)).normalize():   # month not finished
        mtd = float(r.iloc[-1]) if len(r) else None
        r = r.iloc[:-1]
    return r, mtd


def demean_trailing(r: pd.Series, months: int = TREND_MONTHS) -> pd.Series:
    """r minus the average of the previous ``months`` returns (no look-ahead); the first year is dropped."""
    return (r - r.shift(1).rolling(months, min_periods=months).mean()).dropna()


def year_month_table(r: pd.Series) -> pd.DataFrame:
    """Rows = years, columns = Jan..Dec, values = monthly % return."""
    t = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.values}).pivot(index="y", columns="m", values="r")
    t = t.reindex(columns=range(1, 13))
    t.columns = MONTHS
    return t


def month_stats(r: pd.Series) -> pd.DataFrame:
    """One row per calendar month: mean %, median %, hit rate % (share of years up), years."""
    df = pd.DataFrame({"r": r.values, "m": r.index.month})
    g = df.groupby("m")["r"]
    out = pd.DataFrame({"mean_%": g.mean(), "median_%": g.median(),
                        "hit_rate_%": g.apply(lambda x: (x > 0).mean() * 100), "years": g.count()}).reindex(range(1, 13))
    out.index = MONTHS
    return out
