"""
Orchestrator: trade log + market data + analytics -> one Book object the UI renders.

Kept separate from app.py so the same function serves the live book and the
pre-trade "what if" book (trade log + proposed trade).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import pnl
import positions
import universe as U
from analytics.beta import dxy_equivalent
from logger import get_logger, log_step
from market_data import MarketData
from risk import CONF, LOOKBACK, RiskReport, compute_risk

log = get_logger(__name__)

MIN_EXPOSURE_USD = 1_000.0  # currencies with less net exposure are treated as flat


@dataclass
class Settings:
    conf: float = CONF          # fixed methodology (risk.py): historical simulation, 95%, 500 days
    lookback: int = LOOKBACK
    var_limit: float = 50_000.0
    ccy_limit: float = 5_000_000.0


@dataclass
class Book:
    live: pd.DataFrame                 # live trades with price_now, inception_pnl, daily_pnl
    positions: pd.DataFrame            # one row per pair, P&L and risk columns
    exposure: pd.Series                # USD value per currency (USD included)
    exposure_nonusd: pd.Series         # FX-risk-bearing exposures
    pair_exposure: pd.DataFrame        # pairs x currencies
    risk: RiskReport | None
    daily_pnl: float
    inception_pnl: float
    gross: float
    dxy_equiv: float
    excluded: list[str] = field(default_factory=list)   # trades dropped for missing prices
    checks: dict[str, bool] = field(default_factory=dict)


@log_step
def build_book(trades: pd.DataFrame, md: MarketData, betas: pd.Series, settings: Settings) -> Book:
    live = positions.live_trades(trades)
    excluded: list[str] = []
    if not live.empty:
        priced = live["base"].map(md.has) & live["quote"].map(md.has)
        if not priced.all():
            excluded = live.loc[~priced, "trade_id"].tolist()
            log.error("Trades without market data excluded from the book: %s", excluded)
            live = live[priced]

    q_now, q_prev = md.now, md.prev
    tp = pnl.trade_pnl(live, q_now, q_prev, md.prev_date)

    checks = {}
    if not live.empty:
        gap = float((pnl.mtm_usd(live, q_now) - pnl.mtm_usd_pair_formula(live, q_now)).abs().max())
        checks["Leg MTM = pair-formula MTM"] = gap < 1.0
        if gap >= 1.0:
            log.warning("MTM check failed: leg vs pair formula differ by $%.2f", gap)

    exposure = positions.exposure_by_currency(live, q_now)
    exposure_nonusd = exposure.drop(U.BASE_CCY, errors="ignore")
    exposure_nonusd = exposure_nonusd[exposure_nonusd.abs() >= MIN_EXPOSURE_USD]
    pair_exp = positions.pair_exposure_matrix(live, q_now)
    checks["All live trades priced"] = not excluded

    pos = positions.positions_table(live, q_now).join(pnl.pnl_by_pair(tp))
    risk = None
    if not pair_exp.empty:
        # completed days only: today's unfinished day is not a full scenario (see MarketData.last_complete_date)
        returns = md.simple_returns(list(pair_exp.columns)).loc[:md.last_complete_date]
        risk = compute_risk(pair_exp, returns, pos["size_usd"], settings.conf, settings.lookback)
    if risk is not None:
        pos = pos.join(pd.DataFrame({
            "position_var": risk.position_var, "component_var": risk.component_var,
            "marginal_var_per_mm": risk.marginal_var_per_mm, "es_contribution": risk.component,
            "marginal_es_per_mm": risk.marginal_per_mm, "pct_risk": risk.pct_risk,
        }))
        checks["ES contributions add up to ES"] = abs(risk.component.sum() - risk.es) < 1.0
        checks["Component VaRs add up to VaR"] = abs(risk.component_var.sum() - risk.var_hs) < 1.0
    else:
        for col in ("position_var", "component_var", "marginal_var_per_mm", "es_contribution", "marginal_es_per_mm",
                    "pct_risk"):
            pos[col] = np.nan

    book = Book(
        live=tp, positions=pos, exposure=exposure, exposure_nonusd=exposure_nonusd, pair_exposure=pair_exp,
        risk=risk, daily_pnl=float(tp["daily_pnl"].sum()) if not tp.empty else 0.0,
        inception_pnl=float(tp["inception_pnl"].sum()) if not tp.empty else 0.0,
        gross=float(exposure_nonusd.abs().sum()), dxy_equiv=dxy_equivalent(exposure_nonusd, betas),
        excluded=excluded, checks=checks,
    )
    log.info("Book: %d live trades, %d pairs, daily %.0f, inception %.0f, VaR %.0f",
             len(tp), len(pos), book.daily_pnl, book.inception_pnl, risk.var_hs if risk else 0.0)
    return book


def book_summary(book: Book, ccys: list[str] | None = None) -> dict:
    """Key numbers for the pre-trade comparison."""
    out = {
        "VaR (historical)": book.risk.var_hs if book.risk else 0.0,
        "Expected shortfall": book.risk.es if book.risk else 0.0,
        "Gross exposure": book.gross,
        "DXY-equivalent": book.dxy_equiv,
    }
    for c in ccys or []:
        if c != U.BASE_CCY:
            out[f"{c} exposure"] = float(book.exposure.get(c, 0.0))
    return out


def build_alerts(book: Book, md: MarketData, settings: Settings,
                 extra: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """(level, message) with level in {'error', 'warning', 'info'}; most severe first.
    ``extra`` carries flags raised elsewhere (cross check, data cleaning, analytics self-checks)."""
    alerts: list[tuple[str, str]] = list(extra or [])
    if book.excluded:
        alerts.append(("error", f"Trades excluded (no market data): {', '.join(book.excluded)}"))
    if book.risk:
        used = book.risk.var_hs / settings.var_limit if settings.var_limit else 0
        if used > 1:
            alerts.append(("error", f"VaR ${book.risk.var_hs:,.0f} is above the limit ${settings.var_limit:,.0f} "
                                    f"({used:.0%})"))
        elif used > 0.8:
            alerts.append(("warning", f"VaR at {used:.0%} of limit"))
        bt = book.risk.backtest
        if bt is not None and len(bt):
            expected = (1 - settings.conf) * len(bt)
            if bt["breach"].sum() > 1.6 * expected + 2:
                alerts.append(("warning", f"VaR backtest: {int(bt['breach'].sum())} breaches vs ~{expected:.0f} "
                                          "expected - model may understate risk"))
    for c, e in book.exposure_nonusd.items():
        if abs(e) > settings.ccy_limit:
            alerts.append(("warning", f"{c} exposure ${e:,.0f} exceeds the ${settings.ccy_limit:,.0f} limit"))
    stale = [c for c in book.exposure_nonusd.index if c in md.status.index and md.status.loc[c, "stale"]]
    if stale:
        alerts.append(("warning", f"Stale prices for held currencies: {', '.join(stale)}"))
    if md.source != "live":
        alerts.append(("warning", "Live data unavailable - using the bundled snapshot"))
    for note in md.notes:
        alerts.append(("info", note))
    for name, ok in book.checks.items():
        if not ok:
            alerts.append(("warning", f"Sanity check failed: {name} (see Diagnostics)"))
    if book.gross and abs(book.dxy_equiv) > 0.5 * book.gross:
        direction = "short" if book.dxy_equiv < 0 else "long"
        alerts.append(("info", f"Book is mostly a dollar bet: {direction} ${abs(book.dxy_equiv) / 1e6:,.1f}m "
                               f"DXY-equivalent ({abs(book.dxy_equiv) / book.gross:.0%} of gross)"))
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(alerts, key=lambda a: order[a[0]])
