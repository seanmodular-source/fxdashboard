"""
Tab renderers. Each function draws one part of the page from the Book / MarketData / MarketAnalytics
objects - no calculations of its own beyond presentation.

    Overview   - How am I doing?            KPIs, alerts, exposure, risk contributors
    Positions & risk - How much could I lose, what do I own?  VaR, ES, backtest; positions with risk columns; blotter
    Analytics  - Is my book well positioned?  rich/cheap, seasonality, DXY beta & peers, sessions + calendar
    Diagnostics - Can I trust the numbers?   data status, data checks (bad prints, cross check), logs
"""
from __future__ import annotations

import platform

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import attribution as A
import clock
import crosscheck as XC
import engine as E
import holidays
import market_data as M
import positions as P
import trades as T
import ui
import universe as U
from analytics import MarketAnalytics
from analytics import calendar_events as CE
from analytics import richcheap as RC
from analytics import seasonality as SE
from analytics import sessions as SS
from logger import find_in_log, get_logger, read_log_bytes, set_level, set_session_id, tail_log

log = get_logger(__name__)

HELP = {
    "daily": "Change in USD mark-to-market since the previous daily close (taken at 08:00 SGT). Trades done after "
             "that close count their full P&L.",
    "inception": "USD mark-to-market of all live trades since they were opened (closed-out P&L stays locked in).",
    "var": "Historical-simulation VaR: today's positions replayed over the last 500 completed days. At {conf:.0%} "
           "confidence the loss should not exceed this on {days} days out of {of}.",
    "es": "Expected shortfall: the average loss of the 25 worst of the last 500 replayed days (the worst 5%).",
    "gross": "Sum of absolute USD exposures across non-USD currencies: Σ |E_c|.",
    "dxy": "The DXY position that behaves like your book: DXY-equivalent = −Σ_c E_c × β_c (E_c = USD exposure, "
           "β_c = 3-month USD beta to DXY). Negative = like being short the dollar. DXY +1% → P&L ≈ DXY-equiv × 1%.",
    "div": "Sum of stand-alone position VaRs minus portfolio VaR: risk that offsets across positions.",
    "backtest": "Days in the last year when the scenario loss exceeded the VaR computed from the prior window.",
}


def var_help(conf: float) -> str:
    of = 20 if round(conf, 2) == 0.95 else 100
    return HELP["var"].format(conf=conf, days=round(conf * of), of=of)


# ============================================================================ header
def render_header(md: M.MarketData, book: E.Book, settings: E.Settings,
                  alerts: list[tuple[str, str]] | None = None) -> None:
    if md.live_marks:
        as_of = f"Prices as of {clock.fmt(md.as_of)}"
    else:
        as_of = f"Prices as of {md.last_date:%d %b %Y} close ({clock.close_time(md.last_date)})"
    badges = [ui.badge(as_of),
              ui.badge("Live · Yahoo Finance", "good") if md.source == "live" else ui.badge("Fallback snapshot", "warn")]
    if book.risk and settings.var_limit:
        used = book.risk.var_hs / settings.var_limit
        kind = "good" if used < 0.8 else ("warn" if used <= 1 else "bad")
        badges.append(ui.badge(f"VaR {used:.0%} of limit", kind))
    flags = [m for level, m in (alerts or []) if level in ("error", "warning")]
    if flags:
        badges.append(ui.badge(f"⚠ {len(flags)} flag{'s' if len(flags) > 1 else ''} · see Diagnostics", "warn"))
    st.markdown(f'<div class="fx-header"><span class="fx-title">FX Portfolio Monitor</span>{"".join(badges)}</div>',
                unsafe_allow_html=True)


# ============================================================================ overview
def _pnl_attribution(book: E.Book, md: M.MarketData) -> None:
    by_ccy = A.pnl_by_currency(book.live, md)
    if by_ccy.empty:
        st.info("No live trades to attribute.")
        return
    gap = max(abs(by_ccy["daily_pnl"].sum() - book.daily_pnl), abs(by_ccy["inception_pnl"].sum() - book.inception_pnl))
    if gap > 1.0:
        log.warning("P&L by currency does not add up to the headline (gap $%.2f)", gap)
        st.warning(ui.md_safe(f"P&L by currency differs from the headline by {ui.fmt_usd(gap)} - see Diagnostics."))
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        period = st.radio("Period", ["Inception", "Daily"], horizontal=True, key="attr_period")
        col = "inception_pnl" if period == "Inception" else "daily_pnl"
        ui.plot(ui.signed_bar(by_ccy[col], f"{period} P&L by currency", pos_color=ui.GAIN, neg_color=ui.LOSS,
                              pos_name="Gain", neg_name="Loss"), key="ov_attr_ccy")
    with c2:
        t = by_ccy.rename(columns={"daily_pnl": "Daily P&L", "inception_pnl": "Inception P&L"})
        t.loc["Total"] = t.sum()
        t.index.name = "Currency"
        ui.dataframe(ui.style_table(t, pnl=["Daily P&L", "Inception P&L"]), width="stretch")
        st.caption(ui.md_safe(f"Adds up to the headline: daily {ui.fmt_usd(book.daily_pnl, signed=True)}, "
                              f"inception {ui.fmt_usd(book.inception_pnl, signed=True)}."))

    ccys = tuple(sorted(c for c in P.holdings_units(book.live).index if c != U.BASE_CCY))
    hourly = M.load_hourly_q(ccys)
    sa = A.session_attribution(book.live, md, hourly) if not hourly.empty else None
    if sa is None:
        st.info("The session and weekday breakdown needs hourly prices (not available offline or for this book).")
        return
    sess = sa.by_session_ccy.sum(axis=1)
    sess.index = [f"{z}<br>{clock.SESSION_SGT[z]} SGT" for z in sess.index]
    c3, c4 = st.columns(2, gap="large")
    with c3:
        ui.plot(ui.signed_vbar(sess, "P&L by FX session (SGT)", pos_name="Gain", neg_name="Loss"), key="ov_attr_sess")
        t = sa.by_session_ccy.T
        t["Total"] = t.sum(axis=1)
        t.index.name = "Currency"
        ui.dataframe(ui.style_table(t, pnl=list(t.columns)), width="stretch")
    with c4:
        ui.plot(ui.signed_vbar(sa.by_weekday["total"], "P&L by weekday", pos_name="Gain", neg_name="Loss"),
                key="ov_attr_wd")
        w = sa.by_weekday.rename(columns={"total": "Total", "avg_per_day": "Avg per day", "up_days_%": "Up days %",
                                          "days": "Days"})
        w.index.name = "Weekday"
        ui.dataframe(ui.style_table(w, pnl=["Total", "Avg per day"], num={"Up days %": ".0f", "Days": ".0f"}),
                     width="stretch")
    total = sa.daily.sum()
    parts = [f"sessions {ui.fmt_usd(total, signed=True)}"]
    if abs(sa.before_window) >= 0.5:
        parts.append(f"P&L before the hourly history {ui.fmt_usd(sa.before_window, signed=True)}")
    if abs(sa.mark_gap) >= 0.5:
        parts.append(f"latest price vs last hourly bar {ui.fmt_usd(sa.mark_gap, signed=True)}")
    st.caption(ui.md_safe(
        f"Period: trading days {sa.start:%d %b %Y} to {sa.end:%d %b %Y} (each day 08:00 SGT to 08:00 SGT; the latest "
        f"day may still be running). Adds up to inception P&L: {' + '.join(parts)} = "
        f"{ui.fmt_usd(total + sa.before_window + sa.mark_gap, signed=True)} "
        f"(headline {ui.fmt_usd(book.inception_pnl, signed=True)})."))


MARKET_DEFAULT = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "USDSGD"]


def _chg_html(label: str, v: float) -> str:
    color = ui.UP_TEXT if v >= 0 else ui.DOWN_TEXT
    return f"{label}: <span style='color:{color}'>{v:+.2f}%</span>"


def _markets(md: M.MarketData) -> None:
    ui.section("Markets: FX pairs of interest")
    ui.method_note("markets")
    options = list(dict.fromkeys(U.TRADABLE_PAIRS + U.g10_crosses()))
    c1, c2 = st.columns([3, 1])
    picks = c1.multiselect("Pairs", options, default=[p for p in MARKET_DEFAULT if p in options], key="mk_pairs")
    hist = c2.radio("History", ["6M", "1Y", "2Y"], index=2, horizontal=True, key="mk_hist")
    prices = RC.pair_prices(md.q, picks) if picks else pd.DataFrame()
    picks = [p for p in picks if p in prices.columns and prices[p].notna().sum() > 22]
    if not picks:
        st.info("Choose at least one pair.")
        return
    n = {"6M": 126, "1Y": 252, "2Y": 504}[hist]
    chg = {p: {k: (prices[p].dropna().iloc[-1] / prices[p].dropna().iloc[-1 - d] - 1) * 100
               for k, d in (("CoD", 1), ("CoW", 5), ("CoM", 21))} for p in picks}
    for i in range(0, len(picks), 4):
        cols = st.columns(4, gap="small")
        for col, p in zip(cols, picks[i:i + 4]):
            s = prices[p].dropna().iloc[-n:]
            dp = U.price_decimals(p) + 1
            c = chg[p]
            title = (f"<b>{p}</b> ({s.iloc[-1]:.{dp}f})<br>"
                     f"{_chg_html('CoD', c['CoD'])} · {_chg_html('CoW', c['CoW'])} · {_chg_html('CoM', c['CoM'])}")
            fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=ui.BLUE, width=1.6),
                                       showlegend=False, hovertemplate=f"%{{x|%d %b %Y}}: %{{y:.{dp}f}}<extra></extra>"))
            fig.update_layout(title=dict(text=title, font=dict(size=11)), margin=dict(l=8, r=8, t=58, b=8))
            fig.update_xaxes(tickformat="%b'%y", nticks=5)
            with col:
                ui.plot(fig, height=230, key=f"mk_{p}")
    week = pd.Series({p: chg[p]["CoW"] for p in picks})
    month = pd.Series({p: chg[p]["CoM"] for p in picks})
    b1, b2 = st.columns(2, gap="large")
    for col, s, title, key in ((b1, week, "Weekly change (last 5 trading days)", "mk_week"),
                               (b2, month, "Monthly change (last 21 trading days)", "mk_month")):
        with col:
            ui.plot(ui.signed_bar(s, title, pos_color=ui.UP, neg_color=ui.DOWN, pos_name="Up", neg_name="Down",
                                  value_fmt=lambda v: f"{v:+.2f}%", tick_suffix="%"), key=key)
    st.caption("Pairs as quoted (+ = the first currency stronger), built from the USD prices. Daily close = 08:00 SGT.")


def render_overview(book: E.Book, md: M.MarketData, settings: E.Settings) -> None:
    ui.method_note("overview")
    r = book.risk
    used = (r.var_hs / settings.var_limit) if (r and settings.var_limit) else 0.0
    ui.kpi_cards([
        {"label": "Daily P&L", "value": ui.fmt_usd(book.daily_pnl, signed=True), "tone": ui.tone(book.daily_pnl),
         "sub": f"since {clock.close_time(md.prev_date)}", "help": HELP["daily"]},
        {"label": "Inception P&L", "value": ui.fmt_usd(book.inception_pnl, signed=True),
         "tone": ui.tone(book.inception_pnl), "sub": f"{len(book.live)} live trades", "help": HELP["inception"]},
        {"label": f"VaR {settings.conf:.0%} · 1 day", "value": ui.fmt_usd(r.var_hs if r else 0),
         "tone": "bad" if used > 1 else ("warn" if used > 0.8 else ""),
         "sub": f"{used:.0%} of {ui.fmt_usd(settings.var_limit)} limit", "help": var_help(settings.conf)},
        {"label": "Expected shortfall 95%", "value": ui.fmt_usd(r.es if r else 0),
         "sub": "avg of the 25 worst days", "help": HELP["es"]},
        {"label": "Gross exposure", "value": ui.fmt_usd(book.gross), "sub": f"{len(book.exposure_nonusd)} currencies",
         "help": HELP["gross"]},
    ])

    ui.section("What am I exposed to?")
    ui.method_note("exposure")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        exp = book.exposure.drop(U.BASE_CCY, errors="ignore")
        if book.exposure.get(U.BASE_CCY) is not None:
            exp[f"{U.BASE_CCY} (base)"] = book.exposure[U.BASE_CCY]
        if exp.empty:
            st.info("No open positions.")
        else:
            ui.plot(ui.signed_bar(exp, "Net exposure by currency (USD value held)", pos_name="Long (own)",
                                  neg_name="Short (owe)", neutral=[f"{U.BASE_CCY} (base)"]), key="ov_exp")
    with c2:
        if r is None:
            st.info("No FX risk to decompose.")
        else:
            ui.plot(ui.signed_bar(r.component, "Who uses the risk budget: contribution to ES (95%)",
                                  pos_name="Adds risk", neg_name="Hedge"), key="ov_comp")
            st.caption(ui.md_safe(f"Each bar = the position's average loss on the book's 25 worst replayed days. "
                                  f"They add up exactly to ES {ui.fmt_usd(r.es)}. Negative bars made money on those "
                                  "days: hedges."))

    ui.section("Where am I making or losing money?")
    ui.method_note("pnl")
    _pnl_attribution(book, md)

    _markets(md)


# ============================================================================ positions
def commit_book(df: pd.DataFrame, flash: str, backup: bool = False) -> None:
    """Dashboard -> file: make ``df`` the book in this session AND in data/portfolio.csv, so the two always match.
    If the file cannot be written (e.g. open in Excel) the change is kept in the session and flagged."""
    ss = st.session_state
    ss["trades"] = df
    try:
        ss["book_sig"] = T.save_book(df, backup=backup)
        ss.pop("book_save_error", None)
        ss["flash"] = flash
    except OSError as exc:
        ss["book_save_error"] = (f"Could not save {T.book_path().name} ({exc}). Close it if it is open in Excel; "
                                 "the change is kept in this session only.")
        log.warning("Book save failed: %s", exc)
        ss["flash"] = flash + " (NOT saved to file)"


def _bind_session() -> None:
    """Callbacks run before the script body, so tag their log lines with the session id explicitly."""
    set_session_id(st.session_state.get("sid", "-"))


def _cancel_cb() -> None:
    _bind_session()
    tid = st.session_state.get("cancel_id")
    if tid:
        commit_book(T.cancel_trade(st.session_state["trades"], tid), f"Trade {tid} cancelled")


def _reset_cb() -> None:
    _bind_session()
    commit_book(T.load_seed(), "Book reset to the sample portfolio (previous file kept as portfolio.backup.csv)",
                backup=True)
    log.info("Blotter reset to seed")


def _upload_cb() -> None:
    _bind_session()
    f = st.session_state.get("upload_file")
    if f is None:
        st.session_state["upload_errors"] = ["Choose a CSV file first."]
        return
    df, errors = T.read_upload(f.getvalue())
    if errors:
        st.session_state["upload_errors"] = errors
    else:
        commit_book(df, f"Loaded {len(df)} trades from {f.name} (previous file kept as portfolio.backup.csv)",
                    backup=True)


def render_positions(book: E.Book, md: M.MarketData) -> None:
    for e in st.session_state.get("book_errors", []):
        st.error(f"{T.book_path().name} was not loaded: {e}. Fix the file; the last valid book is shown.")
    if st.session_state.get("book_save_error"):
        st.error(st.session_state["book_save_error"])
    pos = book.positions
    if pos.empty:
        st.info("No live positions. Book a trade from the ticket in the sidebar, or reset to the sample portfolio.")
    else:
        ui.section("Positions", "Net position per pair. Avg entry is the break-even price of the net position.")
        disp = pd.DataFrame({
            "Side": pos["side"], "Net notional": pos["net_notional"], "Avg entry": pos["avg_entry"],
            "Price": pos["price_now"], "USD size": pos["size_usd"], "Daily P&L": pos["daily_pnl"],
            "Inception P&L": pos["inception_pnl"], "Position VaR": pos["position_var"], "Component VaR": pos["component_var"],
            "Marginal VaR / $1m": pos["marginal_var_per_mm"], "Contribution to ES": pos["es_contribution"],
            "Marginal ES / $1m": pos["marginal_es_per_mm"], "% of risk": pos["pct_risk"] * 100,
        })
        disp.index.name = "Pair"
        ui.dataframe(
            ui.style_table(disp, money=["USD size", "Position VaR", "Component VaR", "Marginal VaR / $1m",
                                        "Contribution to ES", "Marginal ES / $1m"],
                           pnl=["Daily P&L", "Inception P&L"], price=["Avg entry", "Price"], pct=["% of risk"],
                           num={"Net notional": ",.0f"}),
            width="stretch",
            column_config={
                "Position VaR": st.column_config.Column(help="Stand-alone 1-day historical VaR of this pair"),
                "Component VaR": st.column_config.Column(help="This pair's share of the book's VaR: its average loss on the 11 replayed days ranked around the VaR day (20th-30th worst), scaled so the column adds up to VaR; negative = hedge"),
                "Marginal VaR / $1m": st.column_config.Column(help="VaR change per $1m more of this position, for small changes"),
                "Contribution to ES": st.column_config.Column(help="This pair's average loss on the book's 25 worst days; the column adds up to ES; negative = hedge"),
                "Marginal ES / $1m": st.column_config.Column(help="ES change per $1m more of this position, for small changes (use the pre-trade check for real sizes)"),
                "% of risk": st.column_config.Column(help="Contribution to ES ÷ ES"),
                "Net notional": st.column_config.Column(help="In base currency; negative = short"),
                "USD size": st.column_config.Column(help="Net notional × today's USD price of the first currency; the size the marginal columns are per $1m of"),
            },
        )
        st.caption(ui.md_safe(f"Totals: daily {ui.fmt_usd(book.daily_pnl, signed=True)} · inception "
                              f"{ui.fmt_usd(book.inception_pnl, signed=True)} · {len(pos)} pairs"))
        r = book.risk
        if r is not None:
            st.caption(ui.md_safe(
                f"Risk columns: Component VaRs add up to VaR {ui.fmt_usd(r.var_hs)}; contributions add up to ES "
                f"{ui.fmt_usd(r.es)}; % of risk = contribution ÷ ES; position VaRs add up to "
                f"{ui.fmt_usd(r.position_var.sum())} (diversification {ui.fmt_usd(r.diversification)}). "
                "Negative = a hedge on the book's bad days."))
        ui.plot(ui.signed_bar(pos["inception_pnl"], "Inception P&L by pair", pos_color=ui.GAIN, neg_color=ui.LOSS,
                              pos_name="Gain", neg_name="Loss"), key="pos_pnl")

    ui.section("Blotter", "Every trade ever booked. Positions are rebuilt from this log on every change.")
    trades_df = st.session_state["trades"]
    pairs = sorted(trades_df["pair"].unique())
    pick = st.multiselect("Filter by pair", pairs, key="blotter_filter", placeholder="All pairs")
    bl = trades_df.merge(book.live[["trade_id", "price_now", "daily_pnl", "inception_pnl"]], on="trade_id",
                         how="left") if not book.live.empty else trades_df.assign(price_now=np.nan, daily_pnl=np.nan,
                                                                                   inception_pnl=np.nan)
    if pick:
        bl = bl[bl["pair"].isin(pick)]
    disp = pd.DataFrame({
        "Trade": bl["trade_id"], "Date": pd.to_datetime(bl["trade_date"]).dt.strftime("%Y-%m-%d"),
        "Pair": bl["pair"], "Side": bl["side"], "Notional": bl["notional"], "Price": bl["price"],
        "Market": bl["price_now"], "Daily P&L": bl["daily_pnl"], "Inception P&L": bl["inception_pnl"],
        "Status": bl["status"], "Booked at": bl["timestamp"],
    }).set_index("Trade")
    styler = ui.style_table(disp, pnl=["Daily P&L", "Inception P&L"], price=["Price", "Market"],
                            num={"Notional": ",.0f"})
    styler = styler.apply(lambda row: ["color: #898781"] * len(row) if row["Status"] == "CANCELLED" else
                          [""] * len(row), axis=1)
    ui.dataframe(styler, width="stretch")

    c1, c2, c3 = st.columns([1.3, 1.3, 1], gap="medium")
    with c1:
        live_ids = trades_df.loc[trades_df["status"] == "LIVE", "trade_id"].tolist()
        st.selectbox("Cancel a trade", live_ids, index=None, key="cancel_id", placeholder="Choose trade id")
        st.button("Cancel trade", on_click=_cancel_cb, disabled=not live_ids, width="stretch")
    with c2:
        st.file_uploader("Load a blotter (CSV)", type=["csv"], key="upload_file",
                         help="Columns: trade_date, pair, side, notional, price (+ optional trade_id, timestamp, "
                              "status)")
        st.button("Load file", on_click=_upload_cb, width="stretch")
    with c3:
        st.download_button("💾 Download blotter", data=T.to_csv_bytes(trades_df),
                           file_name=f"blotter_{clock.now_sgt():%Y%m%d_%H%M}_SGT.csv", mime="text/csv",
                           width="stretch", on_click="ignore")
        st.button("Reset to sample", on_click=_reset_cb, width="stretch")
    for e in st.session_state.pop("upload_errors", []) or []:
        st.error(e)
    st.caption(f"The book is saved to `data/{T.book_path().name}` on every change, and edits made to that file "
               "(e.g. in Excel) appear here within a few seconds. Download blotter = a copy; Reset to sample and "
               "Load file keep the previous file as `portfolio.backup.csv`.")


# ============================================================================ risk
def render_positions_risk(book: E.Book, md: M.MarketData, ma: MarketAnalytics | None, settings: E.Settings) -> None:
    """One tab: how much it could lose (VaR, ES, backtest), then what I hold (positions with risk columns, blotter)."""
    ui.method_note("positions")
    ui.section("Risk: how much could I lose, and why?")
    render_risk(book, md, ma, settings)
    st.divider()
    render_positions(book, md)


def render_risk(book: E.Book, md: M.MarketData, ma: MarketAnalytics | None, settings: E.Settings) -> None:
    ui.method_note("risk")
    r = book.risk
    if r is None:
        st.info("No open FX exposure - nothing to risk-manage yet.")
        return
    bt = r.backtest
    breaches = int(bt["breach"].sum()) if bt is not None else 0
    expected = (1 - r.conf) * len(bt) if bt is not None else 0
    ui.kpi_cards([
        {"label": f"Historical VaR {r.conf:.0%}", "value": ui.fmt_usd(r.var_hs), "sub": f"{r.n_obs}-day lookback",
         "help": var_help(r.conf)},
        {"label": f"Expected shortfall {r.conf:.0%}", "value": ui.fmt_usd(r.es),
         "sub": f"avg of the {len(r.tail_days)} worst days", "help": HELP["es"]},
        {"label": "Diversification benefit", "value": ui.fmt_usd(r.diversification),
         "sub": f"Σ position VaR {ui.fmt_usd(r.position_var.sum())}", "help": HELP["div"]},
        {"label": "VaR backtest", "value": f"{breaches} / {len(bt) if bt is not None else 0}",
         "tone": "warn" if bt is not None and breaches > 1.6 * expected + 2 else "",
         "sub": f"breaches · expected ≈ {expected:.0f}", "help": HELP["backtest"]},
    ])

    with st.expander("Model checks: VaR backtest and correlations"):
        m1, m2 = st.columns(2, gap="large")
        with m1:
            if bt is None:
                st.info("Not enough history for a backtest at this lookback.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt.index, y=bt["pnl"], mode="markers", name="Scenario P&L",
                                         marker=dict(color=ui.MUTED, size=5),
                                         hovertemplate="%{x|%d %b %Y}: %{y:$,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=bt.index, y=-bt["var"], mode="lines", name="−VaR (prior window)",
                                         line=dict(color=ui.INK_2, width=2)))
                br = bt[bt["breach"]]
                fig.add_trace(go.Scatter(x=br.index, y=br["pnl"], mode="markers", name=f"Breach ({len(br)})",
                                         marker=dict(color=ui.LOSS, size=9, symbol="x"),
                                         hovertemplate="%{x|%d %b %Y}: %{y:$,.0f}<extra></extra>"))
                fig.update_layout(title=f"Backtest: {breaches} breaches vs ≈{expected:.0f} expected")
                fig.update_yaxes(tickformat="$~s")
                ui.plot(fig, height=340, key="risk_bt")
        with m2:
            ccys = list(book.exposure_nonusd.index)
            if len(ccys) < 2:
                st.info("Correlations need at least two currencies.")
            else:
                corr = md.simple_returns(ccys).loc[:md.last_complete_date].iloc[-r.lookback:].corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index, colorscale=ui.DIVERGING, zmin=-1, zmax=1,
                    text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                    hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="ρ")))
                fig.update_layout(title=f"Correlation of daily moves vs USD ({r.lookback}d)")
                fig.update_yaxes(autorange="reversed")
                ui.plot(fig, height=340, key="risk_corr")


# ============================================================================ analytics
LEVEL_COLS = {f"level_z_{n}": k for k, n in RC.LEVEL_WINDOWS.items()}   # level_z_21 -> "1M" ...
MOVE_COLS = {f"stretch_z_{h}": k for k, h in RC.MOVE_HORIZONS.items()}  # stretch_z_5 -> "1W" ...
RET_COLS = {f"ret_{h}d": k for k, h in RC.MOVE_HORIZONS.items()}        # ret_5d -> "1W" ...
EVENT_COLORS = {"ME": "#4a3aa7", "MM": "#eda100", "HD": "#e87ba4", "FS": ui.INK_2, "FE": ui.INK_2}


def _z_heatmap(z: pd.DataFrame, title: str, key: str, showscale: bool = True) -> None:
    """Green (+) / red (−) z-score heatmap (colour clipped at ±3, labels unclipped)."""
    fig = go.Figure(go.Heatmap(
        z=z.clip(-3, 3).values, x=list(z.columns), y=list(z.index), colorscale=ui.Z_DIVERGING, zmin=-3, zmax=3,
        text=z.map(ui.fmt_z).values, texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        showscale=showscale, xgap=2, ygap=2,
        colorbar=dict(title="z", tickvals=[-3, -2, -1, 0, 1, 2, 3],
                      ticktext=["−3 cheap", "−2", "−1", "0", "+1", "+2", "+3 rich"])))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    ui.plot(fig, height=max(260, 30 * len(z) + 120), key=key)


def _rich_cheap_panels(rc: pd.DataFrame, key: str, sign_note: str, extra: pd.DataFrame | None = None) -> None:
    """Level z and move z heatmaps side by side, then the % change chart and table (currencies or crosses)."""
    c1, c2 = st.columns([4, 6], gap="medium")
    with c1:
        _z_heatmap(rc[list(LEVEL_COLS)].rename(columns=LEVEL_COLS), "Level z: price vs its own average", f"{key}_lvl",
                   showscale=False)
    with c2:
        _z_heatmap(rc[list(MOVE_COLS)].rename(columns=MOVE_COLS), "Move z: how unusual was the move", f"{key}_mv")
    ui.section("% change", sign_note)
    hz = st.radio("Horizon", list(RC.MOVE_HORIZONS), index=1, horizontal=True, key=f"{key}_hz")
    s = rc[f"ret_{RC.MOVE_HORIZONS[hz]}d"]
    ui.plot(ui.signed_bar(s, f"% change over {hz}", pos_color=ui.UP, neg_color=ui.DOWN, pos_name="Up",
                          neg_name="Down", value_fmt=lambda v: f"{v:+.2f}%", tick_suffix="%"), key=f"{key}_bar")
    t = rc[list(RET_COLS)].rename(columns={c: f"{k} %" for c, k in RET_COLS.items()})
    if extra is not None:
        t = extra.join(t)
    ui.dataframe(ui.style_table(t, signed={f"{k} %": "+.2f" for k in RET_COLS.values()}), width="stretch")


def _rich_cheap(book: E.Book, ma: MarketAnalytics) -> None:
    ui.method_note("rich_cheap")
    held_only = st.toggle("Held currencies only", value=bool(len(book.exposure_nonusd)), key="rc_held")
    ccys = [c for c in (book.exposure_nonusd.index if held_only else ma.rich_cheap.index) if c in ma.rich_cheap.index]
    if not ccys:
        st.info("No currencies to show.")
        return
    rc = ma.rich_cheap.loc[ccys]
    rc.index.name = "Ccy"
    _rich_cheap_panels(rc, "rc", "Currency vs USD: + = currency stronger (green), − = weaker (red).")
    st.caption("Level z compares today's price with the prior 1M / 3M / 6M / 1Y; move z divides each move by the "
               "currency's normal volatility before it. Prices trend, so 'rich can get richer': context, not a signal.")


SEASON_PAST = "#c9c8c1"      # earlier years: faint
SEASON_NOW = "#e8202a"       # this year: bright red
SEASON_RECENT = [ui.BLUE, ui.ORANGE]   # the two latest completed years (dotted): older, then newer


def _season_fig(paths: pd.DataFrame, stats: pd.DataFrame, x_of, cur: int, past: list[int], title: str,
                today_x=None, band: bool = False, x_title: str = "") -> go.Figure:
    """Seasonal paths: faint earlier years, dotted blue/orange last two years, black average, dashed median, red this year."""
    fig = go.Figure()
    recent = past[-2:]
    first = True
    for y in past:
        if y not in paths.columns or y in recent:
            continue
        s = paths[y].dropna()
        fig.add_trace(go.Scatter(x=x_of(s.index), y=s.values, mode="lines", line=dict(color=SEASON_PAST, width=1),
                                 name="Earlier years", legendgroup="past", showlegend=first,
                                 hovertemplate=f"{y}: %{{y:+.2f}}%<extra></extra>"))
        first = False
    if band:
        fig.add_trace(go.Scatter(x=x_of(stats.index), y=stats["p30"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x_of(stats.index), y=stats["p70"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(82,81,78,0.13)", name="30th–70th pct", hoverinfo="skip"))
    for y, color in zip(recent, SEASON_RECENT[-len(recent):] if recent else []):
        if y in paths.columns:
            s = paths[y].dropna()
            fig.add_trace(go.Scatter(x=x_of(s.index), y=s.values, mode="lines", name=str(y),
                                     line=dict(color=color, width=2, dash="dot"),
                                     hovertemplate=f"{y}: %{{y:+.2f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_of(stats.index), y=stats["median"], mode="lines", name="Median",
                             line=dict(color=ui.INK, width=2, dash="dash"),
                             hovertemplate="Median: %{y:+.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=x_of(stats.index), y=stats["mean"], mode="lines", name="Average",
                             line=dict(color=ui.INK, width=3.5), hovertemplate="Average: %{y:+.2f}%<extra></extra>"))
    if cur in paths.columns:
        s = paths[cur].dropna()
        fig.add_trace(go.Scatter(x=x_of(s.index), y=s.values, mode="lines", name=f"{cur} (this year)",
                                 line=dict(color=SEASON_NOW, width=3, dash="dot"),
                                 hovertemplate=f"{cur}: %{{y:+.2f}}%<extra></extra>"))
    if today_x is not None:
        fig.add_vline(x=today_x, line=dict(color=ui.UP, width=1.5, dash="dot"))
        fig.add_annotation(x=today_x, y=1, yref="paper", yanchor="top", xanchor="left", xshift=4, text="today",
                           showarrow=False, font=dict(size=10, color=ui.UP_TEXT))
    fig.add_hline(y=0, line=dict(color=ui.AXIS, width=1))
    fig.update_layout(title=title, xaxis_title=x_title, margin=dict(t=120))
    fig.update_yaxes(ticksuffix="%", title_text="% change")
    return fig


def _season_heatmap(tbl: pd.DataFrame, stats: pd.DataFrame, title: str) -> go.Figure:
    """Year x month heatmap (1 dp), plus mean / median / hit rate rows. Each row is coloured on its own scale."""
    rows, z, text = [], [], []
    for y, vals in tbl.sort_index(ascending=False).iterrows():
        v = vals.values.astype(float)
        scale = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
        rows.append(str(y))
        z.append(v / (scale or 1.0))
        text.append(["" if np.isnan(x) else f"{x:+.1f}" for x in v])
    rows.append(" ")
    z.append([np.nan] * 12)
    text.append([""] * 12)
    for label, col in (("Mean", "mean_%"), ("Median", "median_%")):
        v = stats[col].values.astype(float)
        rows.append(label)
        z.append(v / (np.nanmax(np.abs(v)) or 1.0))
        text.append(["" if np.isnan(x) else f"{x:+.1f}" for x in v])
    h = stats["hit_rate_%"].values.astype(float)
    rows.append("Hit rate")
    z.append((h - 50) / 50)
    text.append(["" if np.isnan(x) else f"{x:.0f}%" for x in h])
    fig = go.Figure(go.Heatmap(
        z=z, x=SE.MONTHS, y=rows, text=text, texttemplate="%{text}", colorscale=ui.Z_DIVERGING, zmin=-1, zmax=1,
        showscale=False, xgap=2, ygap=2, hovertemplate="%{y} %{x}: %{text}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top")
    return fig


def _seasonality(book: E.Book, md: M.MarketData) -> None:
    ui.method_note("seasonality")
    held = [p for p in book.positions.index if book.positions.loc[p, "side"] != "FLAT"]
    options = list(dict.fromkeys(held + U.TRADABLE_PAIRS + U.g10_crosses()))
    c1, c2, c3 = st.columns([1.2, 0.8, 1])
    pair = c1.selectbox("Pair", options, key="seas_pair")
    n_years = int(c2.number_input("History (years)", min_value=3, max_value=11, value=10, step=1, key="seas_years",
                                  help="Completed years to compare (up to 11)"))
    mode = c3.radio("Monthly figures", ["Raw", "Demeaned"], horizontal=True, key="seas_mode",
                    help="Demeaned = each month minus the average of the previous 12 months (no look-ahead)")
    base, quote = U.parse_pair(pair)
    ccys = tuple(sorted(c for c in (base, quote) if c != U.BASE_CCY))
    q, source = M.load_long_history(ccys, "12y")
    if not q.empty:   # the app's own recent closes (with the hourly top-up) win where both exist
        q = md.q[[c for c in ccys if c in md.q.columns]].combine_first(q)
    price = RC.pair_prices(q, [pair]).get(pair) if not q.empty else None
    if price is None or price.dropna().empty:
        st.info(f"No price history for {pair}.")
        return
    price = price.dropna()
    today = clock.today_sgt()
    cur = today.year
    years = [y for y in range(cur - n_years, cur + 1) if pd.Timestamp(y - 1, 12, 31) >= price.index[0]]
    past = [y for y in years if y < cur]
    if len(past) < 3:
        st.info("Not enough history for a seasonality view.")
        return
    # 1) path from a date (the picture-1 chart)
    ui.section("Path from a date", "What usually happened next, from the same calendar date in each past year.")
    f1, f2 = st.columns(2)
    start = pd.Timestamp(f1.date_input("Start date", value=today.date(), key="seas_start"))
    horizon = int(f2.number_input("Calendar days ahead", min_value=5, max_value=365, value=60, step=5, key="seas_h"))
    fp = SE.forward_paths(price, start.month, start.day, horizon, years)
    start_cur = pd.Timestamp(cur, start.month, min(start.day, pd.Timestamp(cur, start.month, 1).days_in_month))
    in_window = start_cur <= today <= start_cur + pd.Timedelta(days=horizon)
    ui.plot(_season_fig(fp, SE.summary(fp, past), lambda idx: [start_cur + pd.Timedelta(days=int(i)) for i in idx],
                        cur, past, f"{pair} from {start_cur:%d %b}, next {horizon} days, each year "
                                   f"(+ = {base} stronger; {past[0]}–{past[-1]})",
                        today if in_window else None), height=430, key="seas_fwd")

    # 2) month profile and 3) year profile
    month = st.selectbox("Month for the month profile", range(1, 13), index=today.month - 1,
                         format_func=lambda m: SE.MONTHS[m - 1], key="seas_month")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        mp = SE.month_paths(price, month, years)
        ui.plot(_season_fig(mp, SE.summary(mp, past), lambda idx: list(idx), cur, past,
                            f"{SE.MONTHS[month - 1]}: move from the previous month-end",
                            today.day if month == today.month else None,
                            x_title="Day of month (0 = previous month-end close)"), height=430, key="seas_mprof")
    with m2:
        jan1 = pd.Timestamp(cur, 1, 1)
        yp = SE.year_paths(price, years)
        yp = yp[yp.index <= (pd.Timestamp(cur, 12, 31) - jan1).days]
        ui.plot(_season_fig(yp, SE.summary(yp, past), lambda idx: [jan1 + pd.Timedelta(days=int(i)) for i in idx],
                            cur, past, "Year to date: move since the previous year-end", today, band=True),
                height=430, key="seas_yprof")

    # 4) monthly bars vs this year and 5) heatmap
    r_all, mtd = SE.monthly_returns(price, running=md.last_complete_date < md.last_date)
    r_use = SE.demean_trailing(r_all) if mode == "Demeaned" else r_all
    if mtd is not None and mode == "Demeaned":
        mtd = mtd - float(r_all.iloc[-12:].mean())
    stats = SE.month_stats(r_use[(r_use.index.year >= cur - n_years) & (r_use.index.year < cur)])
    tbl = SE.year_month_table(r_use[r_use.index.year >= cur - n_years])
    mtd_month = SE.MONTHS[price.index[-1].month - 1]
    if mtd is not None:
        if cur not in tbl.index:
            tbl.loc[cur] = np.nan
        tbl.loc[cur, mtd_month] = mtd
    cur_row = tbl.loc[cur].dropna() if cur in tbl.index else pd.Series(dtype=float)
    labels = [f"{m} ({h:.0f}%)" for m, h in zip(stats.index, stats["hit_rate_%"].fillna(0))]
    fig = go.Figure([
        go.Bar(x=labels, y=stats["mean_%"], name="Average", marker=dict(color=ui.INK_2, cornerradius=3),
               hovertemplate="%{x} average: %{y:+.2f}%<extra></extra>"),
        go.Bar(x=labels, y=stats["median_%"], name="Median", marker=dict(color="#b9b8b1", cornerradius=3),
               hovertemplate="%{x} median: %{y:+.2f}%<extra></extra>"),
        go.Scatter(x=[labels[SE.MONTHS.index(m)] for m in cur_row.index], y=cur_row.values, name=f"{cur}",
                   mode="lines+markers", line=dict(color=SEASON_NOW, width=2.5), marker=dict(size=8),
                   hovertemplate=f"{cur} %{{x}}: %{{y:+.2f}}%<extra></extra>"),
    ])
    fig.update_layout(barmode="group", title=f"Monthly seasonality ({mode.lower()}): {past[0]}–{past[-1]} "
                                             f"average and median vs {cur}",
                      xaxis_title="Month (hit rate = share of years the month was up)")
    fig.update_yaxes(ticksuffix="%", title_text="% move in the month")
    ui.plot(fig, height=400, key="seas_bars")
    ui.plot(_season_heatmap(tbl, stats, f"Every month of every year, % ({mode.lower()}; {cur} {mtd_month} = month to "
                                         "date). Each row coloured on its own scale"),
            height=34 * (len(tbl) + 4) + 120, key="seas_heat")
    st.caption(f"Source: {source}. Moves are % changes of {pair} as quoted (+ = {base} stronger). Month-end = close "
               "of the last trading day of the month.")


def _dxy_beta(book: E.Book, ma: MarketAnalytics) -> None:
    ui.method_note("dxy_beta")
    sc = ma.scatter.assign(usd_pct_21=lambda d: (np.exp(d["usd_ret_21"] / 100) - 1) * 100)  # log -> plain %
    dxy_pct = (np.exp(ma.dxy_ret_21 / 100) - 1) * 100
    held = set(book.exposure_nonusd.index)
    fig = go.Figure()
    for is_held, color, name, size in ((False, ui.MUTED, "Other currencies", 8), (True, ui.BLUE, "Held", 11)):
        sub = sc[sc.index.isin(held) == is_held]
        fig.add_trace(go.Scatter(
            x=sub["beta_63"], y=sub["usd_pct_21"], mode="markers+text", text=sub.index, textposition="top center",
            textfont=dict(size=10 if not is_held else 11, color=ui.INK_2), name=name, customdata=sub["group"],
            marker=dict(size=size, color=color, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="%{text} (%{customdata})<br>USD β %{x:.2f}<br>USD vs ccy, 1M: %{y:+.2f}%<extra></extra>"))
    lo = min(float(sc["beta_63"].min()), 0.0) - 0.1
    hi = max(float(sc["beta_63"].max()), 1.0) + 0.1
    xs = np.linspace(lo, hi, 50)
    fig.add_trace(go.Scatter(x=xs, y=(np.exp(xs * ma.dxy_ret_21 / 100) - 1) * 100, mode="lines",
                             line=dict(color=ui.INK_2, width=2), name="β × DXY 1M move", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[1.0], y=[dxy_pct], mode="markers+text", text=["DXY"], textposition="bottom center",
                             marker=dict(symbol="star", size=15, color=ui.INK), name="DXY (synthetic)",
                             hovertemplate=f"DXY 1M: {dxy_pct:+.2f}%<extra></extra>"))
    fig.update_layout(title="Who moved more or less than the dollar implies? (1M)",
                      xaxis_title="USD beta to DXY (63 days)", yaxis_title="USD vs currency, 1M % change (+ = USD stronger)")
    fig.update_xaxes(showgrid=True)
    ui.plot(fig, height=460, key="beta_scatter")
    st.caption("Above the line: the currency did worse than its beta implies (own weakness). Below: it held up "
               "better. DXY is rebuilt from Yahoo FX closes so timestamps match.")

    ui.section("Layered move breakdown", "Total move = dollar (β × DXY) + bloc (peer median) + own, in log % so the "
                                         "three parts add up exactly; the plain % total is in the title.")
    held_sorted = book.exposure_nonusd.abs().sort_values(ascending=False).index.tolist()
    options = held_sorted + [c for c in ma.rich_cheap.index if c not in held_sorted]
    c1, c2 = st.columns([1, 1])
    ccy = c1.selectbox("Currency", options, key="beta_ccy")
    h = c2.radio("Horizon", [5, 21], format_func=lambda x: "1 week" if x == 5 else "1 month", horizontal=True,
                 key="beta_h", index=1)
    d = ma.decomposition[h].loc[ccy]
    fig = go.Figure(go.Waterfall(
        x=["Dollar (β × DXY)", "Bloc", "Own", "Total"], y=[d["dollar"], d["bloc"], d["own"], d["total"]],
        measure=["relative", "relative", "relative", "total"],
        text=[f"{v:+.2f}%" for v in (d["dollar"], d["bloc"], d["own"], d["total"])], textposition="outside",
        increasing=dict(marker=dict(color=ui.UP)), decreasing=dict(marker=dict(color=ui.DOWN)),
        totals=dict(marker=dict(color=ui.INK_2)), connector=dict(line=dict(color=ui.AXIS, width=1)),
        hovertemplate="%{x}: %{text}<extra></extra>"))
    plain = (np.exp(d["total"] / 100) - 1) * 100
    fig.update_layout(title=f"{ccy} vs USD over {'1 week' if h == 5 else '1 month'} (+ = {ccy} stronger): "
                            f"{d['total']:+.2f}% log = {plain:+.2f}% plain", yaxis_title="log % move", showlegend=False)
    ui.plot(fig, height=330, key="beta_wf")

    t = ma.betas.rename(columns={"beta_63": "USD β 3M", "r2_63": "R² 3M", "beta_252": "USD β 1Y", "r2_252": "R² 1Y"})
    t.index.name = "Ccy"
    with st.expander("Beta table (all currencies)"):
        ui.dataframe(ui.style_table(t, num={c: ".2f" for c in t.columns}), width="stretch")


def _peers(book: E.Book, ma: MarketAnalytics) -> None:
    ui.method_note("peers")
    groups = list(U.PEER_GROUPS)
    default = 0
    if len(book.exposure_nonusd):
        top = book.exposure_nonusd.abs().idxmax()
        default = groups.index(U.PRIMARY_GROUP.get(top, groups[0]))
    c1, c2 = st.columns([1, 1])
    group = c1.selectbox("Peer group", groups, index=default, key="peer_group")
    h = c2.radio("Horizon", [5, 21], format_func=lambda x: "1 week" if x == 5 else "1 month", horizontal=True,
                 key="peer_h", index=1)
    members = [c for c in U.PEER_GROUPS[group] if c in ma.decomposition[h].index]
    # each currency's bloc / own is measured against its main group; say so when that is not the group shown
    label = {c: c if U.PRIMARY_GROUP.get(c) == group else f"{c} (vs {U.PRIMARY_GROUP.get(c)})" for c in members}
    d = ma.decomposition[h].loc[members].rename(index=label)
    ui.plot(ui.signed_bar(d["own"], f"Own move vs {group} peers ({'1W' if h == 5 else '1M'}, log %), after removing "
                                     "the dollar",
                          pos_color=ui.UP, neg_color=ui.DOWN, pos_name="Beat its bloc", neg_name="Lagged its bloc",
                          value_fmt=lambda v: f"{v:+.2f}%", tick_suffix="%"),
            key="peer_bar")
    t = d.rename(columns={"total": "Total log %", "dollar": "Dollar log %", "bloc": "Bloc log %",
                          "own": "Own log %"}).join(
        ma.peer_z.rename(index=label).rename(columns={"peer_z_63": "Peer z 3M", "peer_z_252": "Peer z 1Y"}))
    t.insert(0, "Exposure", book.exposure_nonusd.rename(index=label).reindex(t.index).fillna(0.0))
    t.index.name = "Ccy"
    ui.dataframe(ui.style_table(t, money=["Exposure"], z=["Peer z 3M", "Peer z 1Y"],
                                signed={c: "+.2f" for c in ["Total log %", "Dollar log %", "Bloc log %", "Own log %"]}),
                 width="stretch")
    st.caption("Bloc = median of the other members' dollar-adjusted moves (leave-one-out). Own = what is left: "
               "the currency-specific story. Peer z = how stretched that own move is vs its history.")


def _session_chart(ccy: str, zr: pd.DataFrame, daily: pd.Series, zscores: pd.DataFrame, events: pd.DataFrame,
                   title: str, days: int = 60) -> go.Figure:
    show = zr.iloc[-days:]
    level = np.exp(daily.reindex(show.index).fillna(0).cumsum() / 100) * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.06)
    fig.update_yaxes(title_text=f"{ccy} index", row=1, col=1)
    fig.update_yaxes(title_text="Session move, % log", row=2, col=1)
    fig.add_trace(go.Scatter(x=level.index, y=level.values, mode="lines", line=dict(color=ui.INK_2, width=2),
                             name=f"{ccy} index", showlegend=False,
                             hovertemplate="%{x|%d %b}: %{y:.2f}<extra></extra>"), row=1, col=1)
    pos_bottom = np.zeros(len(show))
    neg_bottom = np.zeros(len(show))
    star_x, star_y, star_t = [], [], []
    for zone in SS.ZONES:
        vals = show[zone].fillna(0).values
        fig.add_trace(go.Bar(x=show.index, y=vals, name=f"{zone} {clock.SESSION_SGT[zone]}",
                             marker=dict(color=ui.SESSION_COLORS[zone]),
                             hovertemplate=f"{zone} %{{x|%d %b}}: %{{y:+.2f}}%<extra></extra>"), row=2, col=1)
        mids = np.where(vals > 0, pos_bottom + vals / 2, neg_bottom + vals / 2)
        zz = zscores[zone].reindex(show.index)
        for i, (dt, zv) in enumerate(zz.items()):
            if pd.notna(zv) and abs(zv) >= SS.Z_FLAG and vals[i] != 0:
                star_x.append(dt)
                star_y.append(mids[i])
                star_t.append(f"{zone}: {zv:+.1f}σ")
        pos_bottom += np.where(vals > 0, vals, 0)
        neg_bottom += np.where(vals < 0, vals, 0)
    fig.add_trace(go.Scatter(x=star_x, y=star_y, mode="markers", name="|z| ≥ 1", text=star_t,
                             marker=dict(symbol="star", size=11, color=ui.INK, line=dict(color="#fcfcfb", width=1)),
                             hovertemplate="%{x|%d %b} %{text}<extra></extra>"), row=2, col=1)
    # Calendar events: dotted lines on both panels, codes labelled above the top panel.
    if not events.empty:
        shown = events[events["plotted_date"].isin(show.index)]
        for i, (d, grp) in enumerate(shown.groupby("plotted_date")):
            codes = list(dict.fromkeys(grp["event_code"]))
            fig.add_vline(x=d, line=dict(color=EVENT_COLORS.get(codes[0], ui.MUTED), width=1.4, dash="dot"),
                          row="all", col=1)
            fig.add_annotation(x=d, y=1.0 + 0.035 * (i % 2), xref="x", yref="paper", yanchor="bottom",
                               showarrow=False, text="/".join(codes), font=dict(size=9, color=ui.INK_2))
        for code in dict.fromkeys(shown["event_code"]):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"{code} = {CE.CODES[code]}",
                                     line=dict(color=EVENT_COLORS[code], dash="dot", width=1.6)))
    fig.update_layout(barmode="relative", bargap=0.2, title=title, margin=dict(t=120, b=10),
                      legend=dict(orientation="h", y=-0.08, yanchor="top", x=0, xanchor="left"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def _sessions(book: E.Book, md: M.MarketData) -> None:
    ui.method_note("sessions")
    held = book.exposure_nonusd.abs().sort_values(ascending=False).index.tolist()
    options = held + sorted(c for c in U.SESSION_RELIABLE if c not in held)
    default = next((i for i, c in enumerate(options) if c in U.SESSION_RELIABLE), 0)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    ccy = c1.selectbox("Currency", options, index=default, key="sess_ccy")
    lookback = int(c2.number_input("z-score lookback (trading days)", min_value=5, max_value=250, value=SS.LOOKBACK,
                                   step=5, key="sess_lookback", help="How many previous days each session is compared with"))
    days = int(c3.number_input("Days shown", min_value=5, max_value=250, value=60, step=5, key="sess_days",
                               help="Limited by the hourly history Yahoo provides"))
    if ccy not in U.SESSION_RELIABLE:
        st.warning(f"{ccy} trades mainly onshore; Yahoo's out-of-hours quotes are thin, so treat sessions as "
                   "indicative only.")
    hourly = M.load_hourly_q((ccy,))
    if hourly.empty or ccy not in hourly.columns:
        st.info("Hourly data is unavailable right now (offline mode or Yahoo limit). Check Diagnostics.")
        return
    zr, daily = SS.session_returns(hourly[ccy])
    gap_bps = (zr.sum(axis=1, min_count=1) - daily).abs().iloc[1:].max() * 100
    if pd.notna(gap_bps) and gap_bps > 0.5:  # session_returns logs the same check
        st.warning(f"{ccy} session moves do not add up to the daily move (max gap {gap_bps:.2f} bps). "
                   "Treat this view with care; the log has details.")
    days = min(days, len(zr))
    zs = SS.session_zscores(zr, lookback)
    pair = U.ticker_of(ccy).replace("=X", "")
    shown_days = zr.index[-days:]
    events = CE.map_to_trading_days(CE.events_between(pair, shown_days.min(), shown_days.max()), shown_days)
    upcoming = CE.upcoming(pair, clock.today_sgt(), 7)
    summ_all = SS.session_summary(zr.iloc[-days:])
    head = " | ".join(f"{z} {summ_all.loc[z, 'cumulative_%']:+.1f}%, {summ_all.loc[z, 'hit_rate_%']:.0f}% up"
                      for z in SS.ZONES)
    title = (f"{ccy} vs USD by session, last {days} days (+ = {ccy} stronger; ★ = |z| ≥ 1)<br>"
             f"<sub>{head} · {CE.title_line(upcoming)}</sub>")
    ui.plot(_session_chart(ccy, zr, daily, zs, events, title, days), height=640, key="sess_chart")

    summ = summ_all.drop(columns="avg_%").rename(columns={"cumulative_%": "Cumulative %",
                                                         "hit_rate_%": "Up sessions %", "days": "Days"})
    summ.insert(0, "Window (SGT)", [clock.SESSION_SGT[z] for z in summ.index])
    summ[f"Today z ({lookback}d)"] = zs.iloc[-1].reindex(SS.ZONES)
    summ.index.name = "Session"
    ui.dataframe(ui.style_table(summ, z=[f"Today z ({lookback}d)"],
                                signed={"Cumulative %": "+.2f"},
                                num={"Up sessions %": ".0f", "Days": ".0f"}), width="stretch")
    st.caption(f"Last {days} trading days, currency terms (+ = {ccy} stronger vs USD). Each trading day runs "
               "08:00 SGT to 08:00 SGT; the Monday ~05:00 SGT reopen counts as Monday's Asia session.")

    ui.section("Calendar", f"Holidays in {', '.join(CE.COUNTRY.get(c, c) for c in U.parse_pair(pair))}, month-end, "
                           "mid-month and fiscal start / end. Weekend events are shown on the next trading day. Holiday "
                           f"source: Python package `holidays` {holidays.__version__} (PyPI); the other dates are "
                           "computed from the calendar.")
    e1, e2 = st.columns([1.4, 1], gap="large")
    with e1:
        st.markdown("**Events in the chart window**")
        if events.empty:
            st.caption("None.")
        else:
            ev = events.assign(event_date=events["event_date"].dt.strftime("%Y-%m-%d"),
                               plotted_date=events["plotted_date"].dt.strftime("%Y-%m-%d"))
            ui.dataframe(ev.rename(columns={"event_date": "Event date", "plotted_date": "Shown on",
                                            "event_code": "Code", "event_name": "Event"}).set_index("Event date"),
                         width="stretch", height=260)
    with e2:
        st.markdown(f"**Next 7 days** (from {clock.today_sgt():%d %b %Y} SGT)")
        if upcoming.empty:
            st.caption("No events.")
        else:
            up = upcoming.assign(event_date=upcoming["event_date"].dt.strftime("%a %d %b"))
            ui.dataframe(up.rename(columns={"event_date": "Date", "event_code": "Code", "event_name": "Event"})
                         .set_index("Date"), width="stretch")
    if held:
        hourly_book = M.load_hourly_q(tuple(sorted(held)))
        if not hourly_book.empty:
            sp = SS.book_session_pnl(hourly_book, book.exposure_nonusd, days=20)
            fig = go.Figure([go.Bar(x=sp.index, y=sp[z], name=f"{z} {clock.SESSION_SGT[z]}",
                                    marker=dict(color=ui.SESSION_COLORS[z]),
                                    hovertemplate=f"{z} %{{x|%d %b}}: %{{y:$,.0f}}<extra></extra>") for z in SS.ZONES])
            fig.update_layout(barmode="relative",
                              title="What-if: TODAY's exposures over the last 20 days, by session (actual P&L "
                                    "by session is on the Overview tab)")
            fig.update_yaxes(tickformat="$~s")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            ui.plot(fig, height=320, key="sess_book")
            tot = sp.sum()
            st.caption(ui.md_safe("Total over the window: " +
                                  " · ".join(f"{z} {ui.fmt_usd(tot[z], signed=True)}" for z in SS.ZONES)))


def render_analytics(book: E.Book, md: M.MarketData, ma: MarketAnalytics | None) -> None:
    if ma is None:
        st.error("Market analytics are unavailable (see Diagnostics).")
        return
    views = ["Rich / cheap", "Seasonality", "DXY beta & peers", "Sessions"]
    view = st.segmented_control("View", views, default="Rich / cheap", key="ana_view", label_visibility="collapsed")
    view = view or "Rich / cheap"
    if view == "Rich / cheap":
        _rich_cheap(book, ma)
    elif view == "Seasonality":
        _seasonality(book, md)
    elif view == "DXY beta & peers":
        _dxy_beta(book, ma)
        st.divider()
        ui.section("Peers: is it the bloc or the currency itself?")
        _peers(book, ma)
    else:
        _sessions(book, md)


# ============================================================================ diagnostics
def _clear_caches() -> None:
    _bind_session()
    st.cache_data.clear()
    log.info("Caches cleared by user")
    st.session_state["flash"] = "Caches cleared - data reloaded"


def _toggle_debug() -> None:
    _bind_session()
    level = "DEBUG" if st.session_state.get("debug_log") else "INFO"
    set_level(level)
    log.info("Log level set to %s", level)


def _data_checks(md: M.MarketData, xc: XC.CrossCheck | None) -> None:
    ui.section("Data checks", "Bad Yahoo prints removed from history, and held currencies checked against Yahoo's "
                              "quoted crosses (a second price series, separate from the USD pairs every number uses).")
    if xc is not None:
        st.caption(xc.message)
        if not xc.table.empty:
            t = xc.table.drop(columns=["days_checked"], errors="ignore").rename(columns={
                "check": "Check cross", "legs": "Built from", "implied_now": "Implied today", "quoted_now": "Quoted today",
                "gap_now_bps": "Gap today (bps)", "typical_gap_bps": "Typical gap (bps)",
                "p90_gap_bps": "1-in-10-day gap (bps)", "days_checked": "Days checked",
                "days_over_50bps": "Days > 50 bps", "max_gap_bps": "Largest gap (bps)", "max_gap_date": "On",
                "status": "Status"})
            t.index.name = "Ccy"
            ui.dataframe(ui.style_table(t, price=["Implied today", "Quoted today"],
                                        num={"Gap today (bps)": "+.1f", "Typical gap (bps)": ".1f",
                                             "1-in-10-day gap (bps)": ".1f", "Largest gap (bps)": ".0f",
                                             "Days > 50 bps": ".0f"}),
                         width="stretch")
            st.caption("A reference cross is used only if it normally agrees with our prices (typical gap ≤ "
                       f"{XC.RELIABLE_MEDIAN_BPS:.0f} bps, 1-in-10-day gap ≤ {XC.RELIABLE_P90_BPS:.0f} bps); "
                       "otherwise the currency shows ⚪ Not checked rather than raising false alarms.")
        if not xc.flagged.empty:
            st.markdown(f"**Days in the VaR window where a quoted cross and the USD legs disagreed by more than "
                        f"{XC.GAP_BPS:.0f} bps**")
            f = xc.flagged.assign(date=pd.to_datetime(xc.flagged["date"]).dt.strftime("%Y-%m-%d")).rename(
                columns={"date": "Date", "check": "Cross", "ccy": "Ccy", "gap_bps": "Gap (bps)", "note": "Explanation"})
            ui.dataframe(ui.style_table(f.set_index("Date"), num={"Gap (bps)": "+,.0f"}), width="stretch", height=220)
    corr = xc.corrections if xc is not None else md.corrections
    if corr.empty:
        st.caption("No bad prints found in the price history.")
        return
    st.markdown(f"**Bad prints removed from the price history ({len(corr)})**: each replaced by the midpoint of the "
                "day before and the day after")
    cols = {"date": "Date", "ticker": "Ticker", "bad_price": "Bad price", "clean_price": "Replaced with",
            "jump_in_%": "Move in %", "jump_out_%": "Move out %", "size_sigma": "Size (σ)", "cross_check": "Cross check"}
    d = corr[[c for c in cols if c in corr.columns]].rename(columns=cols)
    d["Date"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    ui.dataframe(ui.style_table(d.set_index("Date"), price=["Bad price", "Replaced with"],
                                num={"Move in %": "+.2f", "Move out %": "+.2f", "Size (σ)": ".0f"}),
                 width="stretch", height=260)


def render_diagnostics(md: M.MarketData, book: E.Book | None, xc: XC.CrossCheck | None = None,
                       ma: MarketAnalytics | None = None, alerts: list[tuple[str, str]] | None = None) -> None:
    if alerts:
        ui.section("Flags", "Limits, data quality and model checks (the header badge counts the orange and red ones).")
        icons = {"error": "🔴", "warning": "🟠", "info": "🔵"}
        for level, msg in alerts:
            st.markdown(f"{icons[level]} {ui.md_safe(msg)}")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        ui.section("Data status", f"Source: {md.source} · rows {len(md.q)} · last close {md.last_date:%Y-%m-%d} · "
                                  f"loaded {clock.fmt(md.fetched_at, '%H:%M:%S')}")
        ui.dataframe(md.status, width="stretch", height=260)
        for n in md.notes:
            st.caption(f"• {n}")
    with c2:
        ui.section("Sanity checks")
        if book is None:
            st.caption("Book not built.")
        else:
            checks = {**book.checks, **(ma.checks if ma is not None else {}), **(xc.checks if xc is not None else {})}
            for name, ok in checks.items():
                st.markdown(f"{'✅' if ok else '❌'} {name}")
        ui.section("Tools")
        st.button("🔄 Refresh prices / clear cache", on_click=_clear_caches, width="stretch")
        st.toggle("Debug logging (all sessions)", key="debug_log", on_change=_toggle_debug)
        st.download_button("⬇ Download app.log", data=read_log_bytes(), file_name="app.log", mime="text/plain",
                           width="stretch", on_click="ignore")
        st.caption(f"Python {platform.python_version()} · streamlit {st.__version__} · pandas {pd.__version__} · "
                   f"numpy {np.__version__} · yfinance {yf.__version__} · plotly {plotly.__version__}")

    _data_checks(md, xc)

    ui.section("Log", "Newest at the bottom. Search an error ID shown in a red box to see its full traceback.")
    l1, l2 = st.columns([1, 2])
    level = l1.selectbox("Minimum level", ["INFO", "WARNING", "ERROR", "DEBUG"], key="log_level")
    query = l2.text_input("Find error ID or text", key="log_query", placeholder="e.g. 7F3K2A")
    lines = find_in_log(query) if query else tail_log(80, level)
    st.code("\n".join(lines) or "(no matching log lines)", language=None, height=320, wrap_lines=True)
