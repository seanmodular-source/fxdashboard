"""
FX Portfolio Monitor - Streamlit entry point.

    streamlit run app.py

Flow of every script run (Streamlit reruns this file on each interaction):
    1. logging + session id          (logger.py)
    2. trade log, synced both ways with data/portfolio.csv  (trades.py; sync_book + watch_book below)
    3. market data (cached 15 min)   (market_data.py)
    4. market analytics (cached)     (analytics/)
    5. book = positions + P&L + risk (engine.py)
    6. cross check of held currencies vs quoted EUR crosses  (crosscheck.py; never raises)
    7. sidebar ticket, header, three tabs, diagnostics  (views.py)
Each tab is wrapped so one failure shows an error id instead of breaking the page.
"""
from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

import clock
import crosscheck as XC
import engine as E
import market_data as M
import scenarios as S
import trades as T
import ui
import universe as U
import views
from analytics import MarketAnalytics, compute_market_analytics
from analytics import richcheap as RC
from logger import get_logger, new_error_id, set_session_id, setup_logging

st.set_page_config(page_title="FX Portfolio Monitor", page_icon="💱", layout="wide",
                   initial_sidebar_state="expanded")
setup_logging()
log = get_logger("app")
ss = st.session_state

if "sid" not in ss:
    ss["sid"] = uuid.uuid4().hex[:4]
    set_session_id(ss["sid"])
    log.info("New session started (streamlit %s, pandas %s)", st.__version__, pd.__version__)
set_session_id(ss["sid"])
ui.inject_css()

CONTROL_FLOW = {"RerunException", "StopException", "RerunData"}


def safe_render(name: str, fn, *args) -> None:
    """Run one page section; on failure log the traceback with an error id and show it to the user."""
    try:
        fn(*args)
    except Exception as exc:
        if type(exc).__name__ in CONTROL_FLOW:
            raise
        err = getattr(exc, "err_id", None)
        if err is None:
            err = new_error_id()
            log.exception("[%s] %s failed: %s", err, name, exc)
        st.error(f"**{name}** could not be displayed - error ID `{err}`. "
                 "Open 🛠 Diagnostics at the bottom of the page and search for this ID.")


@st.cache_data(show_spinner="Computing market analytics...")
def get_analytics(cache_key: str, _q: pd.DataFrame) -> MarketAnalytics:
    """Cached per market-data load (the leading underscore stops Streamlit hashing the frame)."""
    return compute_market_analytics(_q)


# ----------------------------------------------------------------------------- sidebar: settings
def settings_controls() -> E.Settings:
    d = E.Settings()
    st.caption(f"VaR / ES: historical simulation, {d.conf:.0%}, 1 day, last {d.lookback} completed days (fixed).")
    var_limit = st.number_input("VaR limit ($)", min_value=0.0, value=d.var_limit, step=5_000.0, format="%.0f",
                                key="set_var_limit")
    ccy_limit = st.number_input("Currency exposure limit ($)", min_value=0.0, value=d.ccy_limit, step=500_000.0,
                                format="%.0f", key="set_ccy_limit")
    st.toggle("Expand 📘 method notes", value=False, key="notes_expand_all",
              help="Open every 'How it works' box at once (they are closed by default; click one to open it)")
    return E.Settings(var_limit=var_limit, ccy_limit=ccy_limit)


# ----------------------------------------------------------------------------- sidebar: ticket
def _context_line(ccy: str, buying: bool, ma: MarketAnalytics | None) -> str | None:
    """One line of market context for a currency the ticket is about to buy or sell."""
    if ma is None or ccy == U.BASE_CCY or ccy not in ma.rich_cheap.index:
        return None
    rc, pz = ma.rich_cheap.loc[ccy], ma.peer_z.reindex([ccy]).iloc[0]
    beta = ma.betas["beta_63"].get(ccy, float("nan"))
    return (f"{'Buying' if buying else 'Selling'} **{ccy}**: {ui.fmt_z(rc['level_z_252'])} vs 1Y, "
            f"{ui.fmt_z(pz['peer_z_63'])} vs peers, 1M move {ui.fmt_z(rc['stretch_z_21'])}, USD β {beta:.2f}")


def _cross_context_line(pair: str, md: M.MarketData) -> str | None:
    """Rich/cheap of a non-USD pair itself (e.g. AUDNZD): the relative-value view of a cross trade."""
    base, quote = U.parse_pair(pair)
    if U.BASE_CCY in (base, quote):
        return None
    rc = RC.rich_cheap_table(RC.pair_prices(md.q, [pair]))
    if rc.empty:
        return None
    r = rc.iloc[0]
    return (f"**{pair}** itself: {ui.fmt_z(r['level_z_252'])} vs 1Y, {ui.fmt_z(r['level_z_63'])} vs 3M, "
            f"1M move {ui.fmt_z(r['stretch_z_21'])} (+ = {base} expensive vs {quote})")


@st.dialog("Pre-trade check", width="large")
def confirm_dialog(md: M.MarketData, ma: MarketAnalytics | None, settings: E.Settings, book: E.Book) -> None:
    p = ss.get("pending")
    if not p:
        st.write("Nothing to confirm.")
        return
    base, quote = U.parse_pair(p["pair"])
    st.markdown(f"#### {p['side']} {p['notional']:,.0f} {p['pair']} @ {p['price']:,.6g}")
    ui.method_note("pre_trade")
    st.caption(f"Trade date {pd.Timestamp(p['trade_date']):%d %b %Y} · market {p['market']:,.6g}")
    for w in ss.get("pending_warnings", []):
        st.warning(w)
    buying_base = p["side"] == "BUY"
    for ccy, buying in ((base, buying_base), (quote, not buying_base)):
        line = _context_line(ccy, buying, ma)
        if line:
            st.markdown(line)
    cross_line = _cross_context_line(p["pair"], md)
    if cross_line:
        st.markdown(cross_line)

    trial = T.with_trade(ss["trades"], T.make_trade(ss["trades"], p["pair"], p["side"], p["notional"], p["price"],
                                                    p["trade_date"]))
    betas = ma.betas["beta_63"] if ma is not None else pd.Series(dtype=float)
    after = E.build_book(trial, md, betas, settings)
    cmp = S.compare_books(E.book_summary(book, [base, quote]), E.book_summary(after, [base, quote]))
    cmp.index.name = "Metric"
    # Change is signed but not coloured: these are risk / exposure numbers, where up is not "good" (green)
    styler = ui.style_table(cmp, money=["Before", "After"]).format(
        lambda v: ui.fmt_usd(v, signed=True, compact=False), subset=["Change"])
    ui.dataframe(styler, width="stretch")
    if after.risk and settings.var_limit:
        used = after.risk.var_hs / settings.var_limit
        msg = f"VaR after trade: {ui.fmt_usd(after.risk.var_hs)} ({used:.0%} of limit)"
        (st.error if used > 1 else st.warning if used > 0.8 else st.success)(ui.md_safe(msg))

    c1, c2 = st.columns(2)
    if c1.button("Confirm trade", type="primary", width="stretch"):
        trade = T.make_trade(ss["trades"], p["pair"], p["side"], p["notional"], p["price"], p["trade_date"])
        views.commit_book(T.add_trade(ss["trades"], trade),
                          f"Booked {trade['trade_id']}: {trade['side']} {trade['notional']:,.0f} {trade['pair']} @ "
                          f"{trade['price']:,.6g}")
        ss.pop("pending", None)
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        ss.pop("pending", None)
        log.info("Pre-trade preview cancelled for %s", p["pair"])
        st.rerun()


def render_ticket(md: M.MarketData, ma: MarketAnalytics | None, settings: E.Settings, book: E.Book) -> None:
    st.markdown("### 🎫 Trade ticket")
    pairs = [p for p in U.TRADABLE_PAIRS if all(md.has(c) for c in U.parse_pair(p))]
    pair = st.selectbox("Pair", pairs, key="tk_pair")
    base, _quote = U.parse_pair(pair)
    mkt = md.pair_price(pair)
    dp = U.price_decimals(pair) + 1
    st.caption(f"Market {mkt:,.{dp}f}")
    today = clock.today_sgt()
    with st.form("ticket", border=False):
        side = st.segmented_control("Side", ["BUY", "SELL"], default="BUY", key="tk_side")
        notional = st.number_input(f"Notional ({base})", min_value=0.0, value=1_000_000.0, step=250_000.0,
                                   format="%.0f", key="tk_notional")
        price = st.number_input("Price", min_value=0.0, value=float(round(mkt, dp)), step=10.0 ** -(dp - 1),
                                format=f"%.{dp}f")
        trade_date = st.date_input("Trade date", value=clock.trading_day_now().date(), max_value=today.date(),
                                   help="Trading day, 08:00 SGT to 08:00 SGT: a trade done before 08:00 SGT "
                                        "belongs to the previous day")
        submitted = st.form_submit_button("Preview trade", type="primary", width="stretch")
    if not submitted:
        return
    td = pd.Timestamp(trade_date)
    ref = md.pair_price(pair, md.q_on(td)) if td < md.last_date else mkt
    errors, warnings = T.validate_ticket(pair, side or "", notional, price, td, ref, today,
                                         all(md.has(c) for c in U.parse_pair(pair)))
    if errors:
        log.info("Ticket rejected: %s", errors)
        for e in errors:
            st.error(e)
        return
    ss["pending"] = {"pair": pair, "side": side, "notional": notional, "price": price, "trade_date": td,
                     "market": ref}
    ss["pending_warnings"] = warnings
    log.info("Pre-trade preview: %s %s %s @ %s", side, f"{notional:,.0f}", pair, price)
    confirm_dialog(md, ma, settings, book)


# ----------------------------------------------------------------------------- book file sync
BOOK_POLL_S = 3  # how often an open dashboard checks data/portfolio.csv for outside edits


def sync_book() -> None:
    """File -> dashboard: (re)load data/portfolio.csv on the first run and whenever its signature changed.
    A file with errors is not loaded: the last valid book stays on screen and the errors are flagged."""
    sig = T.book_signature()
    if "trades" in ss and sig == ss.get("book_sig"):
        return
    df, errors = T.read_book()
    ss["book_sig"] = T.book_signature()          # read_book may have just created the file
    if errors:
        ss["book_errors"] = errors
        if "trades" not in ss:                   # first load and the file is broken: start with an empty book
            ss["trades"] = pd.DataFrame(columns=T.COLUMNS)
        return
    reloaded = "trades" in ss
    ss["trades"] = df
    ss.pop("book_errors", None)
    if reloaded:
        ss["flash"] = f"Book reloaded from {T.book_path().name}: {len(df)} trades"
        log.info("Book reloaded after an outside edit: %d trades", len(df))


@st.fragment(run_every=BOOK_POLL_S)
def watch_book() -> None:
    """Runs every few seconds on its own: if the CSV was edited outside the app, rerun the page to reload it."""
    if T.book_signature() != ss.get("book_sig"):
        st.rerun(scope="app")


# ----------------------------------------------------------------------------- main
def main() -> None:
    sync_book()
    watch_book()
    flash = ss.pop("flash", None)
    if flash:
        st.toast(flash, icon="✅")

    with st.sidebar:
        ticket_box = st.container()
        with st.expander("⚙ Settings", expanded=False):
            settings = settings_controls()
        st.button("🔄 Refresh prices", on_click=views._clear_caches, width="stretch")

    try:
        md = M.load_market_data()
    except Exception as exc:
        err = getattr(exc, "err_id", None) or new_error_id()
        log.exception("[%s] No market data at all: %s", err, exc)
        st.error(f"Market data could not be loaded (live and fallback failed) - error ID `{err}`.")
        st.stop()

    ma: MarketAnalytics | None
    try:
        ma = get_analytics(f"{md.source}|{md.fetched_at}|{md.last_date}", md.q)
    except Exception as exc:
        err = getattr(exc, "err_id", None) or new_error_id()
        log.exception("[%s] Market analytics failed: %s", err, exc)
        ma = None
    betas = ma.betas["beta_63"] if ma is not None else pd.Series(dtype=float)

    try:
        book = E.build_book(ss["trades"], md, betas, settings)
        xc = XC.run(md, list(book.exposure_nonusd.index), settings.lookback)
        extra = list(xc.alerts)
        extra += [("error", f"{T.book_path().name} not loaded - {e} (the last valid book is shown)")
                  for e in ss.get("book_errors", [])]
        if ss.get("book_save_error"):
            extra.append(("error", ss["book_save_error"]))
        if ma is None:
            extra.append(("warning", "Market analytics failed - Analytics tab unavailable (see Diagnostics)."))
        else:
            extra += [("warning", f"Sanity check failed: {name} (see Diagnostics)")
                      for name, ok in ma.checks.items() if not ok]
        alerts = E.build_alerts(book, md, settings, extra)
    except Exception as exc:
        err = getattr(exc, "err_id", None) or new_error_id()
        if not hasattr(exc, "err_id"):
            log.exception("[%s] Building the book failed: %s", err, exc)
        st.error(f"The portfolio could not be valued - error ID `{err}`. Try **Reset to sample** below or check "
                 "Diagnostics.")
        st.button("Reset to sample", on_click=views._reset_cb)
        with st.expander("🛠 Diagnostics", expanded=True):
            safe_render("Diagnostics", views.render_diagnostics, md, None)
        st.stop()

    with ticket_box:
        safe_render("Trade ticket", render_ticket, md, ma, settings, book)

    views.render_header(md, book, settings, alerts)
    tabs = st.tabs(["Overview", "Positions & risk", "Analytics"])
    with tabs[0]:
        safe_render("Overview", views.render_overview, book, md, settings)
    with tabs[1]:
        safe_render("Positions & risk", views.render_positions_risk, book, md, ma, settings)
    with tabs[2]:
        safe_render("Analytics", views.render_analytics, book, md, ma)

    st.divider()
    with st.expander("🛠 Diagnostics: data status, data checks, sanity checks and logs"):
        safe_render("Diagnostics", views.render_diagnostics, md, book, xc, ma, alerts)


main()
