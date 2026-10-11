"""
Look and feel: colour tokens, Plotly template, number formatters, KPI cards, table styling.

Colour rules (one meaning per colour, nothing decorative):
  * Long / short exposure and other signed quantities -> blue / orange (diverging pair)
  * Up / down: P&L, z-scores, % moves                 -> green / red, always with a +/- sign printed as well
    (teal-green #0e8a6c vs red #d64545: validated, colour-blind separation dE 9.2; the sign is the backup cue)
  * Correlations                                      -> blue / red diverging, neutral grey at 0
  * Sessions Asia / Europe / US                       -> blue / orange / aqua (validated CVD-safe)
"""
from __future__ import annotations

import html
import math

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- tokens
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
UP = "#0e8a6c"           # positive: gains, rich, currency stronger
DOWN = "#d64545"         # negative: losses, cheap, currency weaker
UP_TEXT = "#0a6b54"      # darker steps for text on the light surface
DOWN_TEXT = "#b02f2f"
GAIN, LOSS, GAIN_TEXT, LOSS_TEXT = UP, DOWN, UP_TEXT, DOWN_TEXT
WARN = "#fab219"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_MID = "#f0efec"
SESSION_COLORS = {"Asia": BLUE, "Europe": ORANGE, "US": AQUA}
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, NEUTRAL_MID], [0.75, "#f0a3a3"], [1.0, "#b42a2a"]]
Z_DIVERGING = [[0.0, DOWN], [0.25, "#f2a8a3"], [0.5, NEUTRAL_MID], [0.75, "#94d1bc"], [1.0, UP]]  # red - grey - green
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

pio.templates["fxdash"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, size=12, color=INK_2),
    title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=1, yref="container", yanchor="top",
               pad=dict(t=12, l=4)),
    colorway=[BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="", showgrid=False),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks=""),
    hoverlabel=dict(font=dict(family=FONT, size=12)),
    margin=dict(l=8, r=16, t=80, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
    bargap=0.25,
))
pio.templates.default = "plotly_white+fxdash"

CSS = """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 2rem; max-width: 1500px;}
.fx-header {display:flex; flex-wrap:wrap; align-items:center; gap:.6rem 1rem; margin-bottom:.4rem;}
.fx-title {font-size:1.55rem; font-weight:650; letter-spacing:-.01em; margin-right:.4rem;}
.fx-badge {display:inline-flex; align-items:center; gap:.35rem; font-size:.8rem; padding:.18rem .6rem;
           border-radius:999px; border:1px solid rgba(11,11,11,.12); color:#52514e; background:rgba(11,11,11,.03);}
.fx-badge.good {color:#006300; border-color:rgba(12,163,12,.35); background:rgba(12,163,12,.07);}
.fx-badge.warn {color:#8a5d00; border-color:rgba(250,178,25,.55); background:rgba(250,178,25,.10);}
.fx-badge.bad  {color:#b42a2a; border-color:rgba(208,59,59,.40); background:rgba(208,59,59,.08);}
.kpi-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:.75rem; margin:.4rem 0 1rem;}
.kpi {border:1px solid rgba(11,11,11,.10); border-radius:12px; padding:.75rem .9rem; background:rgba(255,255,255,.6);}
.kpi-label {font-size:.78rem; color:#52514e; display:flex; justify-content:space-between; gap:.4rem;}
.kpi-help {cursor:help; color:#898781;}
.kpi-value {font-size:1.45rem; font-weight:650; margin-top:.15rem; color:#0b0b0b; white-space:nowrap;}
.kpi-value.good {color:#006300;} .kpi-value.bad {color:#b42a2a;} .kpi-value.warn {color:#8a5d00;}
.kpi-sub {font-size:.75rem; color:#898781; margin-top:.1rem;}
.fx-section {font-size:1.02rem; font-weight:620; margin:1.1rem 0 .3rem;}
.fx-caption {font-size:.8rem; color:#898781; margin-top:-.2rem; margin-bottom:.4rem;}
div[data-testid="stTabs"] button p {font-size:.95rem; font-weight:560;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- formatters
def _is_num(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def fmt_usd(x, signed: bool = False, compact: bool = True) -> str:
    """$12.4k / $1.25m; signed adds +/- (true minus sign for display)."""
    if not _is_num(x):
        return "–"
    sign = "+" if (signed and x > 0) else ("−" if x < 0 else "")
    a = abs(float(x))
    if compact and a >= 1e6:
        body = f"${a / 1e6:,.2f}m"
    elif compact and a >= 1e4:
        body = f"${a / 1e3:,.1f}k"
    else:
        body = f"${a:,.0f}"
    return sign + body


def fmt_price(x) -> str:
    if not _is_num(x):
        return "–"
    return f"{x:,.2f}" if abs(x) >= 20 else f"{x:,.4f}"


def fmt_z(x) -> str:
    return "–" if not _is_num(x) else f"{x:+.1f}σ"


def fmt_pct(x, decimals: int = 1, signed: bool = True) -> str:
    if not _is_num(x):
        return "–"
    return f"{x:+.{decimals}f}%" if signed else f"{x:.{decimals}f}%"


def md_safe(text: str) -> str:
    """Escape $ so Streamlit markdown does not read '$1 ... $2' as a LaTeX formula."""
    return text.replace("$", r"\$")


def tone(x) -> str:
    if not _is_num(x) or abs(x) < 0.5:
        return ""
    return "good" if x > 0 else "bad"


# ----------------------------------------------------------------------------- components
def kpi_cards(cards: list[dict]) -> None:
    """cards: [{label, value, sub, tone ('good'|'bad'|'warn'|''), help}] rendered as one responsive grid."""
    items = []
    for c in cards:
        help_html = (f'<span class="kpi-help" title="{html.escape(c["help"])}">ⓘ</span>' if c.get("help") else "")
        items.append(
            f'<div class="kpi"><div class="kpi-label"><span>{html.escape(c["label"])}</span>{help_html}</div>'
            f'<div class="kpi-value {c.get("tone", "")}">{html.escape(c["value"])}</div>'
            f'<div class="kpi-sub">{html.escape(c.get("sub", ""))}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def method_note(key: str) -> None:
    """'📘 How it works' box from notes.NOTES: plain-English explanation, the maths and the code lines.
    Closed by default (click to open); the 'Expand method notes' setting in the sidebar opens them all."""
    from notes import NOTES  # local import keeps ui.py importable on its own
    n = NOTES[key]
    with st.expander(f"📘 How it works: {n['title']}", expanded=bool(st.session_state.get("notes_expand_all", False))):
        st.markdown(md_safe(n["body"].strip()))
        if n.get("formulas"):
            st.code(n["formulas"], language=None, wrap_lines=True)


def badge(text: str, kind: str = "") -> str:
    return f'<span class="fx-badge {kind}">{html.escape(text)}</span>'


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="fx-section">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="fx-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def plot(fig: go.Figure, height: int | None = None, key: str | None = None) -> None:
    if height:
        fig.update_layout(height=height)
    elif fig.layout.height is None:
        fig.update_layout(height=340)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


# ----------------------------------------------------------------------------- tables
def _pnl_color(v) -> str:
    if not _is_num(v) or abs(v) < 0.5:
        return ""
    return f"color: {UP_TEXT if v > 0 else DOWN_TEXT}; font-weight: 560"


def _signed_color(v) -> str:
    if not _is_num(v) or v == 0:
        return ""
    return f"color: {UP_TEXT if v > 0 else DOWN_TEXT}; font-weight: 560"


def _z_bg(v) -> str:
    """Green tint for + z, red for - z; light from |z| >= 1, darker from |z| >= 2."""
    if not _is_num(v) or abs(v) < 1:
        return ""
    alpha = 0.16 if abs(v) < 2 else 0.34
    rgb = "14,138,108" if v > 0 else "214,69,69"
    return f"background-color: rgba({rgb},{alpha})"


def style_table(df: pd.DataFrame, money: list[str] = (), pnl: list[str] = (), price: list[str] = (),
                z: list[str] = (), pct: list[str] = (), signed: dict[str, str] | None = None,
                num: dict[str, str] | None = None):
    """Pandas Styler with consistent number formats and colour rules.
    ``signed`` = {column: format}: numbers shown with a sign and coloured green (+) / red (-)."""
    fmt = {}
    for c in money:
        fmt[c] = lambda v: fmt_usd(v, compact=False)
    for c in pnl:
        fmt[c] = lambda v: fmt_usd(v, signed=True, compact=False)
    for c in price:
        fmt[c] = fmt_price
    for c in z:
        fmt[c] = fmt_z
    for c in pct:
        fmt[c] = lambda v: fmt_pct(v, signed=False)
    for c, f in {**(signed or {}), **(num or {})}.items():
        fmt[c] = (lambda f_: (lambda v: "–" if not _is_num(v) else format(v, f_)))(f)
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    # Streamlit draws a missing number as 'None' whatever the format, so columns with gaps are pre-formatted as
    # text ('–' for gaps); ui.dataframe right-aligns them. Colours are still computed from the numbers.
    shown = df.copy()
    text_cols = [c for c in fmt if df[c].isna().any()]
    for c in text_cols:
        shown[c] = [fmt[c](v) for v in df[c]]
    styler = shown.style.format({c: f for c, f in fmt.items() if c not in text_cols})
    for cols, rule in (([c for c in pnl if c in df.columns], _pnl_color),
                       ([c for c in (signed or {}) if c in df.columns], _signed_color),
                       ([c for c in z if c in df.columns], _z_bg)):
        if cols:
            styler = styler.apply(lambda s, rule=rule: [rule(v) for v in df[s.name]], subset=cols)
    styler._fx_text_cols = text_cols
    return styler


def dataframe(data, **kwargs) -> None:
    """st.dataframe that keeps style_table's pre-formatted number columns right-aligned."""
    cols = getattr(data, "_fx_text_cols", [])
    if cols:
        cfg = dict(kwargs.pop("column_config", None) or {})
        for c in cols:
            cfg[c] = {**(cfg.get(c) or {}), "alignment": "right"}
        kwargs["column_config"] = cfg
    st.dataframe(data, **kwargs)


# ----------------------------------------------------------------------------- charts
def signed_bar(series: pd.Series, title: str, pos_color: str = BLUE, neg_color: str = ORANGE,
               pos_name: str = "Positive", neg_name: str = "Negative", value_fmt=None,
               neutral: list[str] | None = None, tick_suffix: str = "") -> go.Figure:
    """Horizontal bars sorted by value; sign carries meaning through colour + legend + label.
    Values are USD unless a custom ``value_fmt`` is given."""
    s = series.dropna().sort_values()
    neutral = neutral or []
    money = value_fmt is None
    value_fmt = value_fmt or (lambda v: fmt_usd(v, signed=True))
    colors = [MUTED if k in neutral else (pos_color if v >= 0 else neg_color) for k, v in s.items()]
    labels = [value_fmt(v) for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=labels, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>", showlegend=False,
    ))
    for name, color in ((pos_name, pos_color), (neg_name, neg_color)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=name, showlegend=True))
    lo, hi = min(0.0, float(s.min())), max(0.0, float(s.max()))
    span = max(hi - lo, 1e-9)
    pad, edge = 0.28 * span, 0.04 * span  # room for the outside labels; a sliver so zero never sits on the edge
    fig.update_layout(title=title, barmode="overlay", height=max(240, 34 * len(s) + 120),
                      xaxis=dict(showgrid=True, zeroline=True, range=[lo - (pad if lo < 0 else edge),
                                                                      hi + (pad if hi > 0 else edge)]),
                      yaxis=dict(showgrid=False, ticksuffix="  "))
    if money:
        fig.update_xaxes(tickformat="$~s")
    elif tick_suffix:
        fig.update_xaxes(ticksuffix=tick_suffix)
    return fig


def signed_vbar(series: pd.Series, title: str, value_fmt=None, pos_name: str = "Up", neg_name: str = "Down",
                yaxis_title: str = "") -> go.Figure:
    """Vertical bars in the series' own order (sessions, weekdays, months): green up / red down, signed labels.
    Values are USD unless a custom ``value_fmt`` is given."""
    s = series.astype(float)
    money = value_fmt is None
    value_fmt = value_fmt or (lambda v: fmt_usd(v, signed=True))
    colors = [MUTED if not _is_num(v) else (UP if v >= 0 else DOWN) for v in s.values]
    fig = go.Figure(go.Bar(
        x=list(s.index), y=s.values, marker=dict(color=colors, cornerradius=4), showlegend=False,
        text=[value_fmt(v) if _is_num(v) else "" for v in s.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{text}<extra></extra>"))
    for name, color in ((pos_name, UP), (neg_name, DOWN)):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=name, showlegend=True))
    vals = s.dropna()
    lo, hi = (min(0.0, float(vals.min())), max(0.0, float(vals.max()))) if len(vals) else (0.0, 1.0)
    pad = 0.18 * max(hi - lo, 1e-9)
    fig.update_layout(title=title, barmode="overlay", yaxis_title=yaxis_title,
                      yaxis=dict(range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)], zeroline=True))
    if money:
        fig.update_yaxes(tickformat="$~s")
    return fig
