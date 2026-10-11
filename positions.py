"""
Trades -> positions -> currency legs.

First principle: every spot trade is two currency exposures.
    BUY N EURUSD at P  ==  +N EUR (base leg)  and  -N*P USD (quote leg)
Signed notional N is + for BUY and - for SELL, so a SELL simply flips both legs.

All USD values use Q (USD per unit of currency) with Q[USD] = 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import universe as U


def live_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """LIVE trades with base/quote and signed notional columns added."""
    t = trades[trades["status"] == "LIVE"].copy()
    if t.empty:
        for col in ("base", "quote", "signed"):
            t[col] = pd.Series(dtype=object if col != "signed" else float)
        return t
    parts = t["pair"].map(U.parse_pair)
    t["base"] = parts.str[0]
    t["quote"] = parts.str[1]
    t["signed"] = np.where(t["side"] == "BUY", 1.0, -1.0) * t["notional"].astype(float)
    return t


def trade_legs(live: pd.DataFrame) -> pd.DataFrame:
    """One row per currency leg: trade_id, pair, ccy, amount (in that currency's units)."""
    if live.empty:
        return pd.DataFrame(columns=["trade_id", "pair", "ccy", "amount"])
    base = pd.DataFrame({"trade_id": live["trade_id"], "pair": live["pair"], "ccy": live["base"],
                         "amount": live["signed"]})
    quote = pd.DataFrame({"trade_id": live["trade_id"], "pair": live["pair"], "ccy": live["quote"],
                          "amount": -live["signed"] * live["price"].astype(float)})
    return pd.concat([base, quote], ignore_index=True)


def _q_full(q_row: pd.Series) -> pd.Series:
    q = q_row.copy()
    q[U.BASE_CCY] = 1.0
    return q


def exposure_by_currency(live: pd.DataFrame, q_row: pd.Series) -> pd.Series:
    """Net USD value held in each currency (USD row included; it carries no FX risk in a USD book)."""
    legs = trade_legs(live)
    if legs.empty:
        return pd.Series(dtype=float)
    legs["usd"] = legs["amount"] * legs["ccy"].map(_q_full(q_row))
    return legs.groupby("ccy")["usd"].sum().sort_values()


def holdings_units(live: pd.DataFrame) -> pd.Series:
    """Net amount held in each currency, in that currency's own units (e.g. AUD 4,000,000)."""
    legs = trade_legs(live)
    if legs.empty:
        return pd.Series(dtype=float)
    return legs.groupby("ccy")["amount"].sum()


def pair_exposure_matrix(live: pd.DataFrame, q_row: pd.Series) -> pd.DataFrame:
    """Rows = pairs, columns = non-USD currencies, values = USD exposure of that pair's legs."""
    legs = trade_legs(live)
    legs = legs[legs["ccy"] != U.BASE_CCY]
    if legs.empty:
        return pd.DataFrame()
    legs["usd"] = legs["amount"] * legs["ccy"].map(_q_full(q_row))
    return legs.pivot_table(index="pair", columns="ccy", values="usd", aggfunc="sum", fill_value=0.0)


def positions_table(live: pd.DataFrame, q_row: pd.Series) -> pd.DataFrame:
    """
    One row per pair: net notional, break-even entry, current price, USD size.

    Break-even (average entry) S* solves sum_k N_k (S - P_k) = 0  ->  S* = sum N_k P_k / sum N_k.
    It is undefined (NaN) when the pair is fully closed (net notional = 0).
    """
    if live.empty:
        return pd.DataFrame(columns=["base", "quote", "net_notional", "side", "avg_entry", "price_now",
                                     "size_usd", "n_trades"])
    q = _q_full(q_row)
    g = live.assign(np_=live["signed"] * live["price"].astype(float)).groupby("pair")
    out = pd.DataFrame({
        "base": g["base"].first(),
        "quote": g["quote"].first(),
        "net_notional": g["signed"].sum(),
        "sum_np": g["np_"].sum(),
        "n_trades": g.size(),
    })
    out.loc[out["net_notional"].abs() < 1e-6, "net_notional"] = 0.0
    out["avg_entry"] = np.where(out["net_notional"] != 0, out["sum_np"] / out["net_notional"].replace(0, np.nan),
                                np.nan)
    out["price_now"] = out["base"].map(q) / out["quote"].map(q)
    out["size_usd"] = out["net_notional"] * out["base"].map(q)
    out["side"] = np.select([out["net_notional"] > 0, out["net_notional"] < 0], ["LONG", "SHORT"], "FLAT")
    return out.drop(columns="sum_np")
