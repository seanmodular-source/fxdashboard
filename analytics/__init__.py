"""
Market analytics: is the book well positioned?

compute_market_analytics(q) runs once per market-data load (it does not depend on trades) and returns
everything the Analytics tab, the pre-trade context and the DXY-equivalent need.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import universe as U
from logger import get_logger, log_step

from . import beta as beta_mod
from . import peers as peers_mod
from . import richcheap as rc_mod

log = get_logger(__name__)

HORIZONS = (5, 21)


@dataclass
class MarketAnalytics:
    dxy: pd.Series                        # synthetic DXY level
    betas: pd.DataFrame                   # per ccy: beta_63, r2_63, beta_252, r2_252  (USD-beta, ~ +1)
    rich_cheap: pd.DataFrame              # per ccy: level z, move z, % change (richcheap.py)
    decomposition: dict[int, pd.DataFrame]  # horizon -> per ccy: total, dollar, bloc, own  (% log moves)
    resid_z: pd.DataFrame                 # per ccy: residual z for each horizon
    peer_z: pd.DataFrame                  # per ccy: peer_z_63, peer_z_252
    scatter: pd.DataFrame                 # per ccy: beta_63, usd_ret_21 (%), group
    dxy_ret_21: float                     # synthetic DXY 21-day % log move
    checks: dict[str, bool] = field(default_factory=dict)  # calculation self-checks (shown in Diagnostics)


def layered_decomposition(lr: pd.DataFrame, dollar: pd.DataFrame, bloc: pd.DataFrame, own: pd.DataFrame,
                          h: int) -> pd.DataFrame:
    """Sum of the last h daily parts, in %. total = dollar + bloc + own exactly."""
    tail = slice(-h, None)
    out = pd.DataFrame({
        "total": lr.iloc[tail].sum(),
        "dollar": dollar.iloc[tail].sum(),
        "bloc": bloc.iloc[tail].sum(),
        "own": own.iloc[tail].sum(),
    }) * 100
    gap = (out["total"] - out[["dollar", "bloc", "own"]].sum(axis=1)).abs().max()
    if gap > 1e-9:
        log.warning("Layered decomposition does not add up (max gap %.2e %%)", gap)
    return out


def residual_z(resid: pd.DataFrame, h: int, vol_window: int = 63) -> pd.Series:
    move = resid.iloc[-h:].sum()
    sigma = resid.iloc[-(h + vol_window):-h].std()
    return move / (sigma * np.sqrt(h))


@log_step
def compute_market_analytics(q: pd.DataFrame) -> MarketAnalytics:
    b = beta_mod.beta_analytics(q)
    lr, dollar, resid = b["lr"], b["dollar"], b["resid"]
    bloc, own = peers_mod.bloc_and_own(resid)

    decomposition = {h: layered_decomposition(lr, dollar, bloc, own, h) for h in HORIZONS}
    resid_z = pd.DataFrame({f"resid_z_{h}": residual_z(resid, h) for h in HORIZONS})
    peer_z = pd.DataFrame({"peer_z_63": peers_mod.peer_z(own, 63), "peer_z_252": peers_mod.peer_z(own, 252)})

    dxy = b["dxy"]
    dxy_ret_21 = float(np.log(dxy.iloc[-1] / dxy.iloc[-22]) * 100)
    usd_ret_21 = -(np.log(q.iloc[-1] / q.iloc[-22]) * 100)
    scatter = pd.DataFrame({
        "beta_63": b["table"]["beta_63"],
        "usd_ret_21": usd_ret_21,
        "group": pd.Series({c: U.PRIMARY_GROUP.get(c, "Other") for c in q.columns}),
    }).dropna()

    gaps = [(d["total"] - d[["dollar", "bloc", "own"]].sum(axis=1)).abs().max() for d in decomposition.values()]
    checks = {"Move breakdown adds up (dollar + bloc + own = total)": bool(np.nanmax(gaps) < 1e-9)}

    return MarketAnalytics(
        dxy=dxy, betas=b["table"], rich_cheap=rc_mod.rich_cheap_table(q), decomposition=decomposition,
        resid_z=resid_z, peer_z=peer_z, scatter=scatter, dxy_ret_21=dxy_ret_21, checks=checks,
    )
