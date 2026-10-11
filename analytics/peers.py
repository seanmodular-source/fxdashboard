"""
Peer deviation: after removing the dollar, is the move the bloc's or the currency's own?

    bloc_c,t = median of peers' daily residuals (leave-one-out: c itself excluded)
    own_c,t  = residual_c,t - bloc_c,t
So every daily move splits exactly:  total = dollar + bloc + own.

Peer z = z-score of the cumulative 'own' series vs its prior N days (rich/cheap vs peers).
"""
from __future__ import annotations

import pandas as pd

import universe as U


def bloc_and_own(resid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bloc = {}
    for c in resid.columns:
        group = U.PRIMARY_GROUP.get(c)
        peers = [p for p in U.PEER_GROUPS.get(group, []) if p != c and p in resid.columns]
        bloc[c] = resid[peers].median(axis=1, skipna=True) if peers else pd.Series(0.0, index=resid.index)
    bloc_df = pd.DataFrame(bloc).fillna(0.0).where(resid.notna())
    return bloc_df, resid - bloc_df


def peer_z(own: pd.DataFrame, window: int) -> pd.Series:
    cum = own.fillna(0.0).cumsum()
    prior = cum.shift(1).rolling(window, min_periods=int(window * 0.8))
    return ((cum - prior.mean()) / prior.std()).iloc[-1]
