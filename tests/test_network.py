"""
Live-data checks (skipped by default). Run with:   FXDASH_NETWORK_TESTS=1 pytest tests/test_network.py
They verify the Yahoo timestamp handling against ICE's DX-Y.NYB.
"""
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("FXDASH_NETWORK_TESTS") != "1", reason="network test")


def test_synthetic_dxy_aligns_with_ice_dxy_at_lag_zero(monkeypatch):
    monkeypatch.delenv("FXDASH_OFFLINE", raising=False)
    import market_data as M
    from analytics.beta import synthetic_dxy

    md = M._load_live()
    syn = np.log(synthetic_dxy(md.q)).diff()
    ice = M._download_close(["DX-Y.NYB"], period="1y", interval="1d")["DX-Y.NYB"]
    ice.index = pd.DatetimeIndex(ice.index).normalize()
    ice = np.log(ice).diff()
    corr = {lag: pd.concat([syn.shift(lag), ice], axis=1).dropna().corr().iloc[0, 1] for lag in (-1, 0, 1)}
    assert corr[0] > 0.8, corr
    assert corr[0] > corr[-1] and corr[0] > corr[1], corr
