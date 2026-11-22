"""Strategies must be CAUSAL: weights up to a date must not change if later prices are removed."""
import numpy as np
import pandas as pd
import pytest

from strategies.momentum.strategy import build_weights as momentum
from strategies.sector_rotation.strategy import build_weights as sector
from strategies.trend_riskparity.strategy import build_weights as trend


def _prices(n_assets, seed=0):
    idx = pd.bdate_range("2005-01-03", "2024-12-31")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (len(idx), n_assets)), axis=0)),
                        index=idx, columns=[f"T{i}" for i in range(n_assets)])


@pytest.mark.parametrize("fn,kwargs,n", [
    (sector, {"lookback": 126, "k": 3}, 11),
    (momentum, {"lookback": 189, "k": 5, "mode": "long_short"}, 30),
    (trend, {"lookback": 126, "vol_target": 0.08}, 9),
])
def test_no_lookahead(fn, kwargs, n):
    px = _prices(n)
    cutoff = px.index[int(len(px) * 0.7)]
    full = fn(px, **kwargs).loc[:cutoff]
    trunc = fn(px.loc[:cutoff], **kwargs).loc[:cutoff]
    pd.testing.assert_frame_equal(full.fillna(0), trunc.fillna(0))
