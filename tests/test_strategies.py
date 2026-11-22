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


# ---------------------------------------------------------------- pairs trading
from strategies.pairs import strategy as pairs


def _pair_world(seed=3):
    """4 stocks sharing a common trend with mean-reverting gaps (should be pairable) + 4 unrelated random walks."""
    idx = pd.bdate_range("2003-01-01", "2024-12-31")
    n, rng = len(idx), np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0.0003, 0.012, n))

    def ou(theta=0.05, sig=0.01):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] * (1 - theta) + rng.normal(0, sig)
        return x

    cols = {t: np.exp(4 + common + ou()) for t in "ABCD"}
    cols.update({t: np.exp(4 + np.cumsum(rng.normal(0.0003, 0.015, n))) for t in "EFGH"})
    return pd.DataFrame(cols, index=idx), {"linked": list("ABCD"), "unrelated": list("EFGH")}


def test_pairs_no_lookahead():
    px, groups = _pair_world()
    cutoff = px.index[int(len(px) * 0.7)]
    full = pairs.build_weights(px, 2.0, 0.0, groups).loc[:cutoff]
    trunc = pairs.build_weights(px.loc[:cutoff], 2.0, 0.0, groups).loc[:cutoff]
    pd.testing.assert_frame_equal(full, trunc)


def test_pairs_finds_linked_stocks_and_respects_limits():
    px, groups = _pair_world()
    chosen = pd.concat([f["chosen"] for f in pairs.formation(px, groups)])
    assert len(chosen) > 0
    assert set(chosen["a"]).union(chosen["b"]) <= set("ABCD") | set("EFGH")
    share_linked = chosen["a"].isin(list("ABCD")).mean()
    assert share_linked > 0.8                       # the linked group is chosen far more often than random walks
    assert chosen["beta"].between(pairs.BETA_MIN, pairs.BETA_MAX).all()
    assert chosen["half_life"].between(pairs.HL_MIN, pairs.HL_MAX).all()
    w = pairs.build_weights(px, 1.5, 0.0, groups)
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9  # never more than 8 slots of 1/8
    assert (w.abs().sum(axis=1) > 0).any()


def test_pairs_each_ticker_in_at_most_one_pair_per_year():
    px, groups = _pair_world()
    for f in pairs.formation(px, groups):
        used = list(f["chosen"]["a"]) + list(f["chosen"]["b"])
        assert len(used) == len(set(used))


def test_half_life_recovers_known_value():
    rng = np.random.default_rng(0)
    x = np.zeros(20000)
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + rng.normal()        # true half-life = ln(0.5)/ln(0.9) = 6.58 days
    assert abs(pairs.half_life(x) - 6.58) < 0.5
    assert pairs.half_life(np.cumsum(rng.normal(size=5000))) > 60   # a random walk does not mean-revert


def test_pairs_survives_years_where_no_ticker_has_data():
    """Sector ETFs start in 1998-2018 but share a date index with older stocks: the early formation
    windows are ALL NaN and must be skipped, not crash (this was a real KeyError)."""
    px, groups = _pair_world()
    px = px[list("ABCD")]
    groups = {"only_group": list("ABCD")}
    px.loc[:"2010-12-31", :] = np.nan                     # nothing exists yet
    w = pairs.build_weights(px, 2.0, 0.0, groups)
    assert w.loc[:"2010-12-31"].abs().sum().sum() == 0
    assert (w.abs().sum(axis=1) > 0).any()                # and it trades once data exists
    assert len(pairs.formation_summary(px, groups)) > 0
