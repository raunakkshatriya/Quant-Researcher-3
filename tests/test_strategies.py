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


# ---------------------------------------------------------------- FX carry + momentum
from engine import run_backtest
from strategies.fx_carry import data_fx
from strategies.fx_carry.strategy import build_weights as fx_weights


def _fx_world(seed=5, n_ccy=8):
    idx = pd.bdate_range("2004-01-01", "2024-12-31")
    rng = np.random.default_rng(seed)
    cols = [f"C{i}" for i in range(n_ccy)]
    spot = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.005, (len(idx), n_ccy)), axis=0)), index=idx, columns=cols)
    base = rng.uniform(-0.01, 0.05, n_ccy)
    carry = pd.DataFrame(base + 0.002 * np.sin(np.arange(len(idx))[:, None] / 300 + np.arange(n_ccy)), index=idx, columns=cols)
    return spot, carry, data_fx.total_return_index(spot, carry)


def test_fx_no_lookahead():
    spot, carry, tri = _fx_world()
    cutoff = tri.index[int(len(tri) * 0.7)]
    for w_carry in (0.0, 0.5, 1.0):
        full = fx_weights(tri, w_carry, 126, carry, spot).loc[:cutoff]
        trunc = fx_weights(tri.loc[:cutoff], w_carry, 126, carry.loc[:cutoff], spot.loc[:cutoff]).loc[:cutoff]
        pd.testing.assert_frame_equal(full.fillna(0), trunc.fillna(0))


def test_fx_carry_only_buys_highest_carry_and_shorts_lowest():
    spot, carry, tri = _fx_world()
    carry = carry * 0 + np.linspace(-0.02, 0.06, carry.shape[1])      # C0 lowest ... C7 highest, constant
    w = fx_weights(tri, 1.0, 126, carry, spot).dropna()
    last = w.iloc[-1]
    assert set(last[last > 0].index) == {"C5", "C6", "C7"}
    assert set(last[last < 0].index) == {"C0", "C1", "C2"}
    assert abs(last.sum()) < 1e-9 and abs(last.abs().sum() - 1.0) < 1e-9


def test_total_return_index_earns_the_interest_differential():
    idx = pd.bdate_range("2010-01-01", "2010-12-31")
    spot = pd.DataFrame({"X": 1.0}, index=idx)                         # exchange rate does not move
    carry = pd.DataFrame({"X": 0.05}, index=idx)                       # 5% a year differential
    tri = data_fx.total_return_index(spot, carry)
    assert abs(tri["X"].iloc[-1] / 100 - 1 - 0.05) < 0.003
    # short position in a high-carry currency PAYS the differential
    w = pd.DataFrame({"X": -0.5}, index=idx)
    res = run_backtest(tri, w, cost_bps=0, delay=0, cash_rate=pd.Series(0.0, index=idx))
    assert res["ret"].sum() < -0.02


def test_rate_is_not_used_before_it_could_have_been_known_and_goes_stale():
    monthly = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    d = data_fx.daily_rate(monthly, idx, max_stale_days=100)      # default lag: the one real runs use
    assert d.loc[:"2020-02-28"].isna().all()                           # January's value is known from 1 March only
    assert d.loc["2020-03-02"] == 0.01
    assert d.loc["2020-05-04"] == 0.03                                 # March's value arrives 1 May
    assert pd.isna(d.loc["2020-12-30"])                                # nothing new for months: stale, not trusted


def test_fx_currency_without_carry_is_never_traded():
    spot, carry, tri = _fx_world()
    carry["C3"] = np.nan
    w = fx_weights(tri, 0.5, 126, carry, spot)
    assert (w["C3"].fillna(0) == 0).all()
