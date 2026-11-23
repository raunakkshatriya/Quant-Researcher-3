"""Strategies must be CAUSAL: weights up to a date must not change if later prices are removed."""
import numpy as np
import pandas as pd
import pytest

from strategies.momentum.strategy import build_weights as momentum
from strategies.sector_rotation.strategy import build_weights as sector
from strategies.sma.strategy import build_weights as sma
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
    (sma, {"rule": "price>SMA100"}, 20),
    (sma, {"rule": "SMA50>SMA200"}, 20),
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


# ---------------------------------------------------------------- FX: continuing a stale rate series
def _monthly(start, end, value):
    idx = pd.date_range(start, end, freq="MS")
    return pd.Series(value, index=idx, dtype=float)


def _daily(start, end, value):
    return pd.Series(value, index=pd.bdate_range(start, end), dtype=float)


def test_splice_extends_stale_series_with_level_match_and_keeps_history():
    primary = _monthly("2024-01-01", "2026-01-01", 0.02)                  # stale after Jan 2026
    successor = _daily("2024-01-01", "2026-11-20", 0.025)                  # policy rate, 50bp higher, still updating
    out, info = data_fx.splice_rate(primary, successor)
    pd.testing.assert_series_equal(out.loc[:"2026-01-01"], primary)        # history untouched
    assert out.index[-1] >= pd.Timestamp("2026-10-01")
    assert abs(out.loc["2026-06-01"] - 0.02) < 1e-9                        # successor 0.025 minus the 50bp gap
    assert abs(info["offset_bp"] + 50) < 1e-6 and info["months_added"] >= 9


def test_splice_not_used_when_original_is_current_or_overlap_too_short():
    current = _monthly("2024-01-01", "2026-10-01", 0.02)
    out, info = data_fx.splice_rate(current, _daily("2024-01-01", "2026-11-20", 0.03))
    assert info is None and out.equals(current)
    stale_short = _monthly("2026-01-01", "2026-02-01", 0.02)               # only 2 overlapping months
    out, info = data_fx.splice_rate(stale_short, _daily("2026-01-01", "2026-11-20", 0.03))
    assert info is None


def test_splice_makes_recent_carry_known_instead_of_stale():
    idx = pd.bdate_range("2026-01-01", "2026-11-20")
    stale = _monthly("2023-01-01", "2026-01-01", 0.02)
    assert data_fx.daily_rate(stale, idx).iloc[-1] != data_fx.daily_rate(stale, idx).iloc[-1]   # NaN: stale
    fixed, _ = data_fx.splice_rate(stale, _daily("2023-01-01", "2026-11-20", 0.02))
    assert data_fx.daily_rate(fixed, idx).iloc[-1] == pytest.approx(0.02)


# ---------------------------------------------------------------- SMA trend filter
from strategies.sma import strategy as sma_mod


def test_sma_rule_parsing():
    assert sma_mod.parse_rule("price>SMA200") == (0, 200)
    assert sma_mod.parse_rule("SMA50>SMA200") == (50, 200)
    with pytest.raises(ValueError):
        sma_mod.parse_rule("price<SMA200")


def test_sma_holds_rising_stock_and_drops_falling_stock():
    idx = pd.bdate_range("2010-01-01", "2012-12-31")
    n = len(idx)
    px = pd.DataFrame({"UP": 100 * 1.001 ** np.arange(n), "DOWN": 100 * 0.999 ** np.arange(n)}, index=idx)
    w = sma_mod.build_weights(px, "price>SMA50")
    assert (w.iloc[:49] == 0).all().all()                     # not enough history for a 50-day average yet
    assert (w["UP"].iloc[60:] == 0.5).all()                   # in an uptrend: 1/2 of the money (2 stocks available)
    assert (w["DOWN"] == 0).all()                             # below its average: not held, share stays in cash
    assert w.sum(axis=1).max() <= 1.0 + 1e-12


def test_sma_crossover_needs_fast_above_slow():
    idx = pd.bdate_range("2010-01-01", "2013-12-31")
    n = len(idx)
    up_then_down = np.concatenate([100 * 1.002 ** np.arange(n // 2), 100 * 1.002 ** (n // 2) * 0.998 ** np.arange(n - n // 2)])
    px = pd.DataFrame({"X": up_then_down}, index=idx)
    w = sma_mod.build_weights(px, "SMA50>SMA200")["X"]
    assert w.iloc[250] == 1.0 and w.iloc[-1] == 0.0           # held during the rise, out after the fall


def test_equal_weight_index_ignores_stocks_that_do_not_exist_yet():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    px = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    px.loc[:"2020-03-31", "B"] = np.nan                       # B is listed only from April
    ix = sma_mod.equal_weight_index(px)
    assert ix.iloc[-1] == pytest.approx(100.0)                # flat prices -> flat index, no jump when B appears
    px["B"] = px["B"] * np.where(px.index >= "2020-05-01", 2.0, 1.0)     # B doubles in one day
    assert sma_mod.equal_weight_index(px).iloc[-1] == pytest.approx(150.0)
