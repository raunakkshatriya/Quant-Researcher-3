"""The capacity study must charge costs that grow with size, never use today's volume to price today's trade,
and never produce free trades from missing data."""
import numpy as np
import pandas as pd
import pytest

from lab import capacity as cap


def _market(n_days=800, seed=1, n=3, volume=5e6):
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    cols = [f"S{i}" for i in range(n)]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, (n_days, n)), axis=0)), index=idx, columns=cols)
    volumes = pd.DataFrame(volume, index=idx, columns=cols)
    return prices, volumes


def _flip_weights(prices):
    """Rebalance between assets every 20 days so there is steady trading."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(0, len(prices), 20):
        w.iloc[i:i + 20, (i // 20) % prices.shape[1]] = 0.9
    return w


def test_impact_follows_square_root_law():
    prices, volumes = _market()
    adv = cap.adv_dollars(prices, volumes)
    vol = cap.daily_volatility(prices)
    q = pd.DataFrame(1e5, index=prices.index, columns=prices.columns)
    c1 = cap.trade_cost_bps(q, adv, vol)
    c4 = cap.trade_cost_bps(q * 4, adv, vol)
    spread = cap.half_spread_bps(adv)
    ok = c1.notna() & adv.notna()
    ratio = ((c4 - spread)[ok] / (c1 - spread)[ok]).stack().dropna()
    assert len(ratio) > 100 and ratio.round(6).eq(2.0).all()                           # 4x the size -> 2x the impact


def test_spread_tiers_follow_liquidity_and_unknown_is_not_free():
    idx = pd.bdate_range("2020-01-01", periods=3)
    adv = pd.DataFrame({"mega": 5e9, "large": 3e8, "mid": 5e7, "thin": 1e6, "unknown": np.nan}, index=idx)
    sp = cap.half_spread_bps(adv).iloc[0]
    assert sp["mega"] < sp["large"] < sp["mid"] < sp["thin"]
    assert sp["unknown"] >= sp["thin"]


def test_todays_volume_never_prices_todays_trade():
    prices, volumes = _market()
    base = cap.adv_dollars(prices, volumes)
    changed = volumes.copy()
    changed.iloc[400] *= 100                                   # a volume spike on day 400
    after = cap.adv_dollars(prices, changed)
    pd.testing.assert_frame_equal(base.iloc[:401], after.iloc[:401])     # day 400's own liquidity estimate unchanged
    assert (after.iloc[401] != base.iloc[401]).all()                     # it only affects the following days


def test_costs_rise_with_account_size_and_net_sharpe_falls():
    prices, volumes = _market()
    w = _flip_weights(prices)
    table = cap.capacity_table(prices, volumes, w, [1e3, 1e5, 1e7, 1e9], prices.index[60], prices.index[-1])
    assert table["avg_cost_bp"].is_monotonic_increasing
    assert table["sharpe"].is_monotonic_decreasing
    assert table["p95_part_pct"].is_monotonic_increasing
    assert table["avg_cost_bp"].iloc[0] < table["avg_cost_bp"].iloc[-1] / 3


def test_tiny_account_pays_only_about_the_spread():
    prices, volumes = _market(volume=5e7)                      # ~$5bn a day per stock: top liquidity tier, 1 bp half-spread
    _, st = cap.run_at_aum(prices, volumes, _flip_weights(prices), aum=1e3)
    assert 0.9 < st["avg_cost_bp"] < 2.0


def test_no_trading_no_cost_after_the_first_purchase():
    prices, volumes = _market()
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["S0"] = 1.0                                              # buy once, hold
    net, st = cap.run_at_aum(prices, volumes, w, aum=1e6)
    gross = cap.engine.run_backtest(prices, w)["ret"]
    diff = (gross - net)
    assert (diff > 0).sum() == 1                               # exactly one day paid a cost


def test_cost_keeps_rising_beyond_a_days_volume_and_missing_volume_is_not_free():
    prices, volumes = _market()
    adv = cap.adv_dollars(prices, volumes)
    vol = cap.daily_volatility(prices)
    sizes = [1e3, 1e8, 1e9, 1e10, 1e11, 1e12]                  # from tiny up to many times a whole day's volume
    costs = [cap.trade_cost_bps(pd.DataFrame(s, index=prices.index, columns=prices.columns), adv, vol).iloc[100:].mean().mean()
             for s in sizes]
    assert all(np.isfinite(costs)) and all(b > a for a, b in zip(costs, costs[1:]))     # strictly rising: no plateau
    below = pd.DataFrame(0.999 * adv.iloc[100:].mean().mean(), index=prices.index, columns=prices.columns)
    above = below * (1.001 / 0.999)
    c_b, c_a = (cap.trade_cost_bps(x, adv, vol).iloc[100:].mean().mean() for x in (below, above))
    assert abs(c_a - c_b) / c_b < 0.01                          # smooth join where the square-root law hands over
    nan_adv = adv * np.nan
    unknown = cap.trade_cost_bps(pd.DataFrame(1e5, index=prices.index, columns=prices.columns), nan_adv, vol)
    assert (unknown.iloc[60:] >= cap.UNKNOWN_SPREAD_BP).all().all()


def test_find_capacity_interpolates_on_log_scale():
    aums = [1e3, 1e4, 1e5, 1e6]
    values = [1.0, 0.8, 0.4, 0.0]
    assert cap.find_capacity(aums, values, 0.6) == pytest.approx(10 ** 4.5)     # halfway between 0.8 and 0.4 in log10
    assert cap.find_capacity(aums, values, -1.0) is None                         # never reached
    assert cap.find_capacity(aums, values, 2.0) == 1e3                           # already below at the smallest size
