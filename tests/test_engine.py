import numpy as np
import pandas as pd

import engine


def _px(vals):
    return pd.DataFrame({"A": vals}, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def _w(px, first_day):
    w = pd.DataFrame({"A": np.nan}, index=px.index)
    w.iloc[first_day] = 1.0
    return w.ffill()


def test_delay_zero_earns_next_day_but_delay_one_does_not():
    px = _px([100, 100, 150, 150, 150, 150, 150])
    w = _w(px, 1)  # decided with day-1 close
    assert engine.run_backtest(px, w, delay=0)["ret"].iloc[2] == 0.5   # trade day-1 close, earns day 2
    assert engine.run_backtest(px, w, delay=1)["ret"].iloc[2] == 0.0   # trade day-2 close: misses the jump


def test_flat_weights_flat_equity():
    px = _px([100, 101, 102, 103])
    res = engine.run_backtest(px, px * 0, cost_bps=10)
    assert abs(res["equity"].iloc[-1] - 100.0) < 1e-9


def test_round_trip_cost():
    px = _px([100] * 10)
    w = pd.DataFrame({"A": np.nan}, index=px.index)
    w.iloc[1] = 1.0
    w.iloc[5] = 0.0
    res = engine.run_backtest(px, w.ffill(), cost_bps=10, delay=0)
    assert abs(res["ret"].sum() + 2 * 0.001) < 1e-12  # buy then sell: 2 x 10 bp


def test_borrow_fee_only_on_shorts():
    px = _px([100] * 253)
    long = engine.run_backtest(px, px * 0 + 1.0, borrow_bps=100, delay=0)
    short = engine.run_backtest(px, px * 0 - 1.0, borrow_bps=100, delay=0)
    assert abs(long["ret"].sum()) < 1e-12
    assert abs(short["ret"].sum() + 0.01 * 251 / 252) < 1e-4


def test_month_end_drops_unfinished_month():
    idx = pd.bdate_range("2024-01-01", "2024-03-14")
    ends = engine.month_end_dates(idx)
    assert ends[-1] == pd.Timestamp("2024-02-29") and len(ends) == 2


def test_crisis_helpers():
    import config
    idx = pd.bdate_range("2007-01-01", "2010-12-31")
    r = pd.Series(0.001, index=idx)
    dropped = engine.drop_crises(r)
    assert len(dropped) < len(r)
    assert not dropped.loc["2008-06-01":"2008-06-30"].size     # inside the GFC window: removed
    assert dropped.loc["2010-06-01":"2010-06-30"].size          # outside: kept


def test_cash_earns_the_rate_when_flat():
    px = _px([100] * 253)
    rate = pd.Series(0.0252, index=px.index)                 # 2.52% a year = 0.01% a day
    res = engine.run_backtest(px, px * 0, cash_rate=rate, delay=0)
    assert abs(res["ret"].iloc[1:].mean() - 0.0001) < 1e-9


def test_cash_weight_is_one_minus_invested():
    px = _px([100] * 253)
    rate = pd.Series(0.0252, index=px.index)
    half = engine.run_backtest(px, px * 0 + 0.5, cash_rate=rate, delay=0)   # 50% in a flat asset, 50% cash
    assert abs(half["ret"].iloc[5] - 0.00005) < 1e-9


def test_sharpe_is_on_excess_returns():
    from lab.stats import sharpe
    idx = pd.bdate_range("2020-01-01", periods=500)
    r = pd.Series(np.random.default_rng(0).normal(0.0004, 0.01, 500), index=idx)
    engine.set_default_cash_rate(None)
    s0 = sharpe(r)
    engine.set_default_cash_rate(pd.Series(0.05, index=idx))
    s1 = sharpe(r)
    engine.set_default_cash_rate(None)
    assert s1 < s0                                            # beating 5% cash is harder than beating 0%


def test_versus_benchmark_detects_a_real_edge():
    from lab.stats import versus_benchmark
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.0003, 0.01, 2500))
    s = b + rng.normal(0.0004, 0.002, 2500)                   # a steady +0.04% a day over the benchmark
    assert versus_benchmark(s, b)["t_stat"] > 2
    assert abs(versus_benchmark(b + rng.normal(0, 0.002, 2500), b)["t_stat"]) < 3
