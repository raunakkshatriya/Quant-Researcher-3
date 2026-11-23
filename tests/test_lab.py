import numpy as np
import pandas as pd

import risk
from lab import trials
from lab.stats import alpha_beta, deflated_sharpe_prob, expected_max_sharpe, sharpe
from lab.walkforward import walk_forward


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(10, 0.05) < expected_max_sharpe(1000, 0.05)


def test_best_of_many_random_strategies_is_not_convincing():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0, 0.01, (2500, 100)))        # 100 strategies with ZERO edge
    best = sharpe(R).idxmax()
    std = (R.mean() / R.std()).std()
    assert deflated_sharpe_prob(R[best], 100, std) < 0.95      # luck is correctly flagged
    assert deflated_sharpe_prob(R[best], 1, std) > deflated_sharpe_prob(R[best], 100, std)


def test_real_edge_survives_deflation():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))               # Sharpe about 1.6
    assert deflated_sharpe_prob(r, 10, 0.02) > 0.95


def test_walk_forward_only_uses_the_past():
    idx = pd.bdate_range("2010-01-01", "2020-12-31")
    rng = np.random.default_rng(2)
    R = pd.DataFrame({"A": rng.normal(0.002, 0.01, len(idx)), "B": rng.normal(0, 0.01, len(idx))}, index=idx)
    R.loc[R.index.year == 2020, "B"] += 0.05                   # B is wildly good, but ONLY in 2020
    _, picks = walk_forward(R)
    assert picks[2020] == "A"                                   # in 2020 we could not know B would win


def test_risk_limits():
    w = pd.DataFrame({"A": [0.9, 0.3], "B": [0.6, 0.3]})
    out = risk.apply_limits(w, max_position=0.5, max_gross=0.8)
    assert out.abs().sum(axis=1).max() <= 0.8 + 1e-12 and out.abs().max().max() <= 0.5 + 1e-12


def test_trial_count_ignores_reruns(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "DIR", tmp_path)
    monkeypatch.setattr(trials, "TRIALS", tmp_path / "trials.csv")
    row = [{"params": "k=1", "train_sharpe": 1.0, "n_obs": 10, "data_end": "2025-01-01"}]
    trials.log_trials("s", "v1", row); trials.log_trials("s", "v1", row)
    assert trials.trial_count("s") == 1                          # same settings re-run: not a new trial
    trials.log_trials("s", "v2", row)
    assert trials.trial_count("s") == 2                          # new logic version: counts again


def test_alpha_beta_recovers_known_alpha_and_beta():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = 0.0004 + 0.8 * bench + pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)   # +10%/yr alpha, beta 0.8
    r = alpha_beta(strat, bench)
    assert abs(r["beta"] - 0.8) < 0.05
    assert 0.07 < r["alpha"] < 0.13
    assert r["t_alpha"] > 3


def test_alpha_beta_no_alpha_when_strategy_is_just_the_benchmark():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = 1.2 * bench + pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)
    r = alpha_beta(strat, bench)
    assert abs(r["beta"] - 1.2) < 0.08
    assert abs(r["t_alpha"]) < 2.5
    assert alpha_beta(strat.iloc[:10], bench.iloc[:10]) == {}


# ---------------- portfolio views ----------------
from lab.portfolio import (combine_equal, combine_inverse_vol, common_window, lookthrough,
                           risk_contributions, rolling_avg_correlation)


def _books(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    a = pd.Series(rng.normal(0.0004, 0.02, n), index=idx)
    b = pd.Series(rng.normal(0.0002, 0.005, n), index=idx)
    return a, b


def test_common_window_keeps_only_days_everyone_has():
    a, b = _books()
    df = common_window({"a": a, "b": b.iloc[100:]})
    assert df.index[0] == b.index[100] and df.notna().all().all()


def test_rolling_avg_correlation_detects_identical_and_unrelated_books():
    a, b = _books()
    same = pd.DataFrame({"x": a, "y": a * 2})
    assert rolling_avg_correlation(same, window=250).min() > 0.999
    diff = pd.DataFrame({"x": a, "y": b})
    assert abs(rolling_avg_correlation(diff, window=250).mean()) < 0.2


def test_risk_contributions_sum_to_one_and_volatile_book_dominates():
    a, b = _books()
    rc = risk_contributions(pd.DataFrame({"a": a, "b": b}))
    assert abs(rc.sum() - 1) < 1e-9 and rc["a"] > 0.9               # a is 4x as volatile as b
    twin = risk_contributions(pd.DataFrame({"a": a, "c": a}))
    assert abs(twin["a"] - 0.5) < 1e-9


def test_inverse_vol_uses_only_past_volatility():
    a, b = _books()
    df = pd.DataFrame({"a": a, "b": b})
    base = combine_inverse_vol(df, window=100)
    df2 = df.copy()
    df2.iloc[500:] = df2.iloc[500:] * 5                              # rewrite the future
    changed = combine_inverse_vol(df2, window=100)
    pd.testing.assert_series_equal(base.iloc[:390], changed.iloc[:390])    # up to ~day 490 nothing may move
    assert combine_equal(df).iloc[0] == (a.iloc[0] + b.iloc[0]) / 2


def test_lookthrough_groups_and_cash():
    holdings = {"s1": pd.Series({"AAA": 0.3, "BBB": 0.3, "CCC": 0.2}), "s2": pd.Series({"AAA": 1.0})}
    t = lookthrough(holdings, {"AAA": "tech", "BBB": "tech", "CCC": "health"})
    assert abs(t.loc["tech", "s1"] - 0.6) < 1e-12 and abs(t.loc["health", "s1"] - 0.2) < 1e-12
    assert abs(t.loc["cash", "s1"] - 0.2) < 1e-12 and abs(t.loc["cash", "s2"]) < 1e-12
    assert (t.sum() - 1).abs().max() < 1e-12
