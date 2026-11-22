import numpy as np
import pandas as pd

import risk
from lab import trials
from lab.stats import deflated_sharpe_prob, expected_max_sharpe, sharpe
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
