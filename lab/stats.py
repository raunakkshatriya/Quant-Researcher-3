"""Statistics that stop us fooling ourselves.

THE PROBLEM
If you try many versions of a strategy, the best one looks good partly by luck.
Try 100 useless random strategies and the best will show a respectable Sharpe ratio.
The functions here estimate how much of an observed Sharpe could be luck.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import engine

EULER_GAMMA = 0.5772156649


def _excess(r):
    """Subtract what cash would have earned (the engine's cash rate; zero if none is set)."""
    cash = engine.daily_cash(r.index)
    return r.sub(cash, axis=0) if isinstance(r, pd.DataFrame) else r - cash


def sharpe(returns, periods=252):
    """Annualised Sharpe ratio of EXCESS returns (over cash). Works on a Series or each DataFrame column."""
    r = returns.dropna() if isinstance(returns, pd.Series) else returns
    return _excess(r).mean() / r.std() * np.sqrt(periods)


def versus_benchmark(strategy_ret, benchmark_ret, periods=252):
    """Is the strategy really better than the benchmark, or just different?

    Works on the DIFFERENCE between the two daily return series ("active return").
      active_return : average yearly outperformance
      tracking_error: yearly volatility of that difference
      info_ratio    : active_return / tracking_error  (Sharpe of the difference)
      t_stat        : |t| above about 2 means the outperformance is unlikely to be luck.
    (The t-stat ignores autocorrelation, so treat it as a rough guide.)
    """
    a = (strategy_ret - benchmark_ret).dropna()
    n = len(a)
    if n < 60 or a.std() == 0:
        return {}
    return {"active_return": a.mean() * periods, "tracking_error": a.std() * np.sqrt(periods),
            "info_ratio": a.mean() / a.std() * np.sqrt(periods), "t_stat": a.mean() / (a.std() / np.sqrt(n))}


def expected_max_sharpe(n_trials, sharpe_std):
    """Largest (per-period) Sharpe we expect to see by pure luck among `n_trials` useless trials.

    sharpe_std = how spread out the trials' Sharpe ratios are (per-period units).
    Bailey & Lopez de Prado (2014), 'The Deflated Sharpe Ratio'.
    """
    if n_trials < 2 or not np.isfinite(sharpe_std) or sharpe_std <= 0:
        return 0.0
    return sharpe_std * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                         + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe_prob(returns, n_trials, sharpe_std):
    """Probability that the strategy's TRUE Sharpe is above zero after accounting for luck.

    returns    : the chosen strategy's daily returns (train period)
    n_trials   : how many different versions we tried in total
    sharpe_std : spread of the per-period Sharpe ratios across those versions

    Reads like a p-value in reverse: 0.95+ = probably a real edge; below ~0.5 = could easily
    be the best of a lucky bunch. It also corrects for fat tails and skew.
    """
    r = returns.dropna()
    t = len(r)
    if t < 30 or r.std() == 0:
        return float("nan")
    sr = _excess(r).mean() / r.std()             # per-period Sharpe of excess returns
    skew, kurt = r.skew(), r.kurt() + 3.0        # pandas gives EXCESS kurtosis; formula needs raw
    sr0 = expected_max_sharpe(n_trials, sharpe_std)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))
