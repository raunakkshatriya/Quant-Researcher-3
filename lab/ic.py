"""Information coefficient (IC): does a signal today predict relative returns later? (Phase A check, before any backtest.)

Each day, rank the stocks by the signal and by their forward return, and take the correlation of the two rankings.
IC of 0 = the signal is useless; +0.02 to +0.05 per day is already good in practice; negative = backwards.
Forward return respects our trading delay: signal at close t, trade at close t+delay, measure to t+delay+horizon.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def forward_returns(prices, horizon, delay=None):
    delay = config.EXEC_DELAY_DAYS if delay is None else delay
    return prices.shift(-(delay + horizon)) / prices.shift(-delay) - 1


def _masked(signal, fwd, min_names):
    mask = signal.notna() & fwd.notna()
    ok = mask.sum(axis=1) >= min_names
    return signal.where(mask), fwd.where(mask), ok


def daily_ic(signal, fwd, min_names=15):
    """Rank correlation between signal and forward return, one number per day (NaN if too few stocks)."""
    s, f, ok = _masked(signal, fwd, min_names)
    ic = s.rank(axis=1).corrwith(f.rank(axis=1), axis=1)
    return ic.where(ok)


def quintile_spread(signal, fwd, min_names=15, q=5):
    """Average forward return of the top fifth minus the bottom fifth of stocks, per day."""
    s, f, ok = _masked(signal, fwd, min_names)
    r = s.rank(axis=1, pct=True)
    top = f.where(r > 1 - 1.0 / q).mean(axis=1)
    bot = f.where(r <= 1.0 / q).mean(axis=1)
    return (top - bot).where(ok)


def summarize(series, horizon=1):
    """Mean, t-stat (robust to the overlap between consecutive days when horizon > 1), share of positive days."""
    x = series.dropna()
    if len(x) < 60 or x.std() == 0:
        return {}
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": max(horizon, 1)})
    return {"mean": float(x.mean()), "t_stat": float(fit.tvalues[0]), "pct_positive": float((x > 0).mean()), "days": int(len(x))}
