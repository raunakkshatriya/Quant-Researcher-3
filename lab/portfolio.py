"""Portfolio-level views across strategies: the things no single strategy's code can see.

Everything here takes plain tables of daily returns (one column per strategy) and is deliberately simple:
no optimisation, no fitted weights. Choosing weights that look best on the training period would be another round of
curve-fitting, so the combined book is equal-dollar (with one inverse-volatility comparison that only uses PAST volatility).
"""
import numpy as np
import pandas as pd


def common_window(returns):
    """dict name -> daily return Series  ->  DataFrame over the dates where EVERY strategy has a return."""
    return pd.DataFrame(returns).dropna()


def rolling_avg_correlation(df, window=252, step=5):
    """Average pairwise correlation between the columns over a trailing window (sampled every `step` days)."""
    n = df.shape[1]
    if n < 2 or len(df) < window:
        return pd.Series(dtype=float)
    idx, vals = [], []
    for i in range(window, len(df) + 1, step):
        c = df.iloc[i - window:i].corr().values
        vals.append((np.nansum(c) - n) / (n * (n - 1)))
        idx.append(df.index[i - 1])
    return pd.Series(vals, index=idx)


def combine_equal(df):
    """Equal dollars in every strategy, rebalanced daily (a simplification: real rebalancing would be less frequent)."""
    return df.mean(axis=1)


def combine_inverse_vol(df, window=252):
    """Each strategy's share is proportional to 1 / its volatility over the PREVIOUS `window` days (no look-ahead)."""
    vol = df.rolling(window).std().shift(1)
    w = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    w = w.div(w.sum(axis=1), axis=0)
    out = (w * df).sum(axis=1, min_count=1)
    return out[w.notna().all(axis=1)]


def risk_contributions(df, weights=None):
    """Share of the combined book's variance that each strategy accounts for (sums to 1)."""
    n = df.shape[1]
    w = pd.Series(1.0 / n, index=df.columns) if weights is None else pd.Series(weights).reindex(df.columns)
    cov = df.cov() * 252
    total = float(w @ cov @ w)
    return w * (cov @ w) / total


def lookthrough(holdings, groups, other="other"):
    """holdings: dict strategy -> Series(ticker -> weight). Returns group x strategy table of weights (plus 'cash')."""
    cols = {}
    for name, w in holdings.items():
        w = w[w.abs() > 1e-12]
        g = w.groupby(lambda t: groups.get(t, other)).sum()
        g["cash"] = 1.0 - w.sum()
        cols[name] = g
    return pd.DataFrame(cols).fillna(0.0)
