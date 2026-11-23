"""News sentiment strategy: hold the stocks with the most positive recent news.

build_weights(prices, score_sum, count, window, k)
  1. For each stock, average the sentiment of all articles in the last `window` trading days
     (needs at least MIN_ARTICLES articles, otherwise no opinion on that stock).
  2. On the last trading day of each week, hold the top k stocks, 1/k each. Everything else is cash.
Weights only use news that was already public at that day's close (see signal.py), so there is no look-ahead.
"""
import pandas as pd

MIN_ARTICLES = 3        # fewer articles in the window = too noisy to rank
MIN_STOCKS = 15         # need at least this many stocks with an opinion to trade that week


def week_end_dates(index):
    """The last trading day of each calendar week."""
    return pd.Series(index, index=index).groupby(index.to_period("W")).last().values


def build_weights(prices, score_sum, count, window, k):
    score_sum = score_sum.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    count = count.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    s, c = score_sum.rolling(window).sum(), count.rolling(window).sum()
    score = (s / c).where(c >= MIN_ARTICLES)
    weights = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    for day in week_end_dates(prices.index):
        row = score.loc[day].dropna()
        if len(row) < max(MIN_STOCKS, k):
            continue
        w = pd.Series(0.0, index=prices.columns)
        w[row.sort_values(kind="stable").index[-k:]] = 1.0 / k
        weights.loc[day] = w
    return weights.ffill()
