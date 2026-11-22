"""Sector rotation: hold the best-performing sectors, switch once a month."""
import pandas as pd

from engine import month_end_dates

# The 11 sector funds we rotate between (SPDR Select Sector ETFs) and the market benchmark.
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
BENCHMARK = "SPY"


def build_weights(prices, lookback, k):
    """Return a table of weights (same shape as prices).

    On the last trading day of each month:
      1. For every sector, compute its return over the last `lookback` trading days.
      2. Pick the `k` sectors with the highest return.
      3. Give each of them weight 1/k. All other sectors get 0.
    Those weights are then held until the next month-end.
    """
    past_return = prices / prices.shift(lookback) - 1       # return over the last `lookback` days
    weights = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    for day in month_end_dates(prices.index):
        scores = past_return.loc[day].dropna()               # ignore funds with no history yet
        if len(scores) < 6:                                  # too few sectors to rank sensibly
            continue
        best = scores.sort_values().index[-k:]               # the k highest
        row = pd.Series(0.0, index=prices.columns)
        row[best] = 1.0 / k
        weights.loc[day] = row
    return weights.ffill()                                   # hold until the next month-end
