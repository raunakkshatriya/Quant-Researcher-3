"""SMA trend filter: hold each stock only while its price is above its simple moving average.

The simplest trend-following rule there is. We use it as a BASELINE: any fancier strategy should
have to beat it. Read description.txt first.
"""
import re

import pandas as pd

from strategies.momentum.strategy import STOCKS

# The 49 stocks: the momentum list minus GOOG (same company as GOOGL), same list as the pairs strategy.
TICKERS = [t for t in STOCKS if t != "GOOG"]

# The rules the lab tries. "price>SMA50" = hold a stock while its price is above its 50-day average.
# "SMA50>SMA200" = hold while the 50-day average is above the 200-day average (the classic "golden cross").
RULES = ["price>SMA50", "price>SMA100", "price>SMA150", "price>SMA200", "SMA50>SMA200"]


def parse_rule(rule):
    """'price>SMA100' -> (0, 100);  'SMA50>SMA200' -> (50, 200).  fast = 0 means 'the price itself'."""
    m = re.fullmatch(r"(price|SMA(\d+))>SMA(\d+)", rule)
    if not m:
        raise ValueError(f"unknown rule {rule!r}; expected like 'price>SMA200' or 'SMA50>SMA200'")
    return int(m.group(2) or 0), int(m.group(3))


def build_weights(prices, rule):
    """Return a table of weights (same shape as prices).

    Each day, for each stock:  in an uptrend (rule is true) -> hold it;  otherwise -> hold none of it.
    Money is split equally among all stocks that EXIST with enough history that day (so every stock that
    is in an uptrend gets 1 / (number of stocks available)). Stocks not in an uptrend are simply not held:
    that share of the money stays in cash and earns T-bills. A stock needs `slow` days of history first.
    """
    fast, slow = parse_rule(rule)
    slow_avg = prices.rolling(slow, min_periods=slow).mean()            # uses today's close and the days before only
    compare = prices if fast == 0 else prices.rolling(fast, min_periods=fast).mean()
    uptrend = (compare > slow_avg) & slow_avg.notna()                   # NaN comparisons are False: no history, no position
    available = slow_avg.notna().sum(axis=1).replace(0, float("nan"))   # stocks with enough history today
    return uptrend.astype(float).div(available, axis=0).fillna(0.0)


def equal_weight_index(prices):
    """A single price series: equal-weight buy-and-hold of every stock that exists on each day (rebalanced daily,
    a simplification). Used as the fair benchmark: 'did timing beat just owning the same stocks?'.
    A stock with no price yet is simply not in the average, so a new listing does not create a jump."""
    ret = prices.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    return 100 * (1 + ret).cumprod()
