"""Cross-sectional momentum: buy last period's winners (and optionally short the losers)."""
import pandas as pd

from engine import month_end_dates

# The 50 stocks we use: the 50 largest US companies on 2026-09-08 (same list as the older
# SMA study). Ticker -> company name.
STOCKS = {
    "NVDA": "Nvidia", "AAPL": "Apple", "GOOGL": "Alphabet (class A)", "GOOG": "Alphabet (class C)",
    "MSFT": "Microsoft", "AMZN": "Amazon", "AVGO": "Broadcom", "META": "Meta", "TSLA": "Tesla",
    "MU": "Micron", "LLY": "Eli Lilly", "JPM": "JPMorgan Chase", "WMT": "Walmart", "AMD": "AMD",
    "V": "Visa", "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil", "MA": "Mastercard",
    "INTC": "Intel", "ORCL": "Oracle", "ABBV": "AbbVie", "BAC": "Bank of America", "CSCO": "Cisco",
    "PLTR": "Palantir", "CVX": "Chevron", "COST": "Costco", "LRCX": "Lam Research",
    "KO": "Coca-Cola", "CAT": "Caterpillar", "MRK": "Merck", "AMAT": "Applied Materials",
    "UNH": "UnitedHealth", "GE": "GE Aerospace", "MS": "Morgan Stanley", "PG": "Procter & Gamble",
    "DELL": "Dell", "NFLX": "Netflix", "HD": "Home Depot", "GS": "Goldman Sachs",
    "PM": "Philip Morris", "WFC": "Wells Fargo", "PANW": "Palo Alto Networks", "RTX": "RTX",
    "GEV": "GE Vernova", "ANET": "Arista Networks", "KLAC": "KLA", "AMGN": "Amgen",
    "TXN": "Texas Instruments", "C": "Citigroup", "TMO": "Thermo Fisher",
}
BENCHMARK = "SPY"
SKIP_DAYS = 21   # ignore the most recent month (about 21 trading days): very recent winners tend to
                 # bounce back a little, which would pollute the signal ("short-term reversal")
MIN_STOCKS = 20  # don't rank if fewer than this many stocks have enough price history


def build_weights(prices, lookback, k, mode):
    """Return a table of weights (same shape as prices).

    On the last trading day of each month:
      1. Score each stock by its return from `lookback` days ago to 21 days ago.
      2. Sort the stocks by score.
      3. mode "long":        buy the top k stocks, 1/k each.
         mode "long_short":  buy the top k (0.5/k each) AND short the bottom k (-0.5/k each).
                             Total money at work stays at 1.0, long and short roughly cancel,
                             so the market's ups and downs mostly cancel out.
    Weights are held until the next month-end.
    """
    score = prices.shift(SKIP_DAYS) / prices.shift(lookback) - 1
    weights = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    for day in month_end_dates(prices.index):
        s = score.loc[day].dropna()
        if len(s) < max(MIN_STOCKS, 2 * k):
            continue
        ranked = s.sort_values()
        row = pd.Series(0.0, index=prices.columns)
        if mode == "long":
            row[ranked.index[-k:]] = 1.0 / k
        else:
            row[ranked.index[-k:]] = 0.5 / k
            row[ranked.index[:k]] = -0.5 / k
        weights.loc[day] = row
    return weights.ffill()
