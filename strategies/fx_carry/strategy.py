"""FX carry + momentum: buy the currencies with the highest interest rates / strongest trend, short the weakest.

Read description.txt first. Weights are on the TOTAL-RETURN index of each currency (spot move + interest
differential, see data_fx.py), so the ordinary backtest engine earns the carry automatically.
"""
import pandas as pd

from engine import month_end_dates

K = 3          # currencies on each side
MIN_CCY = 6    # do not trade if fewer currencies than this have all the data


def build_weights(prices, w_carry, lookback, carry, spot, k=K):
    """Return a table of weights (same shape as prices).

    On the last trading day of each month:
      1. carry score    = how high the currency's interest-rate differential vs USD is (higher = better)
         momentum score = the currency's exchange-rate change over the last `lookback` days (higher = better)
      2. Turn each score into a rank between 0 and 1 across the currencies, then
         combined = w_carry * carry rank + (1 - w_carry) * momentum rank
         (w_carry = 1: carry only, 0: momentum only, 0.5: half and half)
      3. BUY the top k (0.5/k each), SHORT the bottom k (-0.5/k each). Gross exposure 1.0, long and short
         roughly cancel, so the dollar's overall moves mostly cancel out. Weights are held until next month-end.
    A currency is only eligible if it has an exchange rate, a known carry and enough history on that day.
    """
    spot = spot.reindex(prices.index)[prices.columns]
    carry = carry.reindex(prices.index)[prices.columns]
    mom = spot / spot.shift(lookback) - 1
    weights = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    for day in month_end_dates(prices.index):
        ok = carry.loc[day].notna() & mom.loc[day].notna() & spot.loc[day].notna()
        row = pd.Series(0.0, index=prices.columns)
        if ok.sum() >= max(MIN_CCY, 2 * k):
            score = (w_carry * carry.loc[day][ok].rank(pct=True)
                     + (1 - w_carry) * mom.loc[day][ok].rank(pct=True)).sort_values(kind="stable")
            row[score.index[-k:]] = 0.5 / k
            row[score.index[:k]] = -0.5 / k
        weights.loc[day] = row
    return weights.ffill()
