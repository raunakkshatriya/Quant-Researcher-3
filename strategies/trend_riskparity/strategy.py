"""Cross-asset trend following with inverse-volatility sizing ("risk parity lite")."""
import numpy as np
import pandas as pd

from engine import month_end_dates

# The nine funds we trade: one for each big "asset class".
ASSETS = {
    "SPY": "US large-cap stocks",
    "EFA": "Developed-market stocks outside the US",
    "EEM": "Emerging-market stocks",
    "TLT": "Long-term US government bonds (20+ years)",
    "IEF": "Medium-term US government bonds (7-10 years)",
    "LQD": "Investment-grade corporate bonds",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "VNQ": "US real estate (REITs)",
}
BENCHMARK_WEIGHTS = {"SPY": 0.6, "IEF": 0.4}   # the classic 60/40 portfolio, our fair comparison
VOL_WINDOW = 60       # days used to measure how wiggly each asset has been recently
MIN_ASSETS = 4        # don't trade until at least this many funds have enough history


def build_weights(prices, lookback, vol_target):
    """Return a table of weights (same shape as prices).

    On the last trading day of each month:
      1. TREND FILTER: an asset is "in an uptrend" if its price is higher than `lookback` days ago.
         Assets not in an uptrend get weight 0 (that money stays in cash).
      2. INVERSE-VOLATILITY SIZING: among uptrending assets, weight each by 1 / (its recent
         volatility), so a calm bond gets a bigger share than a wild stock. Shares add to 100%.
      3. VOLATILITY TARGET: estimate the whole portfolio's volatility from the last 60 days. If it
         is above `vol_target` (e.g. 0.08 = 8% a year), scale all weights down proportionally.
         We NEVER scale up (no leverage), so the portfolio can sit below its target.
    Whatever is not invested stays in cash, which we assume earns 0% (a known simplification).
    """
    returns = prices.pct_change()
    vol = returns.rolling(VOL_WINDOW).std() * np.sqrt(252)
    uptrend = prices / prices.shift(lookback) - 1 > 0
    weights = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    for day in month_end_dates(prices.index):
        ok = vol.loc[day].notna() & prices.shift(lookback).loc[day].notna()   # assets with enough history
        if ok.sum() < MIN_ASSETS:
            continue
        held = [c for c in prices.columns if ok[c] and bool(uptrend.loc[day, c]) and vol.loc[day, c] > 0]
        row = pd.Series(0.0, index=prices.columns)
        if held:
            inv = 1.0 / vol.loc[day, held]
            w = inv / inv.sum()
            cov = returns.loc[:day].iloc[-VOL_WINDOW:][held].cov() * 252
            port_vol = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
            scale = min(1.0, vol_target / port_vol) if port_vol > 0 else 1.0
            row[held] = w * scale
        weights.loc[day] = row
    return weights.ffill()
