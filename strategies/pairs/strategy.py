"""Pairs trading (statistical arbitrage): bet that two linked stocks stop drifting apart.

One engine, two universes (see STOCK_GROUPS and ETF_GROUPS). Read description.txt first.

The yearly cycle (every number is a constant below, so it is easy to find and change):
  1. FORM   On the first trading day of each year, take the previous FORM_DAYS (504 = ~2 years)
            of prices. For every pair of tickers INSIDE one group, test whether the pair is
            cointegrated (its price gap keeps returning to an average).
  2. KEEP   Keep pairs with p-value < P_MAX, a sensible hedge ratio, and a half-life of
            HL_MIN..HL_MAX days. Rank by p-value, take the best MAX_PAIRS, each ticker used once.
  3. TRADE  For the rest of the year, watch the z-score of each kept pair's gap and trade it
            (enter at |z| > entry_z, exit when z comes back to exit_z, stop out at STOP_Z).
  4. RESET  On the next first trading day, everything is closed and the pairs are re-chosen.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

FORM_DAYS = 504     # trailing days used to choose pairs (about 2 years)
Z_WINDOW = 60       # days used to measure the gap's "normal" level and wiggle size
MAX_PAIRS = 8       # at most this many pairs at once; each gets 1/MAX_PAIRS of the money
P_MAX = 0.05        # cointegration test must beat this p-value
HL_MIN, HL_MAX = 3, 60          # half-life (days for the gap to close halfway) must be in this range
BETA_MIN, BETA_MAX = 0.2, 5.0   # hedge ratios outside this are not a sensible "pair"
STOP_Z = 4.0        # gap this many standard deviations out = relationship probably broken

# The 49 stocks (the older 50 minus GOOG, which is the same company as GOOGL), in 9 economic groups.
# Pairs are only formed INSIDE a group: that is our economic reason for two stocks to be linked.
STOCK_GROUPS = {
    "semiconductors": ["NVDA", "AVGO", "MU", "AMD", "INTC", "LRCX", "AMAT", "KLAC", "TXN"],
    "software_internet": ["GOOGL", "MSFT", "AMZN", "META", "ORCL", "NFLX", "PLTR", "PANW"],
    "hardware_networking": ["AAPL", "CSCO", "DELL", "ANET"],
    "banks_payments": ["JPM", "BAC", "MS", "GS", "WFC", "C", "V", "MA"],
    "health": ["LLY", "JNJ", "ABBV", "MRK", "UNH", "AMGN", "TMO"],
    "energy": ["XOM", "CVX"],
    "staples": ["WMT", "COST", "KO", "PG", "PM"],
    "industrials": ["CAT", "GE", "RTX", "GEV"],
    "consumer_discretionary": ["TSLA", "HD"],
}
# The 11 SPDR sector funds, all in ONE group (every pair of sectors is allowed: 55 pairs).
ETF_GROUPS = {"sector_etfs": ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]}

_CACHE = {}   # pair choices are the same for every (entry_z, exit_z) setting, so compute them once


def half_life(spread):
    """Days for the gap to close halfway, from a simple regression of its daily change on its level.
    Returns infinity if the gap does not mean-revert at all."""
    s = np.asarray(spread, dtype=float)
    lag, change = s[:-1], np.diff(s)
    x = lag - lag.mean()
    phi = (x * (change - change.mean())).sum() / (x * x).sum()     # change ~ phi * level
    b = 1.0 + phi                                                   # tomorrow's level ~ b * today's
    if not 0.0 < b < 1.0:
        return np.inf
    return float(-np.log(2) / np.log(b))


def test_pairs(window, groups):
    """Test every within-group pair in a window of prices (rows = days). Returns one row per pair tested:
         a, b       : the two tickers (alphabetical)
         beta       : hedge ratio from the regression  log(a) = alpha + beta * log(b)
         pvalue     : Engle-Granger cointegration p-value (smaller = stronger evidence)
         half_life  : days for the gap to close halfway
         passed     : True if it meets all the keep rules
       Tickers with any missing price in the window are skipped (they did not exist for the whole window)."""
    logp = np.log(window)
    rows = []
    for tickers in groups.values():
        have = [t for t in tickers if t in window.columns and window[t].notna().all()]
        for a, b in combinations(sorted(have), 2):
            ya, xb = logp[a].to_numpy(), logp[b].to_numpy()
            beta, alpha = np.polyfit(xb, ya, 1)
            pvalue = float(coint(ya, xb, maxlag=5, autolag=None)[1])
            hl = half_life(ya - beta * xb)
            passed = (pvalue < P_MAX) and (BETA_MIN <= beta <= BETA_MAX) and (HL_MIN <= hl <= HL_MAX)
            rows.append({"a": a, "b": b, "beta": float(beta), "pvalue": pvalue, "half_life": hl, "passed": bool(passed)})
    out = pd.DataFrame(rows, columns=["a", "b", "beta", "pvalue", "half_life", "passed"])
    out["passed"] = out["passed"].astype(bool)      # an empty table (no ticker has data yet) must still filter correctly
    return out


def choose_pairs(tested):
    """From the tested table keep the best MAX_PAIRS passing pairs, each ticker used at most once."""
    chosen, used = [], set()
    if tested.empty or not tested["passed"].any():
        return tested.iloc[0:0]
    for _, r in tested[tested["passed"]].sort_values("pvalue").iterrows():
        if r["a"] in used or r["b"] in used:
            continue
        chosen.append(r)
        used.update([r["a"], r["b"]])
        if len(chosen) == MAX_PAIRS:
            break
    return pd.DataFrame(chosen) if chosen else tested.iloc[0:0]


def year_starts(index):
    """Positions (row numbers) of the first trading day of each calendar year that has FORM_DAYS of history."""
    first = pd.Series(np.arange(len(index)), index=index).groupby(index.year).min()
    return [int(p) for p in first.to_numpy() if p >= FORM_DAYS]


def formation(prices, groups):
    """For every year: which pairs were tested, which passed, which were chosen. Cached."""
    key = (prices.shape, prices.index[0], prices.index[-1], tuple(prices.columns),
           tuple((g, tuple(t)) for g, t in groups.items()), round(float(prices.iloc[-1].sum()), 6))
    if key not in _CACHE:
        out = []
        for pos in year_starts(prices.index):
            tested = test_pairs(prices.iloc[pos - FORM_DAYS: pos + 1], groups)   # uses days up to and including pos
            out.append({"pos": pos, "year": prices.index[pos].year, "tested": tested, "chosen": choose_pairs(tested)})
        _CACHE[key] = out
    return _CACHE[key]


def formation_summary(prices, groups):
    """One row per year: pairs tested / passed the test / chosen. Shows how many 'passes' we would expect by luck."""
    rows = [{"year": f["year"], "tested": len(f["tested"]), "passed": int(f["tested"]["passed"].sum()),
             "chosen": len(f["chosen"])} for f in formation(prices, groups)]
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame(columns=["tested", "passed", "chosen"])


def _positions(z, entry_z, exit_z):
    """Walk through a pair's z-scores day by day. Returns +1 (long the gap: buy A, short B),
    -1 (short the gap: short A, buy B) or 0 (flat) for each day. Starts flat each year."""
    pos = np.zeros(len(z))
    state, blocked = 0, False
    for i, v in enumerate(z):
        if np.isnan(v):
            pos[i] = state
            continue
        if state != 0 and abs(v) > STOP_Z:          # relationship probably broken: take the loss, stand aside
            state, blocked = 0, True
        elif state == 1 and v >= -exit_z:           # gap was too low, has come back up: take profit
            state = 0
        elif state == -1 and v <= exit_z:           # gap was too high, has come back down: take profit
            state = 0
        if blocked and abs(v) < entry_z:            # gap is back to normal: allowed to trade again
            blocked = False
        if state == 0 and not blocked:
            if abs(v) > STOP_Z:                     # already beyond the stop level: do not enter, wait for normal
                blocked = True
            elif v <= -entry_z:
                state = 1
            elif v >= entry_z:
                state = -1
        pos[i] = state
    return pos


def build_weights(prices, entry_z, exit_z, groups):
    """Return a table of weights (same shape as prices).

    For each chosen pair (A, B, hedge ratio beta) the gap is  log(A) - beta*log(B),  turned into a z-score
    with a Z_WINDOW-day average and standard deviation. A pair uses a slot of 1/MAX_PAIRS of our money,
    split so the two legs hedge each other:   A gets 1/(1+beta) of the slot,   B gets beta/(1+beta).
       z <= -entry_z : gap unusually low  -> BUY A, SHORT B
       z >=  entry_z : gap unusually high -> SHORT A, BUY B
    Close when z returns to exit_z; stop out if |z| exceeds STOP_Z. Slots with no pair stay in cash.
    """
    cols = {c: i for i, c in enumerate(prices.columns)}
    logp = np.log(prices)
    W = np.zeros(prices.shape)
    forms = formation(prices, groups)
    slot = 1.0 / MAX_PAIRS
    for k, f in enumerate(forms):
        start = f["pos"]
        end = forms[k + 1]["pos"] if k + 1 < len(forms) else len(prices)      # this year's trading days: [start, end)
        for _, r in f["chosen"].iterrows():
            gap = logp[r["a"]] - r["beta"] * logp[r["b"]]
            gap = gap.iloc[start - Z_WINDOW: end]                              # a little history so z exists on day 1
            z = ((gap - gap.rolling(Z_WINDOW).mean()) / gap.rolling(Z_WINDOW).std()).iloc[Z_WINDOW:].to_numpy()
            p = _positions(z, entry_z, exit_z)
            W[start:end, cols[r["a"]]] += p * slot / (1.0 + r["beta"])
            W[start:end, cols[r["b"]]] -= p * slot * r["beta"] / (1.0 + r["beta"])
    return pd.DataFrame(W, index=prices.index, columns=prices.columns)
