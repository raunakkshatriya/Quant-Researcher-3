"""Capacity study: how much money can a strategy handle before trading costs eat its edge?

THE PROBLEM
Our backtests charge a flat 3-5 bp per $ traded. That is fine for a $100 account, but it cannot be true for $100 million:
big orders push the price against you. The industry's standard rule of thumb is the SQUARE-ROOT LAW:

      impact (as a fraction of price)  =  Y  x  daily volatility  x  sqrt( order size / average daily volume )

so doubling your order does NOT double the cost, but the cost keeps rising with size. Add the bid-ask spread and we get a
cost per trade that depends on (a) how big the trade is in dollars, (b) how liquid the stock is, (c) how jumpy it is.

WHAT THIS MODULE DOES
It re-runs a strategy's weights at many account sizes (AUM), charging each trade its own size-dependent cost, and reports
net performance versus size. It never looks at the test period and logs nothing to the trial log: it is a measurement
of fixed, already-chosen settings, not a search.

ASSUMPTIONS YOU SHOULD DISTRUST A LITTLE (all constants below, all changeable)
 * Y (IMPACT_COEF) comes from published studies, not from our own fills. 0.5-1.0 is typical; we use 1.0 (conservative)
   and the script also shows 0.5 and 2.0.
 * The half-spread by liquidity tier is a rough guide for US large caps and big ETFs.
 * AUM is held constant (not compounded), and we assume every order is worked at the close.
 * Past dollar volume was smaller than today's, so we measure the RECENT years only.
 * Orders above ~5% of a day's volume are outside the range where the formula was fitted; treat those rows as 'very expensive', not as exact.
"""
import numpy as np
import pandas as pd

import engine
from lab.stats import sharpe

IMPACT_COEF = 1.0        # Y in the square-root law
ADV_WINDOW = 20          # days used for average daily dollar volume and volatility
SQRT_LIMIT = 1.0         # up to this order/ADV the square-root law is used; beyond it cost keeps rising LINEARLY (slope 0.5, no plateau)
UNKNOWN_SPREAD_BP = 10.0
BELOW_TIERS_SPREAD_BP = 10.0   # known but thinner than every tier below
# (minimum average daily DOLLAR volume, half-spread in bp): the more liquid, the tighter the spread.
SPREAD_TIERS = [(1e9, 1.0), (1e8, 2.0), (2e7, 5.0)]


def adv_dollars(prices, volumes, window=ADV_WINDOW):
    """Average daily dollar volume known BEFORE each day's close (shifted one day: no peeking at the day we trade)."""
    return (prices * volumes).rolling(window, min_periods=window // 2).mean().shift(1)


def daily_volatility(prices, window=ADV_WINDOW):
    """Standard deviation of daily returns, known before each day's close."""
    return prices.pct_change(fill_method=None).rolling(window, min_periods=window // 2).std().shift(1)


def half_spread_bps(adv):
    """Half the bid-ask spread in bp, by liquidity tier. Unknown liquidity gets the worst case."""
    bp = pd.DataFrame(BELOW_TIERS_SPREAD_BP, index=adv.index, columns=adv.columns)    # below every tier
    for threshold, tier_bp in sorted(SPREAD_TIERS):          # ascending, so the largest threshold met wins
        bp = bp.where(~(adv >= threshold), tier_bp)          # (NaN >= x is False: unknown never matches a tier)
    return bp.where(adv.notna(), UNKNOWN_SPREAD_BP)


def trade_cost_bps(trade_dollars, adv, vol, impact_coef=IMPACT_COEF):
    """Cost in bp per $ traded for each (day, ticker): half-spread + square-root impact.

    Beyond SQRT_LIMIT (an order bigger than a whole day's volume) the square-root law is no longer trustworthy, and in
    reality such an order could not be done in one day at all. We keep the cost RISING (linearly, continuing smoothly
    from the square-root curve) instead of letting it level off, so huge sizes can never look cheap; the run script
    also prints how much of the money trades in orders above 5% of daily volume so you can see where the model is stretched.
    Missing liquidity data is treated as 1% participation and 2% daily volatility, never as free."""
    participation = (trade_dollars / adv).replace([np.inf, -np.inf], np.nan)
    participation = participation.where(adv.notna(), 0.01)
    shape = np.sqrt(participation.clip(upper=SQRT_LIMIT)) + 0.5 * (participation - SQRT_LIMIT).clip(lower=0)
    return half_spread_bps(adv) + impact_coef * vol.fillna(0.02) * shape * 1e4


def run_at_aum(prices, volumes, weights, aum, impact_coef=IMPACT_COEF, borrow_bps=0.0, delay=None):
    """Net daily returns of the strategy at account size `aum` (dollars), plus cost statistics.

    Same engine, same delay, same cash yield as every other backtest; only the trading cost differs: each trade pays its
    own size-dependent cost instead of a flat bp. Returns (net_returns, stats_dict)."""
    gross = engine.run_backtest(prices, weights, cost_bps=0.0, borrow_bps=borrow_bps, delay=delay)["ret"]
    d = engine.config.EXEC_DELAY_DAYS if delay is None else delay
    held = weights.reindex(prices.index).fillna(0.0).shift(1 + d).fillna(0.0)
    traded_w = held.diff().abs().fillna(0.0)                              # fraction of the account traded, per ticker
    adv = adv_dollars(prices, volumes).reindex(prices.index)[prices.columns]
    vol = daily_volatility(prices)[prices.columns]
    cost_bp = trade_cost_bps(traded_w * aum, adv, vol, impact_coef)
    cost = (traded_w * cost_bp / 1e4).sum(axis=1)
    net = gross - cost
    dollars = traded_w * aum
    total_traded = dollars.sum().sum()
    participation = (dollars / adv).where(traded_w > 0)
    stats = {"avg_cost_bp": float((traded_w * cost_bp).sum().sum() / traded_w.sum().sum()) if total_traded > 0 else np.nan,
             "p95_participation": float(participation.stack().quantile(0.95)) if total_traded > 0 else np.nan,
             "share_over_5pct": float(dollars.where(participation > 0.05).sum().sum() / total_traded) if total_traded > 0 else np.nan}
    return net, stats


def capacity_table(prices, volumes, weights, aums, start, end, impact_coef=IMPACT_COEF, borrow_bps=0.0):
    """One row per account size: net Sharpe, excess return over cash per year, cost per $ traded, size of trades vs volume.
    Only days from `start` to `end` are measured (weights may be built from earlier history)."""
    rows = []
    for aum in aums:
        net, st = run_at_aum(prices, volumes, weights, aum, impact_coef, borrow_bps)
        net = net.loc[start:end]
        cash = engine.daily_cash(net.index)
        rows.append({"aum": aum, "sharpe": sharpe(net), "excess_pct_yr": 100 * (net - cash).mean() * 252,
                     "avg_cost_bp": st["avg_cost_bp"], "p95_part_pct": 100 * st["p95_participation"],
                     "share_over_5pct": 100 * st["share_over_5pct"]})
    return pd.DataFrame(rows).set_index("aum")


def find_capacity(aums, values, threshold):
    """Account size at which `values` (falling with size) first drops to `threshold`, interpolated on a log scale.
    Returns None if it never gets there, and aums[0] if it is already below at the smallest size."""
    aums, values = np.asarray(aums, float), np.asarray(values, float)
    if np.isnan(values[0]) or values[0] <= threshold:
        return float(aums[0])
    for i in range(1, len(aums)):
        if values[i] <= threshold:
            f = (values[i - 1] - threshold) / (values[i - 1] - values[i])
            return float(10 ** (np.log10(aums[i - 1]) + f * (np.log10(aums[i]) - np.log10(aums[i - 1]))))
    return None
