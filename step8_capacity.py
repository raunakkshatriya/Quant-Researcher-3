"""STEP 8 - Execution research, Stage A: how much money can each strategy handle?

   python step8_capacity.py              # last 10 years of the TRAIN period
   python step8_capacity.py --years 5    # a shorter / more recent window

Re-runs the FIXED v1 settings of four equity strategies at account sizes from $1k to $10bn, charging each trade a
size-dependent cost (half-spread + square-root market impact, see lab/capacity.py) instead of a flat 3-5 bp.
It never touches the test period, does not search settings and logs nothing to the trial log.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import engine
from data import get_cash_rate, get_prices, get_volumes
from lab.capacity import capacity_table, find_capacity
from lab.stats import sharpe
from risk import apply_limits
from strategies.momentum.strategy import STOCKS, build_weights as momentum
from strategies.sector_rotation.strategy import SECTOR_ETFS, build_weights as sector
from strategies.sma.strategy import TICKERS, build_weights as sma
from strategies.trend_riskparity.strategy import ASSETS, build_weights as trend

YEARS = int(sys.argv[sys.argv.index("--years") + 1]) if "--years" in sys.argv else 10
END = pd.Timestamp(config.TRAIN_END)
START = END - pd.DateOffset(years=YEARS) + pd.Timedelta(days=1)
FINE_AUMS = np.logspace(3, 10, 29)                       # $1k ... $10bn, for finding capacity
SHOWN_AUMS = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
IMPACT_VARIANTS = [0.5, 1.0, 2.0]                        # the impact coefficient Y is the least certain input
FLAT_BPS = {"momentum": 5, "sma": 5, "sector rotation": 3, "trend": 3}   # what the earlier steps charged


def where(x):
    return "not within the $1k-$10bn range tested" if x is None else f"at about {usd(x)}"


def usd(x):
    if x is None:
        return ">$10bn"
    for unit, size in (("bn", 1e9), ("m", 1e6), ("k", 1e3)):
        if x >= size:
            return f"${x / size:.3g}{unit}"
    return f"${x:.0f}"


# ---- the four strategies, with the settings that won on TRAIN in steps 2-4 and 7 (fixed here) ----
engine.set_default_cash_rate(get_cash_rate() if config.USE_CASH_YIELD else None)
tickers = sorted(set(STOCKS) | set(TICKERS) | set(SECTOR_ETFS) | set(ASSETS))
prices_all = get_prices(tickers)
volumes_all = get_volumes(tickers)


def pick(cols):
    cols = [c for c in cols if c in prices_all.columns]
    return prices_all[cols], volumes_all.reindex(columns=cols)


strategies = {}
px, vol = pick(STOCKS)
strategies["momentum"] = (px, vol, apply_limits(momentum(px, lookback=252, k=10, mode="long"), max_position=0.25, max_gross=1.0))
px, vol = pick(TICKERS)
strategies["sma"] = (px, vol, apply_limits(sma(px, rule="price>SMA200"), max_position=0.25, max_gross=1.0))
px, vol = pick(SECTOR_ETFS)
strategies["sector rotation"] = (px, vol, apply_limits(sector(px, lookback=252, k=4), max_gross=1.0))
px, vol = pick(ASSETS)
strategies["trend"] = (px, vol, apply_limits(trend(px, lookback=126, vol_target=0.06), max_position=1.0, max_gross=1.0))

print(f"CAPACITY STUDY (train period only): measured over {START.date()} to {END.date()}, fixed v1 settings.")
print("Cost per trade = half-spread (by liquidity) + Y x daily volatility x sqrt(order / average daily dollar volume), Y = 1.0.\n")

summary, curves = [], {}
for name, (px, vol, w) in strategies.items():
    table = capacity_table(px, vol, w, FINE_AUMS, START, END)
    curves[name] = table["sharpe"]
    gross = sharpe(engine.run_backtest(px, w, cost_bps=0.0)["ret"].loc[START:END])
    flat = sharpe(engine.run_backtest(px, w, cost_bps=FLAT_BPS[name])["ret"].loc[START:END])
    shown = capacity_table(px, vol, w, SHOWN_AUMS, START, END)
    shown.index = [usd(a) for a in shown.index]
    shown = shown.rename(columns={"sharpe": "net Sharpe", "excess_pct_yr": "excess %/yr", "avg_cost_bp": "cost bp/$",
                                  "p95_part_pct": "p95 order % of ADV", "share_over_5pct": "% $ in orders >5% ADV"})
    print(f"=== {name} ===   Sharpe with zero costs {gross:.2f};  with the flat {FLAT_BPS[name]} bp of earlier steps {flat:.2f}")
    print(shown.to_string(float_format=lambda v: f"{v:.2f}"))
    if gross > 0.1:
        half = find_capacity(FINE_AUMS, table["sharpe"].to_numpy(), gross / 2)
        gone = find_capacity(FINE_AUMS, table["excess_pct_yr"].to_numpy(), 0.0)
        by_y = {y: find_capacity(FINE_AUMS, capacity_table(px, vol, w, FINE_AUMS, START, END, impact_coef=y)["sharpe"].to_numpy(),
                                 gross / 2) for y in IMPACT_VARIANTS}
        print(f"Sharpe falls to half its zero-cost value: {where(half)}.  Edge over cash gone: {where(gone)}.")
        print("  how much the answer depends on the impact coefficient Y (Sharpe halves at): "
              + ",  ".join(f"Y={y}: {usd(c)}" for y, c in by_y.items()) + "\n")
    else:
        half = gone = None
        print("No edge to protect in this window (zero-cost Sharpe <= 0.1), so capacity is not meaningful.\n")
    at = lambda a: float(np.interp(np.log10(a), np.log10(FINE_AUMS), table["sharpe"].to_numpy()))
    summary.append({"strategy": name, "Sharpe 0 cost": gross, "Sharpe flat bp": flat, "Sharpe @ $100k": at(1e5),
                    "Sharpe @ $10m": at(1e7), "Sharpe @ $1bn": at(1e9),
                    "Sharpe halves at": usd(half) if gross > 0.1 else "n/a", "edge gone at": usd(gone) if gross > 0.1 else "n/a"})

print("SUMMARY (Sharpe = after size-dependent costs)")
print(pd.DataFrame(summary).set_index("strategy").to_string(float_format=lambda v: f"{v:.2f}"))

# ---- chart: net Sharpe against account size, four lines with different markers and end labels ----
Path("output").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5.2))
style = {"momentum": ("#0072B2", "o"), "sma": ("#D55E00", "s"), "sector rotation": ("#009E73", "^"), "trend": ("#CC79A7", "D")}
for name, series in curves.items():
    color, marker = style[name]
    ax.plot(FINE_AUMS, series.to_numpy(), color=color, marker=marker, markersize=4, markevery=4, linewidth=1.8, label=name)
    ax.annotate(name, (FINE_AUMS[-1], series.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlim(1e3, 6e10)
ax.set_xlabel("account size ($, log scale)")
ax.set_ylabel("Sharpe after costs")
ax.set_title(f"Net Sharpe vs account size, {START.year}-{END.year} (train period, fixed v1 settings)")
ax.axhline(0, color="#888888", linewidth=0.8)
ax.grid(True, color="#dddddd", linewidth=0.6)
ax.legend(loc="lower left", frameon=False)
fig.savefig("output/capacity.png", dpi=120, bbox_inches="tight")
print("\nSaved picture to output/capacity.png")
print("Reminder: Y and the spread tiers are published rules of thumb, not our measured fills. Treat the numbers as orders of magnitude.")
