"""STEP 1 - Our first backtests, and the look-ahead trap.
(These demos use delay=0 on purpose, to keep the idea simple. Real strategies use delay=1.)

Run with:   python step1_buy_and_hold.py

Part A: buy-and-hold Apple. Weight = 1.0 every day. The simplest "strategy" there is.
Part B: a CHEATING strategy that looks at tomorrow's price. It looks amazing and is worthless.
Part C: the same idea done honestly. The magic disappears.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw to a file, no window needed
import matplotlib.pyplot as plt

import config
from data import get_prices
from engine import run_backtest, split_report

prices = get_prices("AAPL")
print(f"AAPL: {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")

# ---- Part A: buy and hold -------------------------------------------------
weights_hold = prices * 0 + 1.0                  # a table full of 1.0
hold = run_backtest(prices, weights_hold, delay=0)
split_report(hold, "A) Buy and hold AAPL")

# ---- Part B: the cheat ----------------------------------------------------
# Rule: "hold Apple on days when Apple goes up". At the end of today we set tomorrow's
# weight by peeking at TOMORROW'S return (shift(-1) pulls tomorrow's number back to today).
# A real trader can never know that. The engine's safety shift is still there; the cheat
# is in the INFORMATION we feed it.
tomorrow_up = (prices.pct_change().shift(-1) > 0).astype(float)
cheat = run_backtest(prices, tomorrow_up, delay=0)
split_report(cheat, "B) CHEAT: hold only on days Apple will rise (uses the future!)")

# ---- Part C: the honest version -------------------------------------------
# Rule: "hold Apple tomorrow if it went up TODAY". Only uses information we really had.
today_up = (prices.pct_change() > 0).astype(float)
honest = run_backtest(prices, today_up, cost_bps=5, delay=0)
split_report(honest, "C) Honest: hold tomorrow if Apple rose today (5 bp cost)")

# ---- Picture ----------------------------------------------------------------
Path("output").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hold.index, hold["equity"], label="A) Buy and hold")
ax.plot(honest.index, honest["equity"], label="C) Honest rule")
ax.set_yscale("log")                              # log scale: equal % moves look equal
ax.axvline(config.TEST_START, color="grey", linestyle="--")
ax.text(config.TEST_START, ax.get_ylim()[1], " test starts", va="top", color="grey")
ax.set_title("Growth of $100 (log scale). Cheat B is left off: it grows too large to share a chart.")
ax.legend()
fig.savefig("output/step1_equity.png", dpi=120, bbox_inches="tight")
print("\nSaved picture to output/step1_equity.png")
