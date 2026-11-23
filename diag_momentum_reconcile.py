"""DIAGNOSTIC - why does momentum v1 show train Sharpe 1.14 in step 3 but 1.05 in step 3b?

   python diag_momentum_reconcile.py

Runs the SAME settings (252d, top 10, long) through two data paths inside one process and prints the difference.
Train period only, logs nothing, never reads the test period.
"""
import numpy as np
import pandas as pd

import config
import engine
from data import get_cash_rate, get_prices
from engine import run_backtest, trim_and_rebase
from lab.stats import sharpe
from risk import apply_limits
from strategies.momentum.strategy import BENCHMARK, STOCKS, build_weights

engine.set_default_cash_rate(get_cash_rate() if config.USE_CASH_YIELD else None)
prices = get_prices(list(STOCKS) + [BENCHMARK])
stocks_full = prices[[t for t in STOCKS if t in prices.columns]]
print(f"Stocks found: {stocks_full.shape[1]} of {len(STOCKS)}; data runs {stocks_full.index[0].date()} to {stocks_full.index[-1].date()}")
print("Missing:", [t for t in STOCKS if t not in prices.columns] or "none")

LIM = {"max_position": 0.25, "max_gross": 1.0}


def run(px, tag):
    w = apply_limits(build_weights(px, 252, 10, "long"), **LIM)
    res = trim_and_rebase(run_backtest(px, w, cost_bps=5, borrow_bps=25))
    return res["ret"].loc[: config.TRAIN_END]


# Path A: exactly what step 3 does (data runs past the train end; trimmed afterwards)
a = run(stocks_full, "A")
# Path B: what step 3b does (data cut at the train end BEFORE building weights)
b = run(stocks_full.loc[: config.TRAIN_END], "B")
common = a.index.intersection(b.index)
print(f"\nA (step 3 path)  : {len(a)} days from {a.index[0].date()}, Sharpe {sharpe(a):.3f}")
print(f"B (step 3b path) : {len(b)} days from {b.index[0].date()}, Sharpe {sharpe(b):.3f}")
d = (a.loc[common] - b.loc[common]).abs()
print(f"Largest daily return difference between A and B: {d.max():.5f} on {d.idxmax().date() if d.max() > 0 else 'n/a'}; days that differ: {(d > 1e-12).sum()}")

# Is the 1.14 even reproducible from the trial log's own data?
try:
    log = pd.read_csv(config.EXPERIMENTS_DIR / "trials.csv") if hasattr(config, "EXPERIMENTS_DIR") else pd.read_csv("experiments/trials.csv")
    row = log[(log.strategy == "momentum") & (log.params == "lookback=252, k=10, mode=long")].iloc[-1]
    print(f"\nTrial log says: Sharpe {row.train_sharpe:.3f}, {row.n_obs} days, data ended {row.data_end}")
    print(f"Your cached data ends {stocks_full.index[-1].date()}. If those dates differ, the prices were re-downloaded since step 3 ran "
          "(adjusted prices shift slightly as dividends are paid), which can move a Sharpe by a few hundredths.")
except Exception as e:
    print("Could not read the trial log:", e)

# Per-stock spot check: do cached prices look like they changed? (first/last price of three names)
print("\nSpot check of cached prices (compare with an old run if you have one):")
print(stocks_full[["AAPL", "MSFT", "JPM"]].dropna().iloc[[0, -1]].round(3).to_string())
