"""STEP 2 - Strategy: sector rotation.   Run with:  python step2_sector_rotation.py

Read strategies/sector_rotation/description.txt first - it explains the idea and the rules.

What this script does:
  1. Downloads the 11 sector ETFs and SPY (S&P 500) prices.
  2. Tries 12 settings (lookback x k) and scores each ON THE TRAIN PERIOD ONLY.
  3. Picks the setting with the best train Sharpe ratio and freezes it.
  4. Only then shows how that frozen setting did in the TEST period.
"""
import itertools

import pandas as pd

import config
from data import get_prices
from engine import metrics, run_backtest, split_report, trim_and_rebase
from plots import plot_equity
from strategies.sector_rotation.strategy import BENCHMARK, SECTOR_ETFS, build_weights

COST_BPS = 3   # cost of trading these very liquid ETFs, per $1 traded

prices = get_prices(SECTOR_ETFS + [BENCHMARK])
sectors = prices[SECTOR_ETFS]
print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}\n")

# ---- 2. try the settings, score on TRAIN only --------------------------------------
LOOKBACKS = [63, 126, 189, 252]   # about 3, 6, 9, 12 months of trading days
KS = [2, 3, 4]                    # hold the top 2, 3 or 4 sectors
rows = []
for lookback, k in itertools.product(LOOKBACKS, KS):
    res = trim_and_rebase(run_backtest(sectors, build_weights(sectors, lookback, k), COST_BPS))
    m = metrics(res.loc[:config.TRAIN_END, "ret"])
    rows.append({"lookback": lookback, "k": k, "train_sharpe": m["sharpe"], "train_return": m["total_return"]})
grid = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)
print("Settings ranked by TRAIN Sharpe (test results are NOT shown here on purpose):")
print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# ---- 3. freeze the best setting ------------------------------------------------------
best = grid.iloc[0]
lookback, k = int(best["lookback"]), int(best["k"])
print(f"\nChosen on train data: lookback={lookback} days, hold top k={k} sectors")

# ---- 4. final run + benchmark ----------------------------------------------------------
weights = build_weights(sectors, lookback, k)
result = trim_and_rebase(run_backtest(sectors, weights, COST_BPS))
spy = run_backtest(prices[[BENCHMARK]], prices[[BENCHMARK]] * 0 + 1.0).loc[result.index[0]:]

split_report(result, f"Sector rotation (lookback={lookback}, k={k}, {COST_BPS} bp cost)")
split_report(spy, "Benchmark: buy and hold SPY over the same dates")

latest = weights.iloc[-1]
print("\nCurrent holdings:", {t: round(w, 2) for t, w in latest.items() if w > 0})

plot_equity({"Sector rotation": result, "SPY buy & hold": spy},
            f"Sector rotation vs SPY (lookback={lookback}, k={k})", "step2_sector_rotation.png")
result.to_csv("output/sector_rotation_equity.csv")
