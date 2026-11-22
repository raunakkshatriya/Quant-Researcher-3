"""STEP 3 - Strategy: cross-sectional momentum on 50 stocks.   Run with:  python step3_momentum.py

Read strategies/momentum/description.txt first.
First run downloads 50 stocks one by one (a few minutes). After that it is instant.
"""
import itertools

import pandas as pd

import config
from data import get_prices
from engine import cost_sensitivity, metrics, run_backtest, split_report, trim_and_rebase
from plots import plot_equity
from strategies.momentum.strategy import BENCHMARK, STOCKS, build_weights

COST_BPS = 5      # per $1 traded: spread + slippage on large US stocks (commission is zero)
BORROW_BPS = 25   # yearly fee on shorted stocks (0.25% a year)

prices = get_prices(list(STOCKS) + [BENCHMARK])
stocks = prices[[t for t in STOCKS if t in prices.columns]]
print(f"{stocks.shape[1]} stocks, data {stocks.index[0].date()} to {stocks.index[-1].date()}")
print("WARNING: these are TODAY's 50 biggest companies. Looking back, we only 'chose' firms that")
print("later became huge, so the TRAIN results are flattered (survivorship bias). Trust the")
print("comparison between settings and the TEST period more than the train totals.\n")

LOOKBACKS = [126, 189, 252]
KS = [5, 10]
MODES = ["long", "long_short"]
rows = []
for lookback, k, mode in itertools.product(LOOKBACKS, KS, MODES):
    w = build_weights(stocks, lookback, k, mode)
    res = trim_and_rebase(run_backtest(stocks, w, COST_BPS, BORROW_BPS))
    m = metrics(res.loc[:config.TRAIN_END, "ret"])
    rows.append({"lookback": lookback, "k": k, "mode": mode,
                 "train_sharpe": m["sharpe"], "train_return": m["total_return"]})
grid = pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)
print("Settings ranked by TRAIN Sharpe (test results hidden on purpose):")
print(grid.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

best = grid.iloc[0]
lookback, k, mode = int(best["lookback"]), int(best["k"]), best["mode"]
print(f"\nChosen on train data: lookback={lookback}, k={k}, mode={mode}")

weights = build_weights(stocks, lookback, k, mode)
result = trim_and_rebase(run_backtest(stocks, weights, COST_BPS, BORROW_BPS))
spy = run_backtest(prices[[BENCHMARK]], prices[[BENCHMARK]] * 0 + 1.0).loc[result.index[0]:]

split_report(result, f"Momentum ({mode}, lookback={lookback}, k={k}, {COST_BPS} bp cost, {BORROW_BPS} bp borrow)")
split_report(spy, "Benchmark: buy and hold SPY over the same dates")
cost_sensitivity(stocks, weights, borrow_bps=BORROW_BPS)

latest = weights.iloc[-1]
print("\nCurrent LONG :", {t: round(w, 3) for t, w in latest.items() if w > 0})
print("Current SHORT:", {t: round(w, 3) for t, w in latest.items() if w < 0})

plot_equity({"Momentum": result, "SPY buy & hold": spy},
            f"Momentum {mode} (lookback={lookback}, k={k}) vs SPY", "step3_momentum.png")
result.to_csv("output/momentum_equity.csv")
