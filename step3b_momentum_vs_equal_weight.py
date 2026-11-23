"""STEP 3b - Re-check momentum against the right yardstick: equal-weight buy & hold of the SAME stocks.

   python step3b_momentum_vs_equal_weight.py

Step 3 judged momentum against SPY. But the 50 stocks are TODAY's biggest companies, and simply owning all of them
equally had a Sharpe about double SPY's (survivorship bias, see step 7). So the fair question is:
does picking the top-10 by momentum add anything beyond owning the whole list?

TRAIN period only. Fixed v1 settings, nothing is searched, nothing is logged to the trial log, the test period is never read.
The grid table at the end is descriptive: it is NOT a new selection step.
"""
import numpy as np
import pandas as pd

import config
import engine
from data import get_cash_rate, get_prices
from engine import metrics, run_backtest, trim_and_rebase
from lab.stats import alpha_beta, sharpe, versus_benchmark
from risk import apply_limits
from strategies.momentum.strategy import BENCHMARK, STOCKS, build_weights
from strategies.sma.strategy import equal_weight_index

LOOKBACK, K, MODE = 252, 10, "long"          # momentum v1 (the settings chosen in step 3)
GRID = [(lb, k) for lb in (126, 252, 378) for k in (5, 10, 15, 20)]   # same 12 settings as step 3's grid
COST_BPS, BORROW_BPS = 5, 25
LIMITS = {"max_position": 0.25, "max_gross": 1.0}

engine.set_default_cash_rate(get_cash_rate() if config.USE_CASH_YIELD else None)
tickers = list(STOCKS)
prices = get_prices(tickers + [BENCHMARK])
stocks = prices[[t for t in tickers if t in prices.columns]].loc[: config.TRAIN_END]
spy_px = prices[[BENCHMARK]].loc[: config.TRAIN_END]


def momentum_returns(lookback, k):
    w = apply_limits(build_weights(stocks, lookback, k, MODE), **LIMITS)
    res = trim_and_rebase(run_backtest(stocks, w, cost_bps=COST_BPS, borrow_bps=BORROW_BPS, delay=config.EXEC_DELAY_DAYS))
    return res["ret"]



mom = momentum_returns(LOOKBACK, K)
start = mom.index[0]
ew = equal_weight_index(stocks).pct_change().loc[start:]   # same definition as step 7 (stocks without history are ignored); index level -> daily returns
spy = spy_px[BENCHMARK].pct_change().loc[start:]
idx = mom.index.intersection(ew.index).intersection(spy.dropna().index)
mom, ew, spy = mom.loc[idx], ew.loc[idx], spy.loc[idx]

print(f"Train period {idx[0].date()} to {idx[-1].date()}  ({len(idx) / 252:.1f} years)")
print("WARNING: all 50 stocks are TODAY's biggest companies, so BOTH momentum and the equal-weight list are flattered by survivorship.\n")

print("=== Side by side (train) ===")
rows = {}
for name, r in (("Momentum v1 (252d, top 10)", mom), ("Equal-weight 50 stocks", ew), ("SPY buy & hold", spy)):
    m = metrics(r)
    rows[name] = {"Sharpe": m["sharpe"], "CAGR": m["cagr"], "Volatility": m["volatility"], "Worst drawdown": m["max_drawdown"]}
tbl = pd.DataFrame(rows).T
tbl["Sharpe"] = tbl["Sharpe"].map("{:.2f}".format)
for c in ("CAGR", "Volatility", "Worst drawdown"):
    tbl[c] = tbl[c].map("{:.1%}".format)
print(tbl.to_string())

print("\n=== Does momentum beat each yardstick? ===")
for label, bench in (("equal-weight 50", ew), ("SPY", spy)):
    v = versus_benchmark(mom, bench)
    a = alpha_beta(mom, bench)
    if v and a:
        print(f"vs {label}: active return {v['active_return']:+.1%}/yr, tracking error {v['tracking_error']:.1%}, "
              f"info ratio {v['info_ratio']:.2f}, t {v['t_stat']:.2f}")
        print(f"   regression: alpha {a['alpha']:+.1%}/yr (t {a['t_alpha']:.2f}), beta {a['beta']:.2f}, R-squared {a['r_squared']:.2f}")

print("\n=== Sharpe by 5-year block ===")
blocks = {}
for lo in range(idx[0].year, idx[-1].year + 1, 5):
    sl = idx[(idx.year >= lo) & (idx.year < lo + 5)]
    if len(sl) < 250:
        continue
    blocks[f"{lo}-{min(lo + 4, idx[-1].year)}"] = {"Momentum": sharpe(mom.loc[sl]), "Equal-weight": sharpe(ew.loc[sl]), "SPY": sharpe(spy.loc[sl])}
print(pd.DataFrame(blocks).T.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n=== All 12 step-3 settings vs equal-weight (descriptive only; train Sharpe) ===")
ew_sharpe = sharpe(ew)
grid_rows = []
for lb, k in GRID:
    r = momentum_returns(lb, k).reindex(ew.index).dropna()
    a = alpha_beta(r, ew)
    grid_rows.append({"lookback": lb, "top k": k, "Sharpe": sharpe(r), "alpha vs EW": a.get("alpha", np.nan), "t(alpha)": a.get("t_alpha", np.nan)})
g = pd.DataFrame(grid_rows)
print(g.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nEqual-weight Sharpe for reference: {ew_sharpe:.2f}.  Settings with Sharpe above it: {(g['Sharpe'] > ew_sharpe).sum()} of {len(g)}; "
      f"with |t(alpha)| >= 2: {(g['t(alpha)'].abs() >= 2).sum()} of {len(g)}.")
print("Reading it: if alpha is small and |t| < 2, most of momentum's Sharpe is just 'own these big winners', not stock-picking skill.")
