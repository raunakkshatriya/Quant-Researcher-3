"""STEP 7 - Strategy: SMA trend filter on 49 stocks (the baseline).

   python step7_sma.py            # train period only (locked test)
   python step7_sma.py --test     # one look at the test period

Read strategies/sma/description.txt first.
"""
import pandas as pd

import config
from data import get_prices
from engine import metrics, run_backtest
from lab.experiment import run_experiment
from strategies.sma.strategy import RULES, TICKERS, build_weights, equal_weight_index

VERSION = "v1"
COST_BPS = 5       # spread + slippage per $ traded (commission is zero)

prices = get_prices(TICKERS + ["SPY"])
stocks = prices[[t for t in TICKERS if t in prices.columns]]
print("WARNING: the 49 stocks are TODAY's biggest companies -> survivorship bias flatters buy-and-hold AND the SMA rule.")
bench = pd.DataFrame({"EqualWeight49": equal_weight_index(stocks)})

out = run_experiment(
    name="sma", version=VERSION, prices=stocks, build_weights=build_weights,
    grid={"rule": RULES}, cost_bps=COST_BPS, borrow_bps=0.0,
    limits={"max_position": 0.25, "max_gross": 1.0},
    benchmark_prices=bench, benchmark_name="equal-weight buy & hold",
)

# ---- extra, train period only: how the rule behaves (nothing here changes any setting) ----
w = out["weights"].loc[: config.TRAIN_END]
held = w.gt(0)
exposure = w.sum(axis=1)
years = len(w) / 252
flips = held.astype(int).diff().abs().sum().sum() / 2          # entries ~ exits, so /2 = round trips
print(f"\nHow the best rule behaves (train): average invested {exposure[exposure.index >= exposure.ne(0).idxmax()].mean():.0%} of money, "
      f"rest in cash; {flips / held.shape[1] / years:.1f} round trips per stock per year (more = more whipsaw and more cost)")
spy = run_backtest(prices[["SPY"]], pd.DataFrame({"SPY": 1.0}, index=prices.index), delay=config.EXEC_DELAY_DAYS)["ret"]
spy = spy.loc[exposure.ne(0).idxmax(): config.TRAIN_END]
print(f"For context, SPY buy & hold over the same train dates: Sharpe {metrics(spy)['sharpe']:.2f}, "
      f"worst drawdown {metrics(spy)['max_drawdown']:.1%}")
