"""STEP 5 - Strategy: pairs trading (statistical arbitrage), on two universes, then compared.

   python step5_pairs.py            # train period only (locked test)
   python step5_pairs.py --test     # one look at the test period for BOTH universes

Read strategies/pairs/description.txt first.
Universe 1: 49 large US stocks in 9 sector groups.   Universe 2: the 11 SPDR sector ETFs.
"""
from functools import partial

from data import get_prices
from engine import run_backtest, trim_and_rebase
from lab.experiment import run_experiment, wants_test
from strategies.pairs.analysis import print_comparison
from strategies.pairs.strategy import ETF_GROUPS, STOCK_GROUPS, formation_summary, build_weights

VERSION = "v1"
GRID = {"entry_z": [1.5, 2.0, 2.5], "exit_z": [0.0, 0.5]}
LIMITS = {"max_position": 0.25, "max_gross": 1.0}
BORROW_BPS = 25
# name, groups, cost per $ traded (in bp). Stocks cost more to trade than big liquid ETFs.
UNIVERSES = [("pairs_stocks", STOCK_GROUPS, 5), ("pairs_etfs", ETF_GROUPS, 3)]

all_prices = get_prices(sorted({t for _, g, _ in UNIVERSES for ts in g.values() for t in ts} | {"SPY"}))
rets, summaries = {}, {}
for name, groups, cost in UNIVERSES:
    tickers = [t for ts in groups.values() for t in ts]
    prices = all_prices[tickers]
    out = run_experiment(name=name, version=VERSION, prices=prices,
                         build_weights=partial(build_weights, groups=groups), grid=GRID,
                         cost_bps=cost, borrow_bps=BORROW_BPS, limits=LIMITS, benchmark_name="cash")
    res = trim_and_rebase(run_backtest(prices, out["weights"], cost, BORROW_BPS))   # weights already risk-limited
    rets[name.replace("pairs_", "")] = res["ret"]
    summaries[name.replace("pairs_", "")] = formation_summary(prices, groups)

print_comparison(rets, summaries, all_prices["SPY"], include_test=wants_test())
