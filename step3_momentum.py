"""STEP 3 - Strategy: cross-sectional momentum on 50 stocks.

   python step3_momentum.py            # train period only (locked test)
   python step3_momentum.py --test     # one look at the test period

Read strategies/momentum/description.txt first. First run downloads 50 stocks (few minutes).
"""
from data import get_prices
from lab.experiment import run_experiment
from strategies.momentum.strategy import BENCHMARK, STOCKS, build_weights

VERSION = "v1"
COST_BPS = 5       # spread + slippage per $1 traded (commission is zero)
BORROW_BPS = 25    # yearly fee on shorts

prices = get_prices(list(STOCKS) + [BENCHMARK])
stocks = prices[[t for t in STOCKS if t in prices.columns]]
print("WARNING: the 50 stocks are TODAY's biggest companies -> survivorship bias flatters TRAIN results.")
run_experiment(
    name="momentum", version=VERSION,
    prices=stocks, build_weights=build_weights,
    grid={"lookback": [126, 189, 252], "k": [5, 10], "mode": ["long", "long_short"]},
    cost_bps=COST_BPS, borrow_bps=BORROW_BPS, benchmark_prices=prices[[BENCHMARK]],
    limits={"max_position": 0.25, "max_gross": 1.0},
)
