"""STEP 2 - Strategy: sector rotation.

   python step2_sector_rotation.py            # train period only (locked test)
   python step2_sector_rotation.py --test     # one look at the test period

Read strategies/sector_rotation/description.txt first.
"""
from data import get_prices
from lab.experiment import run_experiment
from strategies.sector_rotation.strategy import BENCHMARK, SECTOR_ETFS, build_weights

VERSION = "v1"     # bump to "v2" when you change the strategy's LOGIC in strategy.py
COST_BPS = 3

prices = get_prices(SECTOR_ETFS + [BENCHMARK])
run_experiment(
    name="sector_rotation", version=VERSION,
    prices=prices[SECTOR_ETFS], build_weights=build_weights,
    grid={"lookback": [63, 126, 189, 252], "k": [2, 3, 4]},
    cost_bps=COST_BPS, benchmark_prices=prices[[BENCHMARK]],
)
