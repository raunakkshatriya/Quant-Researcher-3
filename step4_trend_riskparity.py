"""STEP 4 - Strategy: cross-asset trend following with inverse-volatility sizing.

   python step4_trend_riskparity.py            # train period only (locked test)
   python step4_trend_riskparity.py --test     # one look at the test period

Read strategies/trend_riskparity/description.txt first.
"""
from data import get_prices
from lab.experiment import run_experiment
from strategies.trend_riskparity.strategy import ASSETS, BENCHMARK_WEIGHTS, build_weights

VERSION = "v1"
COST_BPS = 3

prices = get_prices(list(ASSETS))
run_experiment(
    name="trend_riskparity", version=VERSION,
    prices=prices, build_weights=build_weights,
    grid={"lookback": [126, 189, 252], "vol_target": [0.06, 0.08, 0.10]},
    cost_bps=COST_BPS,
    benchmark_prices=prices[list(BENCHMARK_WEIGHTS)], benchmark_weights=BENCHMARK_WEIGHTS,
    benchmark_name="60/40 SPY/IEF",
    limits={"max_position": 1.0, "max_gross": 1.0},
)
