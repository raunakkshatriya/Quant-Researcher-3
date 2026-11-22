"""STEP 6 - Strategy: FX carry + momentum on 9 currencies vs the US dollar.

   python step6_fx_carry.py            # train period only (locked test)
   python step6_fx_carry.py --test     # one look at the test period

Read strategies/fx_carry/description.txt first. The first thing printed is a data report: read it.
"""
from functools import partial

import pandas as pd

from lab.experiment import run_experiment
from strategies.fx_carry.data_fx import load_fx
from strategies.fx_carry.strategy import build_weights

VERSION = "v1"
COST_BPS = 3          # spread + slippage per $ traded (majors ~1 bp, NOK/SEK/NZD wider)

spot, carry, tri, report = load_fx()
print("\nDATA REPORT (what we actually got):")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(report.to_string(float_format=lambda x: f"{x:.2f}"))
print(f"US rate series last observation: {report.attrs['usd_last_obs']}")
for code, r in report.iterrows():
    if r["status"] == "ok" and not r["carry_known_on_last_day"]:
        print(f"  DATA WARNING: {code} has no usable interest rate on the last day (series stale): it is left out of the ranking")
print(f"\nExchange-rate table: {spot.index[0].date()} to {spot.index[-1].date()}, {spot.shape[1]} currencies")

run_experiment(
    name="fx_carry", version=VERSION,
    prices=tri, build_weights=partial(build_weights, carry=carry, spot=spot),
    grid={"w_carry": [0.0, 0.5, 1.0], "lookback": [126, 252]},
    cost_bps=COST_BPS, borrow_bps=0.0,
    limits={"max_position": 0.25, "max_gross": 1.0},
    benchmark_name="cash",
)
