"""STEP 9 - Re-score sector rotation and trend on the CURRENT Sharpe scale, against fair benchmarks.

   python step9_rescore_on_current_scale.py

Why: early results were logged before cash started earning T-bill interest and before Sharpe was measured over cash
(step 3b reconciliation: momentum 1.14 old scale -> 1.05 current). This re-runs the FIXED v1 settings and the whole
grids, prints BOTH scales side by side, and compares each strategy with its fair yardstick:
   sector rotation -> equal-weight of the 11 sector ETFs (and SPY)      trend -> 60/40 SPY/IEF
TRAIN period only. Nothing is searched, nothing is logged to the trial log, the test period is never read.
"""
import numpy as np
import pandas as pd

import config
import engine
from data import get_cash_rate, get_prices
from engine import metrics, run_backtest, trim_and_rebase
from lab.stats import alpha_beta, sharpe, versus_benchmark
from risk import apply_limits
from strategies.sector_rotation.strategy import BENCHMARK as SPY, SECTOR_ETFS, build_weights as sector
from strategies.trend_riskparity.strategy import ASSETS, BENCHMARK_WEIGHTS, build_weights as trend

CASH = get_cash_rate() if config.USE_CASH_YIELD else None
END = config.TRAIN_END


def run(px, w, cost_bps, cash):
    """Daily returns of the strategy, trimmed to the first day it holds something, train period only."""
    engine.set_default_cash_rate(cash)
    return trim_and_rebase(run_backtest(px, w, cost_bps=cost_bps))["ret"].loc[:END]


def both_scales(px, w, cost_bps):
    """(current-scale Sharpe, old-scale Sharpe, current-scale returns). Old = cash earns 0%, Sharpe with risk-free 0."""
    new = run(px, w, cost_bps, CASH)
    engine.set_default_cash_rate(CASH)
    s_new = sharpe(new)
    old = run(px, w, cost_bps, None)
    s_old = old.mean() / old.std() * np.sqrt(252)
    engine.set_default_cash_rate(CASH)
    return s_new, s_old, new


def fixed_benchmark(px, weights, start):
    """Constant-weight benchmark, rebalanced daily, same engine and delay, restricted to dates from `start`."""
    engine.set_default_cash_rate(CASH)
    w = pd.DataFrame({c: wt for c, wt in weights.items()}, index=px.index)
    return run_backtest(px[list(weights)].dropna(), w.loc[px[list(weights)].dropna().index], delay=config.EXEC_DELAY_DAYS)["ret"].loc[start:END]


def report(name, strat, benches):
    print(f"\n----- {name}: train {strat.index[0].date()} to {strat.index[-1].date()} ({len(strat) / 252:.1f} years) -----")
    rows = {}
    for label, r in [(name, strat)] + list(benches.items()):
        r = r.reindex(strat.index).dropna()
        m = metrics(r)
        rows[label] = {"Sharpe": f"{m['sharpe']:.2f}", "CAGR": f"{m['cagr']:.1%}", "Volatility": f"{m['volatility']:.1%}",
                       "Worst drawdown": f"{m['max_drawdown']:.1%}"}
    print(pd.DataFrame(rows).T.to_string())
    for label, b in benches.items():
        b = b.reindex(strat.index).dropna()
        v, a = versus_benchmark(strat, b), alpha_beta(strat, b)
        if v and a:
            print(f"vs {label}: active {v['active_return']:+.1%}/yr, info ratio {v['info_ratio']:.2f}, t {v['t_stat']:.2f} | "
                  f"alpha {a['alpha']:+.1%}/yr (t {a['t_alpha']:.2f}), beta {a['beta']:.2f}, R-squared {a['r_squared']:.2f}")


# ================= sector rotation v1: lookback 252, top 4 =================
px = get_prices(SECTOR_ETFS + [SPY]).loc[:END]
sec_px = px[SECTOR_ETFS]
grid_rows, results = [], {}
for lb in (63, 126, 189, 252):
    for k in (2, 3, 4):
        s_new, s_old, r = both_scales(sec_px, sector(sec_px, lb, k), 3)
        grid_rows.append({"lookback": lb, "top k": k, "Sharpe (current)": s_new, "Sharpe (old scale)": s_old})
        results[(lb, k)] = r
v1 = results[(252, 4)]
ew_sec = fixed_benchmark(sec_px, {t: 1.0 / len(SECTOR_ETFS) for t in SECTOR_ETFS}, v1.index[0])
spy = fixed_benchmark(px, {SPY: 1.0}, v1.index[0])
report("Sector rotation v1 (252d, top 4)", v1, {"equal-weight 11 sectors": ew_sec, "SPY": spy})
print("\nWhole grid (descriptive only; not a new selection):")
print(pd.DataFrame(grid_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"Equal-weight sectors Sharpe on the same dates: {sharpe(ew_sec.reindex(v1.index).dropna()):.2f}")

# ================= trend v1: lookback 126, vol target 6% =================
tpx = get_prices(list(ASSETS)).loc[:END]
grid_rows, results = [], {}
for lb in (126, 189, 252):
    for vt in (0.06, 0.08, 0.10):
        w = apply_limits(trend(tpx, lb, vt), max_position=1.0, max_gross=1.0)
        s_new, s_old, r = both_scales(tpx, w, 3)
        grid_rows.append({"lookback": lb, "vol target": vt, "Sharpe (current)": s_new, "Sharpe (old scale)": s_old})
        results[(lb, vt)] = r
t1 = results[(126, 0.06)]
sixty40 = fixed_benchmark(tpx, BENCHMARK_WEIGHTS, t1.index[0])
report("Trend v1 (126d, vol target 6%)", t1, {"60/40 SPY/IEF": sixty40})
print("\nWhole grid (descriptive only; not a new selection):")
print(pd.DataFrame(grid_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"60/40 Sharpe on the same dates: {sharpe(sixty40.reindex(t1.index).dropna()):.2f}")
print("\nReading it: a strategy only earns its place if it beats its FAIR yardstick, not just SPY. "
      "Check alpha vs that yardstick and whether |t| >= 2; the old-scale column is only there to compare with earlier notes.")
