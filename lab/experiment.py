"""The experiment runner every strategy uses.

   run_experiment(...)  tries a grid of settings, keeps a lab notebook, measures luck, and
                        keeps the TEST period locked until you ask for it.

Typical session:
   python step3_momentum.py            # train period only: iterate, change grid, try ideas
   python step3_momentum.py --test     # when you believe in it: ONE look at the test period
"""
import itertools
import sys

import pandas as pd

import config
from data import validate_prices
from engine import cost_sensitivity, metrics, run_backtest, split_report, trim_and_rebase
from lab import trials
from lab.stats import deflated_sharpe_prob, sharpe
from lab.walkforward import walk_forward
from plots import plot_equity
from risk import apply_limits


def wants_test():
    return "--test" in sys.argv


def _label(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def run_experiment(name, version, prices, build_weights, grid, cost_bps, borrow_bps=0.0,
                   benchmark_prices=None, limits=None, label=None):
    """
    name / version : strategy id, and a version tag you bump when the LOGIC changes
    prices         : price table the strategy trades
    build_weights  : function(prices, **settings) -> weights table
    grid           : {"setting": [values, ...]}  every combination is tried
    benchmark_prices: price column (DataFrame with 1 column) for buy-and-hold comparison
    limits         : e.g. {"max_position": 0.25, "max_gross": 1.0} applied to every weights table
    """
    print(f"\n=== {name} {version} ===")
    validate_prices(prices)
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]

    def weights_for(params):
        w = build_weights(prices, **params)
        return apply_limits(w, **limits) if limits else w

    # 1. run every setting, keep daily returns on a common start date so scores are comparable
    results, starts = {}, []
    for p in combos:
        res = trim_and_rebase(run_backtest(prices, weights_for(p), cost_bps, borrow_bps))
        if res.empty:
            continue
        results[_label(p)] = (p, res)
        starts.append(res.index[0])
    common_start = max(starts)
    R = pd.DataFrame({lab: r["ret"] for lab, (_, r) in results.items()}).loc[common_start:]
    R_train = R.loc[:config.TRAIN_END]

    # 2. score on TRAIN only, write everything in the lab notebook
    sr = sharpe(R_train)
    table = pd.DataFrame({"train_sharpe": sr,
                          "train_return": (1 + R_train).prod() - 1}).sort_values("train_sharpe", ascending=False)
    trials.log_trials(name, version, [{"params": lab, "train_sharpe": float(sr[lab]),
                                       "n_obs": len(R_train), "data_end": str(prices.index[-1].date())}
                                      for lab in table.index])
    n_total = trials.trial_count(name)

    print(f"Train period {R_train.index[0].date()} to {R_train.index[-1].date()} "
          f"({len(R_train)} days). Test period is locked.")
    show = table.copy()
    print("\nSettings ranked by TRAIN Sharpe:")
    print(show.to_string(float_format=lambda x: f"{x:.2f}"))

    # 3. how much of the best result could be luck?
    best_label = table.index[0]
    daily_sr_std = (R_train.mean() / R_train.std()).std()
    dsr = deflated_sharpe_prob(R_train[best_label], n_total, daily_sr_std)
    wf, picks = walk_forward(R_train)
    print(f"\nBest setting: {best_label}")
    print(f"  Best in-sample Sharpe                : {sr[best_label]:.2f}")
    if len(wf):
        print(f"  Walk-forward Sharpe (picked each year using only the past): {sharpe(wf):.2f}"
              f"  over {wf.index.year.nunique()} years, {len(set(picks.values()))} different settings used")
    print(f"  Total versions tried so far (all sessions): {n_total}")
    print(f"  Deflated Sharpe probability           : {dsr:.2f}   (>= 0.95 = convincing, < 0.5 = could be luck)")
    verdict = ("convincing so far" if dsr >= 0.95 else
               "promising but not proven" if dsr >= 0.75 else "NOT distinguishable from luck yet")
    print(f"  Verdict on train evidence: {verdict}")

    best_params, best_res = results[best_label]
    weights = weights_for(best_params)
    latest = weights.iloc[-1]
    print("  Current holdings:", {t: round(w, 3) for t, w in latest.items() if abs(w) > 1e-9})

    # 4. the locked test period
    if not wants_test():
        print("\nTEST PERIOD NOT SHOWN. When you are ready for ONE honest look:  add  --test")
        return {"params": best_params, "weights": weights}

    looks = trials.test_look_count(name)
    full = best_res
    te = metrics(full.loc[config.TEST_START:, "ret"])
    trials.log_test_look(name, version, best_label, te.get("sharpe"), te.get("total_return"))
    print(f"\n#### TEST PERIOD (look number {looks + 1} for {name}; every look weakens the test) ####")
    split_report(full, f"{name}: {best_label}")
    bench = None
    if benchmark_prices is not None:
        bench = run_backtest(benchmark_prices, benchmark_prices * 0 + 1.0, delay=config.EXEC_DELAY_DAYS)
        bench = bench.loc[full.index[0]:]
        split_report(bench, "Benchmark: buy and hold over the same dates")
    cost_sensitivity(prices, weights, borrow_bps=borrow_bps)
    curves = {name: full}
    if bench is not None:
        curves["Buy & hold"] = bench
    plot_equity(curves, f"{name} ({best_label})", f"{name}.png")
    full.to_csv(f"output/{name}_equity.csv")
    return {"params": best_params, "weights": weights, "result": full}
