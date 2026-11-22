# Industry-readiness standards for this project

Goal: an environment where we can iterate on strategies by trial and error **without fooling ourselves**,
with code that could be handed to a professional desk. These standards apply to every step from now on.

## The honest framing
Nothing here can promise profit. What it can do is stop us from deploying strategies that only look
profitable because we tried many ideas and kept the lucky one. Most backtested "edges" disappear live.
Real-capital decisions are always yours; this is research and paper-trading practice, not investment advice.

## Rules of the lab (what the code enforces)
| Rule | Where | Why |
|---|---|---|
| Test period (2026-01-01 on) is **locked**; shown only with `--test`, and every look is logged | `lab/experiment.py`, `experiments/test_looks.csv` | A test you peek at repeatedly stops being a test |
| Every setting ever tried is logged forever | `lab/trials.py`, `experiments/trials.csv` | The trial count is needed to measure luck |
| Deflated Sharpe probability reported for the chosen setting | `lab/stats.py` | Best-of-N results are inflated; this deflates them |
| Walk-forward score (settings picked each year using only the past) | `lab/walkforward.py` | Honest estimate of "pick the best setting" as a procedure |
| Strategies are pure functions: prices -> target weights | `strategies/*/strategy.py` | Same code can be backtested, paper-traded, or live; the broker is a separate adapter |
| Execution delay of 1 day, costs, borrow fees | `engine.py`, `config.py` | No trading at the price that produced the signal |
| Cost-sensitivity table with every test look | `engine.cost_sensitivity` | Shows fragility to the cost assumption |
| Crisis windows (dot-com, 2008, COVID) reported **with and without**, plus a stress table of what happened inside them; used for reporting only, never to rank settings or hide results | `config.CRISIS_WINDOWS`, `engine.crisis_table` | A strategy must be judged by what it does when markets break, not only when they are calm |
| Cash earns the T-bill rate; Sharpe is on excess returns; strategy is tested against a benchmark's returns (information ratio, t-stat), not only against zero | `data.get_cash_rate`, `engine.py`, `lab/stats.versus_benchmark` | Beating cash and beating a simple alternative are the bars that matter |
| Pre-trade risk limits (max position, max gross) | `risk.py` | Strategy proposes, risk layer disposes |
| Data-quality checks before every run | `data.validate_prices` | Bad data silently ruins backtests |
| Causality tests (no look-ahead) and engine accounting tests, run in CI | `tests/`, `.github/workflows/tests.yml` | Bugs here are the most expensive kind |
| Each strategy has a `description.txt` (idea, rules, exact instruments, limitations) | `strategies/*/` | Anyone can audit what it does |

## How to iterate without self-deception
1. Change settings or ideas, run `python stepN_name.py` (train only). Look at deflated Sharpe and walk-forward Sharpe, not only the best Sharpe.
2. Changing the strategy's **logic**? Bump `VERSION` ("v1" -> "v2") in the run script. It counts as new trials.
3. Commit `experiments/trials.csv` regularly so the trial count survives.
4. When you genuinely believe in a version, run with `--test` **once**. If it fails, do not tune against the test result; treat the failure as information, or wait for new data to accumulate.
5. A result is only interesting if it survives: higher costs, the walk-forward check, a deflated Sharpe near or above 0.95, and a sensible economic reason it should work.

## Known gaps before real money (not yet built)
* **Point-in-time universe**: our 50 stocks are today's winners (survivorship bias). Needs historical index membership.
* **Data**: Yahoo Finance is free but unaudited. Production needs a vendor feed with corporate-action history and delisted names.
* **Broker adapter + paper trading** (Alpaca): orders, fills, reconciliation against our target weights.
* **Realistic execution**: only a flat cost + 1-day delay today; no intraday fills, no market impact, no capacity limits.
* **Live monitoring**: kill switch, drawdown alerts, daily P&L reconciliation, signal logging.
* **Portfolio level**: combining strategies, correlated-risk review, vol targeting.
* **More testing history**: the test window is under one year. A strategy needs far more out-of-sample time, ideally a paper-trading period, before real money.
