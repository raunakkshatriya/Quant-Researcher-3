# Lessons learned (Quant-Researcher-3)

Short, dated, one lesson per heading. Read before changing the engine, the stats helpers, or any benchmark.

## 2026-11-23 - Sharpe changed scale when cash started earning interest
Early trial-log rows (before the cash-yield change, commit 69be10d era) used cash = 0% and Sharpe = mean/std
(risk-free rate 0). Since then Sharpe is measured over T-bills (`lab/stats.py::_excess`, `engine.daily_cash`).
Momentum v1 (252d, top 10): 1.141 on the old scale, 1.047 on the current one - reproduced exactly with
`diag_momentum_reconcile.py`. The gap is larger for strategies that hold lots of cash (trend, sector rotation).
Rules: (1) never compare a trial-log Sharpe with a fresh run without checking which scale each is on;
(2) old rows, including old deflated-Sharpe probabilities, are optimistic; (3) reconcile any new number against the
logged one BEFORE interpreting it.

## 2026-11-23 - Don't let a secondary series trim the comparison window
Step 3b's first version intersected dates with SPY (starts 1993-02) and silently dropped momentum's first two years
(Sharpe 2.70 there), so it reported 0.96 instead of 1.05. Rule: the window is set by the strategies being compared;
a benchmark with shorter history is reported on its own window, with the window printed.

## 2026-11-23 - Judge a stock strategy against equal-weight of the SAME stocks, not SPY
The 50 stocks (and the 49 in the SMA step) are today's biggest companies, so buy-and-hold of the list has a Sharpe
about double SPY's (survivorship). Momentum vs SPY: alpha +15.9%/yr (t 5.3). Momentum vs equal-weight of the same
list: alpha +5.4%/yr (t 2.3), beta 1.07, Sharpe 1.05 vs 0.99. Verdict: weakly supported, a mild noisy tilt, not
a clean edge; beat equal-weight in only 4 of 7 five-year blocks. 3 of the 12 grid settings reach |t| >= 2 and they
are neighbours, not independent evidence. Fair yardsticks: stocks -> equal-weight list; sector rotation ->
equal-weight sectors; trend -> 60/40 SPY/IEF. Remaining fix for survivorship: a point-in-time universe (needs data).

## 2026-11-23 - Test the helper's output type, not just its logic
`equal_weight_index` returns a price LEVEL, not returns. Used as returns it printed a Sharpe of 17 and volatility
of 869,000%. Absurd numbers on first run are a bug signal, not a result. Smoke-test new scripts on synthetic data
where the right answer is known (no true edge -> alpha near 0) before running on real data.

## 2026-11-23 - Order of operations when comparing two conventions
My first old-vs-new Sharpe check computed both after switching cash off, so both lines matched. Compute each
number under its own setting before changing global state (`engine.set_default_cash_rate` is global).

## Test-look ledger (do not tune against these)
momentum 1, trend 2, sector rotation 1, pairs_stocks 1, pairs_etfs 1, fx_carry 2, sma 1.
