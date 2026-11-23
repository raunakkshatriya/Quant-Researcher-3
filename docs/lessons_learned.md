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

## 2026-11-23 - Sector rotation and trend re-scored against fair benchmarks (current Sharpe scale, train)
Sector rotation v1 (252d, top 4): Sharpe 0.47 vs equal-weight of the 11 sector ETFs 0.56 (SPY 0.40); alpha vs
equal-weight +3.0%/yr, t 1.18, beta 0.92, R2 0.84; worst drawdown -46% vs -36%. All 12 grid settings score <= 0.47,
i.e. none beats simply owning all sectors. Verdict: NOT supported.
Trend v1 (126d, vol 6%): Sharpe 0.70 vs 60/40 0.69; return 6.1% vs 8.8%, volatility 6.4% vs 10.7%, drawdown -15% vs
-31%; alpha +2.4%/yr (t 1.98) at beta 0.27, but active return -3.0%/yr (t -1.5). Longer lookbacks are worse, so v1
is the best of 9. Verdict: no Sharpe edge over 60/40; it is a lower-risk, low-correlation diversifier (R2 0.21),
which is a portfolio-level question for the combined-dashboard review, not evidence of alpha.
General: a strategy that beats SPY but not its own fair yardstick has shown nothing. Pattern so far: momentum weak,
sector rotation no, trend tie, pairs no, FX carry weak, SMA no (risk reduction only).

## 2026-11-23 - News sentiment (step 10) PARKED: the free news feed has no history
Built the full pipeline (fetch -> FinBERT on GPU -> 4pm-NY timing -> IC check -> backtest, all unit-tested and
mutation-checked). On the user's free Alpaca paper account the News API returned nothing before 2026-04-29, even for
AAPL and market-wide, despite docs advertising history back to 2015 (`check_news_history.py`). Only 2026 data exists,
which is inside the locked test period, so no honest train/test is possible. GOOGL returned nothing because the feed
tags Alphabet as GOOG. OpenBB cannot help (it is a wrapper over providers; the repos offered were unofficial forks).
Options left open: forward collection (rerun `python step10_news_sentiment.py fetch` now and then; unknown whether the
free window rolls forward and drops old items, so keep the CSVs), a public dataset such as FNSPID (1999-2023, timestamp
precision unverified), or a paid plan (history unlocked? unverified). Expectation stays "probably null" for mega-caps.
Rule: before building a pipeline on a data source, probe a few specific PAST days for the real history depth
(`check_news_history.py` style) - docs and marketing pages can disagree with what a given account is served.

## Test-look ledger (do not tune against these)
momentum 1, trend 2, sector rotation 1, pairs_stocks 1, pairs_etfs 1, fx_carry 2, sma 1.
