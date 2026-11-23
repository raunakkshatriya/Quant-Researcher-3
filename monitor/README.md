# Daily monitor

A GitHub Action (`.github/workflows/daily-monitor.yml`) runs every weekday after the US close. It downloads fresh prices,
rebuilds the ACTIVE books (momentum, trend / risk parity, SMA trend filter, FX carry) with their FROZEN v1 settings and records:

| File | What it holds |
|---|---|
| `latest.md` | Status, paper NAV per book, target weights for the next trading day, entries/exits, data notes |
| `signals.csv` | Every day's non-zero target weights per book (`date, book, ticker, weight`) |
| `paper_nav.csv` | Growth of $100 since 2026-11-20 per book, plus its fair yardstick (`date, book, nav, bench_nav`) |

A red cross on the Actions tab (GitHub emails you) means: stale data, a book that could not be built, or too many tickers missing.

## The rule that keeps this honest
The monitor **records**. It never changes a setting and it computes no Sharpe, alpha or "is it working" verdict.
Looking at P&L every day and adjusting is the fastest way to overfit, and 2026 data is our locked test period.

- Settings change only through a written hypothesis review (a new step, a new trial-log row, a lessons_learned entry), never because of a bad week.
- Scheduled reviews (proposed): end of each quarter. At a review we compare each book's paper NAV with its yardstick over the whole span, and look at correlated risk again.
- A book that needs re-fitting to extremes again and again is a sign the hypothesis is breaking, not a sign the re-fit works.

## Limits
- Prices are Yahoo's adjusted closes. Past values can be rewritten slightly (dividends, splits), so the NAV is recomputed from scratch each run.
- The paper NAV uses the backtest engine (1-day delay, the same costs). It is not a fill-by-fill account.
- The stock universes are today's lists (survivorship bias), so forward results are cleaner than the backtests, but the lists themselves are not point-in-time.
- Run it by hand any time: Actions tab → daily-monitor → Run workflow.
