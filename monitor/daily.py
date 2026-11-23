"""Daily monitor. Run from the repo root:  python -m monitor.daily

1. Downloads fresh prices (the GitHub runner starts with an empty data/ folder, so nothing stale can be reused).
2. Rebuilds every ACTIVE book with its frozen v1 settings (lab/registry.py) up to the last FINAL daily bar.
3. Records target weights and the forward paper NAV in monitor/*.csv and a readable monitor/latest.md.
4. Exits with code 1 if something is wrong (stale data, a book could not be built, too many missing tickers),
   so GitHub shows a red cross and emails you.

It RECORDS. It never changes a setting and never computes a Sharpe or alpha from the new data.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from data import get_prices
from lab import registry
from monitor import ledger

OUT = Path(__file__).parent
MAX_STALE_DAYS = 4          # newest bar older than this (calendar days) = problem (covers a long weekend)
MIN_COVERAGE = 0.95         # share of the strategy universes that must have downloaded


def universe():
    from strategies.momentum.strategy import STOCKS
    from strategies.sma.strategy import TICKERS
    from strategies.trend_riskparity.strategy import ASSETS
    return sorted(set(STOCKS) | set(TICKERS) | set(ASSETS))


def write_outputs(books, warnings, health, end, out=OUT, now=None):
    """Update the CSV logs and latest.md. Returns the list of PROBLEMS (empty = all fine)."""
    problems = list(warnings)
    if not books:
        problems.append("no book could be built")
        as_of = None
    else:
        as_of = max(b["ret"].index[-1] for b in books.values())
        if (end - as_of).days > MAX_STALE_DAYS:
            problems.append(f"newest price bar is {as_of.date()}, {(end - as_of).days} days before {end.date()}: data is stale")

    sig_old = ledger.load_csv(out / "signals.csv", ledger.SIG_COLS)
    nav_old = ledger.load_csv(out / "paper_nav.csv", ledger.NAV_COLS)
    sig_new, nav_new, rows = [], [], []
    for key, b in books.items():
        day = b["ret"].index[-1]
        sig_new.append(ledger.signal_rows(key, day, b["weights"]))
        nav_new.append(ledger.nav_rows(key, b["ret"], b.get("bench"), config.PAPER_START))
        prev_days = sig_old[(sig_old["book"] == key) & (sig_old["date"] < day.strftime("%Y-%m-%d"))]["date"]
        prev = None
        if len(prev_days):
            rows_prev = sig_old[(sig_old["book"] == key) & (sig_old["date"] == prev_days.max())]
            prev = pd.Series(rows_prev["weight"].values, index=rows_prev["ticker"].values)
        cur = b["weights"][b["weights"].abs() > 1e-9]
        entries, exits, turn = ledger.weight_changes(prev, b["weights"])
        nav = nav_new[-1]["nav"].iloc[-1]
        rows.append((key, b["label"], day, nav, len(cur), cur.sum(), turn if prev is not None else None, entries if prev is not None else None, exits if prev is not None else None, cur))

    if sig_new:
        ledger.replace_rows(sig_old, pd.concat(sig_new, ignore_index=True), ["date", "book"]).to_csv(out / "signals.csv", index=False)
        ledger.replace_rows(nav_old, pd.concat(nav_new, ignore_index=True), ["book"]).to_csv(out / "paper_nav.csv", index=False)

    now = now or datetime.now(timezone.utc)
    L = ["# Daily monitor", "",
         f"- Last final price bar used: **{as_of.date() if as_of is not None else 'none'}** (job ran {now:%Y-%m-%d %H:%M} UTC)",
         f"- Paper tracking: every book = $100 at the close of {config.PAPER_START.date()}, then followed forward. "
         "Books start already holding their positions (no entry cost), like the backtests.",
         f"- Status: **{'OK' if not problems else 'PROBLEMS: ' + str(len(problems))}**", ""]
    if problems:
        L += ["## Problems (the job failed because of these)", ""] + [f"- {p}" for p in problems] + [""]
    L += ["## Books", "", "| Book | As of | Paper NAV | Positions | Invested | Turnover vs previous log |", "|---|---|---|---|---|---|"]
    for key, label, day, nav, n, gross, turn, entries, exits, cur in rows:
        L.append(f"| {label} | {day.date()} | ${nav:,.2f} | {n} | {gross:.0%} | {'first log' if turn is None else f'{turn:.1%}'} |")
    L += ["", "## Target weights for the next trading day (top 10 per book)", ""]
    for key, label, day, nav, n, gross, turn, entries, exits, cur in rows:
        top = cur.reindex(cur.abs().sort_values(ascending=False).index).head(10)
        L.append(f"- **{label}**: " + (", ".join(f"{t} {w:.1%}" for t, w in top.items()) or "all cash"))
        if entries or exits:
            L.append(f"  - since previous log: entered {', '.join(entries) or 'none'}; exited {', '.join(exits) or 'none'}")
    if health:
        L += ["", "## Data notes (not failures)", ""] + [f"- {h}" for h in health[:30]]
    L += ["", "---", "This file records what the frozen strategies hold and how $100 would have grown since the paper start. "
          "It does not judge them. Reviews are on scheduled dates only (see monitor/README.md)."]
    (out / "latest.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return problems


def main():
    now = datetime.now(timezone.utc)
    end = ledger.last_final_date(now)
    print(f"using daily bars up to {end.date()}")
    books, warnings = registry.build_books(end=end, only_active=True)

    uni = universe()
    px = get_prices(uni).loc[:end]
    missing = sorted(set(uni) - set(px.columns))
    if 1 - len(missing) / len(uni) < MIN_COVERAGE:
        warnings.append(f"only {len(uni) - len(missing)} of {len(uni)} tickers downloaded; missing: {', '.join(missing)}")
    health = ledger.data_health(px) + [f"{t}: not downloaded" for t in missing]

    problems = write_outputs(books, warnings, health, end, now=now)
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
