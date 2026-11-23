"""Pure helpers for the daily monitor: no downloads and no strategy logic, so they are easy to test.

Two append-friendly CSV files are kept in monitor/ (dates are ISO strings so they never change type):
  signals.csv   date, book, ticker, weight      target weights decided at that day's close (traded the next day)
  paper_nav.csv date, book, nav, bench_nav      growth of $100 since config.PAPER_START (bench = the book's fair yardstick)
"""
import datetime as dt

import pandas as pd

import config

SIG_COLS = ["date", "book", "ticker", "weight"]
NAV_COLS = ["date", "book", "nav", "bench_nav"]


def last_final_date(now_utc, final_hour=21, final_minute=30):
    """Last calendar date whose daily bars are FINAL.

    US stocks close at 20:00 or 21:00 UTC (summer/winter). Before 21:30 UTC on a weekday today's bar may be a half-finished
    intraday number, so we only trust up to yesterday. Weekends: yesterday too (the loaders simply have no newer bar).
    """
    ready = now_utc.weekday() < 5 and (now_utc.hour, now_utc.minute) >= (final_hour, final_minute)
    return pd.Timestamp(now_utc.date() if ready else now_utc.date() - dt.timedelta(days=1))


def forward_nav(ret, start, capital=None):
    """$100 at the close of `start`, then compounded with the daily returns AFTER `start`."""
    capital = config.START_CAPITAL if capital is None else capital
    r = ret.loc[ret.index > start].dropna()
    return pd.concat([pd.Series([capital], index=[pd.Timestamp(start)]), capital * (1 + r).cumprod()])


def nav_rows(book, ret, bench, start):
    nav = forward_nav(ret, start)
    bench_nav = forward_nav(bench, start).reindex(nav.index) if bench is not None else pd.Series(float("nan"), index=nav.index)
    return pd.DataFrame({"date": nav.index.strftime("%Y-%m-%d"), "book": book,
                         "nav": nav.round(4).values, "bench_nav": bench_nav.round(4).values})


def signal_rows(book, as_of, weights):
    w = weights[weights.abs() > 1e-9]
    return pd.DataFrame({"date": pd.Timestamp(as_of).strftime("%Y-%m-%d"), "book": book,
                         "ticker": w.index.astype(str), "weight": w.round(6).values}, columns=SIG_COLS)


def replace_rows(old, new, keys):
    """Drop the rows of `old` that share `keys` with any row of `new`, then add `new`. Re-running a day never duplicates it."""
    if old is None or old.empty:
        out = new
    else:
        marked = old.merge(new[keys].drop_duplicates(), on=keys, how="left", indicator=True)
        out = pd.concat([old[(marked["_merge"] == "left_only").values], new], ignore_index=True)
    return out.sort_values([c for c in ["date", "book", "ticker"] if c in out.columns]).reset_index(drop=True)


def load_csv(path, cols):
    return pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame(columns=cols)


def data_health(prices, max_lag_days=3, max_move=0.25, lookback=5):
    """Plain-language warnings about the newest data: tickers that stopped updating, or suspicious one-day jumps."""
    out, latest = [], prices.index.max()
    for t in prices.columns:
        p = prices[t].dropna()
        if p.empty:
            out.append(f"{t}: no data"); continue
        lag = (latest - p.index[-1]).days
        if lag > max_lag_days:
            out.append(f"{t}: last price {p.index[-1].date()} is {lag} days behind the others")
        mv = p.pct_change().tail(lookback)
        for d, v in mv[mv.abs() > max_move].items():
            out.append(f"{t}: {v:+.0%} move on {d.date()} (check for a split or data error)")
    return out


def weight_changes(prev, cur):
    """prev/cur: Series of weights by ticker (prev may be None). Returns (entries, exits, one-way turnover)."""
    prev = pd.Series(dtype=float) if prev is None else prev
    p = prev[prev.abs() > 1e-9]; c = cur[cur.abs() > 1e-9]
    both = p.index.union(c.index)
    turnover = 0.5 * (c.reindex(both).fillna(0) - p.reindex(both).fillna(0)).abs().sum()
    return sorted(set(c.index) - set(p.index)), sorted(set(p.index) - set(c.index)), float(turnover)
