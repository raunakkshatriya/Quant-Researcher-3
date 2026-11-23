"""Turn scored news articles into a (date x ticker) sentiment table WITHOUT looking ahead.

THE TIMING RULE (the most important thing in this module)
We decide at the 4pm New York close of day t and trade at the next close. So the decision on day t may only use
news published BEFORE 16:00 New York time on day t. News at or after 16:00, on a weekend, or on a holiday is
first usable on the NEXT trading day. (Half-days that close at 1pm are not modelled: news between 1pm and 4pm on
about three days a year is treated as usable that day. Tiny, and noted.)
"""
import pandas as pd

MAX_SYMBOLS = 3          # ignore articles tagged with more than this many tickers (market roundups are noise)
CUTOFF_HOUR = 16         # New York close


def signal_dates(created_utc, trading_days):
    """For each article timestamp: the first trading day whose 4pm-NY close is AFTER the article. NaT if none yet."""
    ny = pd.to_datetime(created_utc, utc=True).dt.tz_convert("America/New_York")
    after_close = (ny.dt.hour >= CUTOFF_HOUR).astype(int)
    day = ny.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close, unit="D")
    pos = trading_days.searchsorted(day.values, side="left")
    ok = pos < len(trading_days)
    out = pd.Series(pd.NaT, index=day.index, dtype="datetime64[ns]")
    out[ok] = trading_days.values[pos[ok]]
    return out


def daily_sentiment(articles, scores, trading_days, max_symbols=MAX_SYMBOLS):
    """Return (score_sum, article_count): two tables, rows = trading_days, columns = tickers.

    articles : columns id, ticker, created_at (UTC), headline, n_symbols
    scores   : Series indexed by article id (FinBERT score in [-1, 1])
    Repeated identical headlines for the same ticker on the same signal day count once.
    """
    a = articles[articles["n_symbols"] <= max_symbols].copy()
    a["score"] = a["id"].map(scores)
    a = a.dropna(subset=["score"])
    a["date"] = signal_dates(a["created_at"], trading_days)
    a = a.dropna(subset=["date"])
    a["hl"] = a["headline"].str.lower().str.strip()
    a = a.drop_duplicates(["ticker", "date", "hl"])
    g = a.groupby(["date", "ticker"])["score"].agg(["sum", "count"])
    s = g["sum"].unstack("ticker").reindex(trading_days).fillna(0.0)
    c = g["count"].unstack("ticker").reindex(trading_days).fillna(0.0)
    return s, c
