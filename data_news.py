"""Historical company news from Alpaca's News API (the articles come from Benzinga).

Free with an Alpaca account (paper-trading keys work). History goes back to 2015.
Keys are read from two environment variables and are NEVER stored in the repo:
    APCA_API_KEY_ID      APCA_API_SECRET_KEY
Each ticker is cached in data/news/<TICKER>.csv with columns:
    id, ticker, created_at (UTC, ISO), headline, n_symbols (how many tickers the article is tagged with)
Re-running resumes from the newest cached article, so it is safe to stop (Ctrl+C) and start again.
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
NEWS_DIR = Path(__file__).parent / "data" / "news"
PAGE_LIMIT = 50                     # the API's maximum articles per call


def _utc(ts):
    """Any date / timestamp (with or without a timezone) as a UTC timestamp."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def api_headers():
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca keys not found. Set the environment variables APCA_API_KEY_ID and "
                           "APCA_API_SECRET_KEY (see docs/ALPACA_SETUP.md), then open a NEW terminal.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _get(session, params, retries=6, pause=0.0):
    """One API call with retry on rate limits (429) and temporary server errors. Fails loudly on bad keys."""
    for attempt in range(retries):
        r = session.get(NEWS_URL, headers=api_headers(), params=params, timeout=60)
        if r.status_code == 200:
            if pause:
                time.sleep(pause)
            return r.json()
        if r.status_code in (401, 403):
            raise RuntimeError(f"Alpaca refused the keys (HTTP {r.status_code}): {r.text[:200]}")
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"  (HTTP {r.status_code}, waiting {wait:.0f}s)")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Alpaca news error HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Alpaca news: too many retries")


def fetch_pages(ticker, start, end=None, session=None, pause=0.35):
    """Yield one list of article-rows per API page, oldest first, for one ticker."""
    session = session or requests.Session()
    params = {"symbols": ticker, "start": _utc(start).isoformat(), "limit": PAGE_LIMIT,
              "sort": "asc", "include_content": "false"}
    if end is not None:
        params["end"] = _utc(end).isoformat()
    token = None
    while True:
        if token:
            params["page_token"] = token
        data = _get(session, params, pause=pause)
        rows = [{"id": int(a["id"]), "ticker": ticker, "created_at": a["created_at"],
                 "headline": (a.get("headline") or "").replace("\n", " ").strip(),
                 "n_symbols": len(a.get("symbols") or [])} for a in data.get("news", [])]
        if rows:
            yield rows
        token = data.get("next_page_token")
        if not token:
            return


def load_cached(ticker, news_dir=NEWS_DIR):
    f = Path(news_dir) / f"{ticker}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["id", "ticker", "created_at", "headline", "n_symbols"])
    return pd.read_csv(f).drop_duplicates(["id", "ticker"])


def download_news(tickers, start="2015-01-01", end=None, session=None, news_dir=NEWS_DIR, pause=0.35):
    """Download (or resume) news for each ticker into the cache. Prints one line per ticker."""
    Path(news_dir).mkdir(parents=True, exist_ok=True)
    for t in tickers:
        f = Path(news_dir) / f"{t}.csv"
        have = load_cached(t, news_dir)
        # resume from just before the newest cached article
        # The API orders by UPDATED date, so an old article can reappear late: re-read the last 3 days (duplicates are dropped).
        s = pd.to_datetime(have["created_at"], utc=True).max() - pd.Timedelta(days=3) if len(have) else _utc(start)
        new = 0
        for rows in fetch_pages(t, s, end, session, pause):
            pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
            new += len(rows)
        total = len(load_cached(t, news_dir))
        print(f"{t:6s} +{new:6d} rows fetched, {total:6d} cached")


def load_news(tickers, news_dir=NEWS_DIR):
    """All cached articles for the tickers as one table (one row per article per ticker)."""
    parts = [load_cached(t, news_dir) for t in tickers]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise RuntimeError("No cached news. Run:  python step10_news_sentiment.py fetch")
    return pd.concat(parts, ignore_index=True)
