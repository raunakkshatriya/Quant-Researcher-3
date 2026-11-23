"""DIAGNOSTIC: how far back does YOUR Alpaca account really serve news? Read-only, ~15 small calls.

   python check_news_history.py
"""
import sys

import requests

import data_news

try:
    headers = data_news.api_headers()
except RuntimeError as e:
    sys.exit(f"FAIL: {e}")


def call(params):
    r = requests.get(data_news.NEWS_URL, headers=headers, params=params, timeout=60)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    return r.json().get("news", []), ""


def show(label, params):
    items, err = call(params)
    if items is None:
        print(f"{label:42s} ERROR {err}")
        return
    dates = [a["created_at"][:10] for a in items]
    print(f"{label:42s} {len(items):3d} articles" + (f", created {min(dates)} .. {max(dates)}" if dates else ""))


print("--- Does AAPL news exist on specific past days? (one-day windows, up to 50 articles each) ---")
for day in ["2016-06-15", "2018-06-15", "2021-06-15", "2023-06-15", "2024-06-14", "2025-06-16", "2026-01-15", "2026-06-15"]:
    nxt = str(__import__("pandas").Timestamp(day) + __import__("pandas").Timedelta(days=1))[:10]
    show(f"AAPL {day}", {"symbols": "AAPL", "start": day, "end": nxt, "limit": 50, "sort": "asc"})

print("\n--- Oldest article the account returns for AAPL (start 2015, ascending) ---")
items, err = call({"symbols": "AAPL", "start": "2015-01-01", "limit": 3, "sort": "asc"})
print(err or [(a["created_at"], a.get("updated_at"), a["headline"][:60]) for a in items])

print("\n--- GOOGL vs GOOG, last 60 days of 2026 so far ---")
for s in ["GOOGL", "GOOG", "GOOGL,GOOG"]:
    show(f"{s} 2026-09-01..2026-11-20", {"symbols": s, "start": "2026-09-01", "end": "2026-11-20", "limit": 50, "sort": "desc"})

print("\n--- No symbol filter, one old day (market-wide) ---")
show("ALL 2019-03-12", {"start": "2019-03-12", "end": "2019-03-13", "limit": 50, "sort": "asc"})
print("\nPaste this whole output back.")
