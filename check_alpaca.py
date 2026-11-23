"""Check that your Alpaca PAPER account and keys work. Read-only: it never places an order.

   python check_alpaca.py
"""
import os
import sys

import requests

import data_news

PAPER_ACCOUNT_URL = "https://paper-api.alpaca.markets/v2/account"

try:
    headers = data_news.api_headers()
except RuntimeError as e:
    sys.exit(f"FAIL: {e}")
print(f"Key id found: {os.environ['APCA_API_KEY_ID'][:4]}... (secret not shown). Paper keys normally start with 'PK'.")

r = requests.get(PAPER_ACCOUNT_URL, headers=headers, timeout=30)
if r.status_code != 200:
    sys.exit(f"FAIL: paper account call returned HTTP {r.status_code}: {r.text[:200]}\n"
             "Check that you copied the PAPER keys (Paper Trading dashboard), and opened a NEW terminal after setting them.")
a = r.json()
print(f"Paper account OK: status {a.get('status')}, equity ${float(a.get('equity', 0)):,.2f}, buying power ${float(a.get('buying_power', 0)):,.2f}, "
      f"trading blocked: {a.get('trading_blocked')}")

n = requests.get(data_news.NEWS_URL, headers=headers, params={"symbols": "AAPL", "limit": 1, "sort": "asc", "start": "2015-01-01T00:00:00Z"}, timeout=30)
if n.status_code != 200:
    sys.exit(f"FAIL: news call returned HTTP {n.status_code}: {n.text[:200]}")
items = n.json().get("news", [])
print("News API OK: oldest AAPL article since 2015 is dated", items[0]["created_at"] if items else "(none returned)")
print("\nAll good. Nothing was bought or sold.")
