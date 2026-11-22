"""STEP 0 - Can your machine download stock prices?

Every strategy starts with the same thing: a table of daily prices.
This script downloads one stock (Apple) and shows you what that table looks like.

Run it with:   python step0_check_data.py
"""
import pandas as pd
import yfinance as yf

TICKER = "AAPL"          # the stock symbol we ask for
START = "2000-01-01"     # "as far back as you can, starting from here"

print(f"Downloading daily prices for {TICKER} ...")

# auto_adjust=True means prices are corrected for stock splits and dividends.
# Without it, a 4-for-1 split looks like a 75% crash. Always use adjusted prices for backtests.
prices = yf.download(TICKER, start=START, auto_adjust=True, progress=False)

if prices.empty:
    raise SystemExit("Nothing came back. Check your internet connection and try again.")

# Newer versions of yfinance return two-level column names like ('Close', 'AAPL').
# We only need the first level ('Close'), so we flatten them.
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(0)

print()
print("First 3 rows (the oldest days):")
print(prices.head(3))
print()
print("Last 3 rows (the most recent days):")
print(prices.tail(3))
print()
print(f"Rows (trading days): {len(prices)}")
print(f"First date: {prices.index[0].date()}   Last date: {prices.index[-1].date()}")
print(f"Missing values in Close: {int(prices['Close'].isna().sum())}")

# Save a copy so later steps can reuse it instead of downloading again.
prices.to_csv(f"data/{TICKER}.csv")
print(f"\nSaved to data/{TICKER}.csv  - Step 0 passed.")
