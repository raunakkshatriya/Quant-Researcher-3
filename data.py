"""Download daily prices once, save them in data/, reuse them afterwards."""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"


def get_prices(tickers, start="1990-01-01"):
    """Return a table of adjusted closing prices: one row per day, one column per ticker.

    Example:
                    AAPL     MSFT
        2000-01-03  0.83     36.0
        2000-01-04  0.76     34.8

    The first time you ask for a ticker it is downloaded and saved to data/<TICKER>.csv.
    Next time the saved file is used (fast, works offline).
    To refresh to today's prices, delete the csv files in data/.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    columns = {}
    for t in tickers:
        file = DATA_DIR / f"{t}.csv"
        if not file.exists():
            print(f"downloading {t} ...")
            df = yf.download(t, start=start, auto_adjust=True, progress=False)
            if df.empty:
                print(f"  WARNING: no data for {t}, skipping")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(file)
        columns[t] = pd.read_csv(file, index_col=0, parse_dates=True)["Close"]
    return pd.DataFrame(columns).sort_index()
