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


def validate_prices(prices, max_daily_move=0.4, stale_days=10):
    """Basic data-quality checks. Bad data silently ruins backtests, so we look before we trust.

    Prints a warning for: price <= 0, a one-day move bigger than +-40% (often a data error or
    an unadjusted split), a price frozen for 10+ days (a dead or halted ticker), and very short
    histories. Returns the list of warnings (empty = clean).
    """
    warnings = []
    for t in prices.columns:
        p = prices[t].dropna()
        if p.empty:
            warnings.append(f"{t}: no data"); continue
        if (p <= 0).any():
            warnings.append(f"{t}: non-positive prices")
        big = p.pct_change().abs() > max_daily_move
        if big.any():
            warnings.append(f"{t}: {int(big.sum())} daily moves above {max_daily_move:.0%} (first: {big.idxmax().date()})")
        frozen = (p.diff() == 0).astype(int).groupby((p.diff() != 0).cumsum()).cumsum().max()
        if frozen >= stale_days:
            warnings.append(f"{t}: price unchanged for {int(frozen)} days in a row")
        if len(p) < 252:
            warnings.append(f"{t}: only {len(p)} days of history")
    for w in warnings:
        print("  DATA WARNING:", w)
    return warnings


def get_cash_rate():
    """Daily series of the 3-month US Treasury bill rate (annual, as a decimal: 0.045 = 4.5%).

    Source: FRED series DTB3 (free). Cached in data/DTB3.csv. Returns None if it cannot be
    downloaded, in which case uninvested cash is assumed to earn 0% (and we say so loudly).
    """
    file = DATA_DIR / "DTB3.csv"
    if not file.exists():
        try:
            print("downloading T-bill rates (FRED DTB3) ...")
            pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3").to_csv(file, index=False)
        except Exception as e:
            print(f"  WARNING: could not download cash rates ({e}). Cash will earn 0%.")
            return None
    raw = pd.read_csv(file, index_col=0, parse_dates=True).iloc[:, 0]
    return (pd.to_numeric(raw, errors="coerce") / 100.0).dropna()


def get_fred_series(series_id):
    """Any FRED series as a Series of DECIMALS (4.5 percent -> 0.045), cached in data/<id>.csv.

    Returns None (with a warning) if it cannot be downloaded, so one missing series never crashes a run.
    """
    file = DATA_DIR / f"{series_id}.csv"
    if not file.exists():
        try:
            print(f"downloading FRED {series_id} ...")
            pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}").to_csv(file, index=False)
        except Exception as e:
            print(f"  WARNING: could not download FRED {series_id} ({e})")
            return None
    raw = pd.read_csv(file, index_col=0, parse_dates=True).iloc[:, 0]
    out = (pd.to_numeric(raw, errors="coerce") / 100.0).dropna()
    return out if len(out) else None


def get_volumes(tickers, start="1990-01-01"):
    """Daily share volume, one column per ticker. Reads the Volume column of the same cached files get_prices uses
    (downloading them first if needed). Yahoo volume is consolidated (all US exchanges) and split-adjusted."""
    if isinstance(tickers, str):
        tickers = [tickers]
    get_prices(tickers, start=start)             # makes sure the cache files exist
    columns = {}
    for t in tickers:
        file = DATA_DIR / f"{t}.csv"
        if not file.exists():
            continue
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        if "Volume" in df.columns:
            columns[t] = df["Volume"]
    return pd.DataFrame(columns).sort_index()
