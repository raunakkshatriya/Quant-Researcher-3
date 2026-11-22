"""Everything FX needs beyond prices: interest-rate differentials and 'total return' price series.

THE KEY IDEA
Holding a foreign currency earns (or pays) the interest-rate DIFFERENCE between it and the dollar,
on top of the exchange-rate move. So the return of "long AUD, funded in USD" for one day is
        spot move  +  (AUD rate - USD rate) / 252
We fold both into one 'total return index' per currency, so the normal backtest engine can trade it.
"""
import numpy as np
import pandas as pd

from data import get_fred_series, get_prices

# code -> (Yahoo ticker, is the Yahoo quote 'foreign per USD' (True) or 'USD per foreign' (False), FRED 3-month
# interbank rate ids to try in order). All are OECD monthly series on FRED.
CURRENCIES = {
    "EUR": ("EURUSD=X", False, ["IR3TIB01EZM156N", "IR3TIB01DEM156N"]),
    "GBP": ("GBPUSD=X", False, ["IR3TIB01GBM156N"]),
    "JPY": ("JPY=X", True, ["IR3TIB01JPM156N"]),
    "CHF": ("CHF=X", True, ["IR3TIB01CHM156N"]),
    "AUD": ("AUDUSD=X", False, ["IR3TIB01AUM156N"]),
    "NZD": ("NZDUSD=X", False, ["IR3TIB01NZM156N"]),
    "CAD": ("CAD=X", True, ["IR3TIB01CAM156N"]),
    "NOK": ("NOK=X", True, ["IR3TIB01NOM156N"]),
    "SEK": ("SEK=X", True, ["IR3TIB01SEM156N"]),
}
USD_RATE_ID = "IR3TIB01USM156N"
# The OECD series for EUR and GBP stopped updating in Jan 2026. Where the OECD series is stale we CONTINUE it
# with a central-bank policy rate that is still published daily (see splice_rate). Used only when needed.
SUCCESSORS = {"EUR": "ECBDFR",     # ECB deposit facility rate (daily, from 1999)
              "GBP": "IUDSOIA"}    # SONIA, the sterling overnight rate (daily). FRED has no Bank Rate series under IUDBEDR (404)
SPLICE_OVERLAP_MONTHS = 6          # months of overlap used to match the two series' levels
SPLICE_MIN_GAP_MONTHS = 2          # splice only if the successor reaches at least this many months past the original
PUBLICATION_LAG_MONTHS = 2   # a monthly value is an AVERAGE over its month and published weeks later:
                             # the value for month m is only treated as known from the start of month m+2
MAX_STALE_DAYS = 150         # if a rate series stops updating, drop that currency after this many days
FFILL_DAYS = 5               # fill holiday gaps in exchange rates by at most this many days


def daily_rate(monthly, index, lag_months=PUBLICATION_LAG_MONTHS, max_stale_days=MAX_STALE_DAYS):
    """Turn a monthly rate series (annual decimals) into a daily series of what was KNOWN on each day.

    1. The value dated month m becomes known at the start of month m + lag_months (no peeking).
    2. It is held until the next value arrives.
    3. If the newest known value is older than max_stale_days, the rate is NaN (we do not trust stale data).
    """
    known_from = monthly.copy()
    known_from.index = known_from.index + pd.DateOffset(months=lag_months)
    union = index.union(known_from.index)
    val = known_from.reindex(union).ffill().reindex(index)
    when = pd.Series(known_from.index, index=known_from.index).reindex(union).ffill().reindex(index)
    age = (index.to_series() - when).dt.days
    return val.where(age <= max_stale_days)


def splice_rate(primary, successor_daily, overlap_months=SPLICE_OVERLAP_MONTHS, min_gap_months=SPLICE_MIN_GAP_MONTHS):
    """Continue a stale monthly rate series with a daily successor series (both annual decimals).

    The successor is averaged into months (same convention as the OECD series). Its level differs a little from the
    original (a policy rate is not a 3-month interbank rate), so we add the AVERAGE gap over the last
    `overlap_months` months both exist. Values up to the original's last month are never changed.
    Returns (series, info). info is None when no splice was needed or possible.
    """
    succ = successor_daily.resample("MS").mean().dropna()
    last = primary.index[-1]
    if succ.empty or succ.index[-1] < last + pd.DateOffset(months=min_gap_months):
        return primary, None                                            # original is not stale: leave it alone
    overlap = primary.index.intersection(succ.index)[-overlap_months:]
    if len(overlap) < 3:
        return primary, None                                            # cannot match levels: do not guess
    offset = float((primary[overlap] - succ[overlap]).mean())
    extension = succ[succ.index > last] + offset
    out = pd.concat([primary, extension])
    return out, {"spliced_after": last.date(), "offset_bp": 1e4 * offset, "months_added": len(extension)}


def total_return_index(spot, carry):
    """Total-return price series per currency = spot move + one day of interest differential.
    Where the exchange rate is missing the series is NaN. Starts at 100."""
    ret = spot.pct_change(fill_method=None) + carry.shift(1).fillna(0.0) / 252
    tri = 100 * (1 + ret.fillna(0.0)).cumprod()
    return tri.where(spot.notna())


def load_fx(start="1990-01-01"):
    """Download (or read from cache) everything. Returns (spot, carry, tri, report).
       spot  : USD per 1 unit of each foreign currency (day x currency)
       carry : known interest differential (foreign rate - USD rate), annual decimal (day x currency)
       tri   : total-return index (day x currency)
       report: one row per currency describing exactly what data we got (printed by the run script)."""
    tickers = [v[0] for v in CURRENCIES.values()]
    raw = get_prices(tickers, start=start)
    usd = get_fred_series(USD_RATE_ID)
    if usd is None:
        raise SystemExit("Could not get the US interest-rate series; carry cannot be computed.")
    spot_cols, rate_cols, rows = {}, {}, []
    for code, (tick, inverted, ids) in CURRENCIES.items():
        row = {"currency": code, "yahoo": tick}
        px = raw[tick].dropna() if tick in raw else pd.Series(dtype=float)
        rate, used = None, None
        for sid in ids:
            rate = get_fred_series(sid)
            if rate is not None:
                used = sid
                break
        if px.empty or rate is None:
            row["status"] = "DROPPED: no " + ("prices" if px.empty else "interest-rate series")
            rows.append(row)
            continue
        if code in SUCCESSORS:
            succ = get_fred_series(SUCCESSORS[code])
            if succ is not None:
                rate, info = splice_rate(rate, succ)
                if info:
                    row["spliced_with"] = SUCCESSORS[code]
                    row["splice_offset_bp"] = info["offset_bp"]
                    row["months_added"] = info["months_added"]
        spot_cols[code] = 1.0 / px if inverted else px
        rate_cols[code] = rate
        row.update({"rate_series": used, "price_first": px.index[0].date(), "price_last": px.index[-1].date(),
                    "rate_first": rate.index[0].date(), "rate_last_obs": rate.index[-1].date(), "status": "ok"})
        rows.append(row)
    if len(spot_cols) < 6:
        raise SystemExit(f"Only {len(spot_cols)} currencies have data; need at least 6.\n{pd.DataFrame(rows)}")
    spot = pd.DataFrame(spot_cols).sort_index()
    spot = spot.ffill(limit=FFILL_DAYS)
    spot = spot[spot.notna().sum(axis=1) >= 6]
    usd_d = daily_rate(usd, spot.index)
    carry = pd.DataFrame({c: daily_rate(r, spot.index) - usd_d for c, r in rate_cols.items()})
    tri = total_return_index(spot, carry)
    report = pd.DataFrame(rows).set_index("currency")
    report.attrs["usd_last_obs"] = usd.index[-1].date()
    last = spot.index[-1]
    report["carry_known_on_last_day"] = [bool(c in carry and pd.notna(carry[c].iloc[-1])) for c in report.index]
    report["avg carry %/yr"] = [100 * carry[c].mean() if c in carry else np.nan for c in report.index]
    return spot, carry, tri, report
