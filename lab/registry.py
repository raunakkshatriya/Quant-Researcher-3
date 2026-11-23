"""One place that knows every strategy's FIXED v1 settings, universe, costs and fair benchmark.

The dashboard builds each strategy's $100 book from here, TRAIN PERIOD ONLY (nothing after config.TRAIN_END is used).
Adding a strategy later = one more `_book_*` function and one line in BUILDERS.
Settings are the ones chosen in the individual steps; nothing is searched here.
"""
import numpy as np
import pandas as pd

import config
import engine
from data import get_cash_rate, get_prices
from engine import run_backtest, trim_and_rebase
from risk import apply_limits

END = config.TRAIN_END
LIMITS = {"max_position": 0.25, "max_gross": 1.0}

# status decides whether a book is part of the combined portfolio ("active") or shown greyed out ("rejected")
STATUS = {
    "momentum": ("Momentum", "Weakly supported", True),
    "trend": ("Trend / risk parity", "Diversifier, no Sharpe edge", True),
    "sma": ("SMA trend filter", "Risk reducer, no Sharpe edge", True),
    "fx_carry": ("FX carry", "Weak, not significant", True),
    "sector": ("Sector rotation", "Rejected", False),
}

TECH_GROUPS = {"semiconductors", "software_internet", "hardware_networking"}   # + XLK for the sector ETFs


def _simulate(prices, weights, cost_bps, borrow_bps=0.0):
    """Daily returns of one book from its first invested day to the train end."""
    res = trim_and_rebase(run_backtest(prices, weights, cost_bps=cost_bps, borrow_bps=borrow_bps, delay=config.EXEC_DELAY_DAYS))
    return res["ret"].loc[:END]


def _constant_mix(prices, mix, start):
    """Benchmark: fixed percentages, rebalanced daily, same engine and delay."""
    px = prices[list(mix)].dropna()
    w = pd.DataFrame({c: wt for c, wt in mix.items()}, index=px.index)
    return run_backtest(px, w, delay=config.EXEC_DELAY_DAYS)["ret"].loc[start:END]


def _book(ret, weights, bench, bench_name, groups):
    return {"ret": ret, "weights": weights.loc[:END].iloc[-1], "bench": bench, "bench_name": bench_name, "groups": groups}


def _momentum():
    from strategies.momentum.strategy import STOCKS, build_weights
    from strategies.pairs.strategy import STOCK_GROUPS
    from strategies.sma.strategy import equal_weight_index
    px = get_prices(list(STOCKS)).loc[:END]
    px = px[[t for t in STOCKS if t in px.columns]]
    w = apply_limits(build_weights(px, 252, 10, "long"), **LIMITS)
    ret = _simulate(px, w, cost_bps=5, borrow_bps=25)
    bench = equal_weight_index(px).pct_change().loc[ret.index[0]:]
    return _book(ret, w, bench, "equal-weight of the same 50 stocks", _stock_groups(STOCK_GROUPS))


def _sma():
    from strategies.pairs.strategy import STOCK_GROUPS
    from strategies.sma.strategy import TICKERS, build_weights, equal_weight_index
    px = get_prices(TICKERS).loc[:END]
    px = px[[t for t in TICKERS if t in px.columns]]
    w = apply_limits(build_weights(px, "price>SMA200"), **LIMITS)
    ret = _simulate(px, w, cost_bps=5)
    bench = equal_weight_index(px).pct_change().loc[ret.index[0]:]
    return _book(ret, w, bench, "equal-weight of the same 49 stocks", _stock_groups(STOCK_GROUPS))


def _sector():
    from strategies.sector_rotation.strategy import SECTOR_ETFS, build_weights
    px = get_prices(SECTOR_ETFS).loc[:END]
    w = build_weights(px, 252, 4)
    ret = _simulate(px, w, cost_bps=3)
    bench = _constant_mix(px, {t: 1.0 / len(SECTOR_ETFS) for t in SECTOR_ETFS}, ret.index[0])
    return _book(ret, w, bench, "equal-weight of the 11 sector ETFs", {t: ("technology" if t == "XLK" else t) for t in SECTOR_ETFS})


ASSET_CLASS = {"SPY": "equities", "EFA": "equities", "EEM": "equities", "IWM": "equities", "QQQ": "equities",
               "TLT": "bonds", "IEF": "bonds", "LQD": "bonds", "HYG": "bonds", "SHY": "bonds",
               "GLD": "gold", "DBC": "commodities", "USO": "commodities", "VNQ": "real estate"}


def _trend():
    from strategies.trend_riskparity.strategy import ASSETS, BENCHMARK_WEIGHTS, build_weights
    px = get_prices(list(ASSETS)).loc[:END]
    w = apply_limits(build_weights(px, 126, 0.06), max_position=1.0, max_gross=1.0)
    ret = _simulate(px, w, cost_bps=3)
    bench = _constant_mix(px, BENCHMARK_WEIGHTS, ret.index[0])
    return _book(ret, w, bench, "60/40 SPY/IEF", {t: ASSET_CLASS.get(t, "other") for t in ASSETS})


def _fx():
    from functools import partial
    from strategies.fx_carry.data_fx import load_fx
    from strategies.fx_carry.strategy import build_weights
    spot, carry, tri, _ = load_fx()
    tri, spot, carry = tri.loc[:END], spot.loc[:END], carry.loc[:END]
    w = apply_limits(partial(build_weights, carry=carry, spot=spot)(tri, 1.0, 252), **LIMITS)      # carry only, lookback unused
    ret = _simulate(tri, w, cost_bps=3)
    return _book(ret, w, None, "cash", {c: "currencies" for c in tri.columns})


def _stock_groups(stock_groups):
    return {t: g for g, ts in stock_groups.items() for t in ts}


BUILDERS = {"momentum": _momentum, "trend": _trend, "sma": _sma, "fx_carry": _fx, "sector": _sector}


def spy_returns():
    return get_prices(["SPY"])["SPY"].loc[:END].pct_change()


def build_books():
    """Returns (books, warnings). A strategy whose data cannot be loaded is skipped with a warning, never faked."""
    engine.set_default_cash_rate(get_cash_rate() if config.USE_CASH_YIELD else None)
    books, warnings = {}, []
    for key, fn in BUILDERS.items():
        label, status, active = STATUS[key]
        try:
            b = fn()
        except (Exception, SystemExit) as e:                     # missing data files, FRED outage (some loaders call sys.exit), etc.
            warnings.append(f"{label}: skipped ({type(e).__name__}: {str(e)[:120] or "data could not be loaded"})")
            continue
        books[key] = {**b, "label": label, "status": status, "active": active}
    return books, warnings
