"""The backtest engine: turns "what we hold each day" into "how much money we have".

THE ONE IDEA YOU MUST UNDERSTAND
--------------------------------
A strategy produces WEIGHTS: for each day and each asset, what fraction of our money we
want to hold.   1.0 = all our money in it,  0.5 = half,  0 = none,  -0.5 = short half.

Weights decided at the END of day t (using prices up to day t) can only earn the return of
day t+1. So we SHIFT the weights down by one day before multiplying by returns.
Forgetting this shift is "look-ahead bias" - the most common way backtests lie.
"""
import numpy as np
import pandas as pd

import config


def run_backtest(prices, weights, cost_bps=0.0, borrow_bps=0.0, delay=None):
    """Simulate a strategy.

    prices    : table of prices (day x ticker)
    weights   : table of target weights, same shape (decided at the end of each day)
    cost_bps  : trading cost per unit traded, in basis points (1 bp = 0.01%).
                5 means: buying or selling $1 costs $0.0005. One number covering commission,
                bid-ask spread and slippage (see each strategy's description.txt).
    borrow_bps: yearly fee on SHORT positions, in bp of the short amount (25 = 0.25% a year).
    delay     : extra days between the signal day and the trade day. None = config.EXEC_DELAY_DAYS.
                With delay=1 a decision made with Monday's close is traded at Tuesday's close
                and first earns Wednesday's return.

    Returns a table with columns: ret (daily return after costs), equity (starts at $100).
    """
    returns = prices.pct_change().fillna(0.0)            # each day's % change
    if delay is None:
        delay = config.EXEC_DELAY_DAYS
    held = weights.reindex(prices.index).fillna(0.0).shift(1 + delay).fillna(0.0)   # <-- the shift
    gross = (held * returns).sum(axis=1)                 # money made before costs
    traded = held.diff().abs().sum(axis=1).fillna(0.0)   # how much we bought/sold that day
    cost = traded * cost_bps / 10_000
    short_amount = (-held.clip(upper=0)).sum(axis=1)     # how much we are short today
    borrow = short_amount * borrow_bps / 10_000 / 252    # daily share of the yearly fee
    ret = gross - cost - borrow
    equity = config.START_CAPITAL * (1 + ret).cumprod()
    return pd.DataFrame({"ret": ret, "equity": equity, "traded": traded})


def metrics(ret, days_per_year=252):
    """Summary numbers for a series of daily returns (risk-free rate assumed 0)."""
    ret = ret.dropna()
    if len(ret) < 20:
        return {}
    equity = (1 + ret).cumprod()
    years = len(ret) / days_per_year
    drawdown = equity / equity.cummax() - 1        # how far below the previous peak we are
    return {
        "total_return": equity.iloc[-1] - 1,
        # CAGR = compound yearly growth. Meaningless for < 1 year, so we leave it out then.
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years >= 1 else np.nan,
        "volatility": ret.std() * np.sqrt(days_per_year),            # yearly size of wiggles
        "sharpe": ret.mean() / ret.std() * np.sqrt(days_per_year) if ret.std() > 0 else np.nan,
        "max_drawdown": drawdown.min(),                              # worst peak-to-trough fall
    }


def split_report(result, name):
    """Print train and test metrics side by side."""
    train = metrics(result.loc[:config.TRAIN_END, "ret"])
    train_ex = metrics(drop_crises(result.loc[:config.TRAIN_END, "ret"]))
    test = metrics(result.loc[config.TEST_START:, "ret"])
    rows = [("Total return", "total_return", "{:.1%}"), ("CAGR (per year)", "cagr", "{:.1%}"),
            ("Volatility", "volatility", "{:.1%}"), ("Sharpe", "sharpe", "{:.2f}"),
            ("Worst drawdown", "max_drawdown", "{:.1%}")]
    print(f"\n{name}")
    print(f"{'':18}{'TRAIN':>12}{'TRAIN ex-crisis':>17}{'TEST':>12}")
    for label, key, fmt in rows:
        f = lambda m: fmt.format(m[key]) if key in m and pd.notna(m[key]) else "-"
        print(f"{label:18}{f(train):>12}{f(train_ex):>17}{f(test):>12}")
    if len(result.loc[config.TEST_START:]):
        end = result.loc[config.TEST_START:, "ret"].add(1).prod() * config.START_CAPITAL
        print(f"$100 invested on {config.TEST_START.date()} is worth ${end:.2f} at the end of the data")


def month_end_dates(index):
    """The last trading day of each month (a finished month only).

    Many strategies only change their holdings once a month. This gives us those days.
    The current, unfinished month is dropped so we never trade on a "fake" month-end.
    """
    s = pd.Series(index, index=index)
    ends = pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())
    if len(ends) and (ends[-1] + pd.offsets.BDay(1)).month == ends[-1].month:
        ends = ends[:-1]
    return ends


def trim_and_rebase(result):
    """Cut off the days before the strategy first held anything and restart equity at $100.

    A strategy that needs 1 year of history to make its first decision has nothing to
    report before that. Starting the $100 on the first day we actually hold something
    keeps the curve honest.
    """
    held = result["traded"] > 0
    if not held.any():
        return result.iloc[0:0]
    out = result.loc[held.idxmax():].copy()
    out["equity"] = config.START_CAPITAL * (1 + out["ret"]).cumprod()
    return out


def cost_sensitivity(prices, weights, costs=(0, 3, 10, 25), **kwargs):
    """Re-run the same strategy with different trading costs and print a small table.

    If the strategy only works at 0-3 bp and dies at 10 bp, its edge is fragile.
    Extra keyword arguments (e.g. borrow_bps=25) are passed on to run_backtest.
    """
    print("\nCost sensitivity (same strategy, different cost per $ traded):")
    print(f"{'cost (bp)':>10}{'train Sharpe':>14}{'train return':>14}{'test Sharpe':>13}{'test return':>13}")
    for c in costs:
        res = trim_and_rebase(run_backtest(prices, weights, cost_bps=c, **kwargs))
        tr, te = metrics(res.loc[:config.TRAIN_END, "ret"]), metrics(res.loc[config.TEST_START:, "ret"])
        g = lambda m, k, f: f.format(m[k]) if k in m else "-"
        print(f"{c:>10}{g(tr, 'sharpe', '{:.2f}'):>14}{g(tr, 'total_return', '{:.0%}'):>14}"
              f"{g(te, 'sharpe', '{:.2f}'):>13}{g(te, 'total_return', '{:.1%}'):>13}")


def crisis_mask(index):
    """True for every date inside one of the crisis windows in config.CRISIS_WINDOWS."""
    mask = pd.Series(False, index=index)
    for start, end in config.CRISIS_WINDOWS.values():
        mask.loc[start:end] = True
    return mask


def drop_crises(series):
    """The same daily-return series with the crisis days removed (for 'ex-crisis' statistics)."""
    return series[~crisis_mask(series.index)]


def crisis_table(strategy_ret, benchmark_ret=None):
    """Print what the strategy (and benchmark) did INSIDE each crisis window.

    This is the stress test. A defensive strategy is supposed to shine here; a strategy that only
    works when crises are removed is fragile, and this table shows it.
    """
    rows = []
    for name, (a, b) in config.CRISIS_WINDOWS.items():
        r = strategy_ret.loc[a:b]
        if len(r) < 20 or r.index[0] > pd.Timestamp(a) + pd.Timedelta(days=30):
            continue                                   # strategy did not exist for this window
        eq = (1 + r).cumprod()
        row = {"window": name, "strategy": eq.iloc[-1] - 1, "worst drawdown": (eq / eq.cummax() - 1).min()}
        if benchmark_ret is not None and len(benchmark_ret.loc[a:b]):
            row["benchmark"] = (1 + benchmark_ret.loc[a:b]).prod() - 1
        rows.append(row)
    if not rows:
        print("  (no crisis windows fall inside this strategy's history)")
        return
    print(pd.DataFrame(rows).set_index("window").to_string(float_format=lambda x: f"{x:.1%}"))
