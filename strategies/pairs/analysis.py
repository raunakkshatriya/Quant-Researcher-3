"""Side-by-side comparison of pairs trading on two universes: WHICH is better, and WHEN.

Everything here is DESCRIPTIVE. We use it to understand behaviour, not to pick settings
(settings were already chosen on the train period by the lab). The regime labels below use
the whole train period's market history, so they must never feed back into a trading rule.
"""
import numpy as np
import pandas as pd

import config
from engine import crisis_mask, metrics
from lab.stats import sharpe


def _fmt(df):
    return df.to_string(float_format=lambda x: f"{x:.2f}")


def overall_table(rets, last_day):
    rows = {}
    for name, r in rets.items():
        r = r.loc[:last_day]
        m, mx = metrics(r), metrics(r[~crisis_mask(r.index)])
        rows[name] = {"first day": str(r.index[0].date()), "CAGR %": 100 * m["cagr"], "vol %": 100 * m["volatility"],
                      "Sharpe": m["sharpe"], "Sharpe ex-crisis": mx.get("sharpe", np.nan),
                      "worst DD %": 100 * m["max_drawdown"]}
    return pd.DataFrame(rows).T


def yearly_table(rets, market, summaries, last_day):
    """Calendar-year return of each strategy, the stock market, and how many pairs each universe traded."""
    out = {}
    for name, r in rets.items():
        out[f"{name} %"] = 100 * (1 + r.loc[:last_day]).groupby(r.loc[:last_day].index.year).prod().sub(1)
    out["SPY %"] = 100 * (1 + market.loc[:last_day]).groupby(market.loc[:last_day].index.year).prod().sub(1)
    for name, s in summaries.items():
        out[f"{name} pairs"] = s["chosen"]
    t = pd.DataFrame(out)
    return t[t[[c for c in t.columns if c.endswith(" %") and c != "SPY %"]].notna().any(axis=1)]


def regime_table(rets, spy_prices, last_day):
    """Sharpe of each strategy under different market conditions (labels from SPY).
       calm / normal / stressed : SPY 63-day realised volatility in the bottom / middle / top third
       market falling / rising  : SPY 63-day return below / above zero
       crisis window            : dot-com bust, global financial crisis, COVID crash (config.CRISIS_WINDOWS)"""
    spy = spy_prices.loc[:last_day]
    r = spy.pct_change()
    vol = r.rolling(63).std() * np.sqrt(252)
    trend = spy / spy.shift(63) - 1
    lo, hi = vol.quantile(1 / 3), vol.quantile(2 / 3)
    labels = {"calm market (low vol)": vol <= lo, "normal market": (vol > lo) & (vol <= hi),
              "stressed market (high vol)": vol > hi, "market falling (63d < 0)": trend < 0,
              "market rising (63d > 0)": trend >= 0, "inside a crisis window": crisis_mask(spy.index)}
    rows = {}
    for lab, mask in labels.items():
        row = {}
        for name, ret in rets.items():
            x = ret.loc[:last_day]
            x = x[mask.reindex(x.index).fillna(False)]
            row[f"{name} Sharpe"] = sharpe(x) if len(x) > 60 else np.nan
            row[f"{name} ret %/yr"] = 100 * x.mean() * 252 if len(x) > 60 else np.nan
        rows[lab] = row
    return pd.DataFrame(rows).T


def decade_table(rets, last_day):
    out = {}
    for name, r in rets.items():
        r = r.loc[:last_day]
        grp = (r.index.year // 5) * 5
        out[f"{name} Sharpe"] = r.groupby(grp).apply(lambda x: sharpe(x) if len(x) > 120 else np.nan)
    t = pd.DataFrame(out)
    t.index = [f"{y}-{y + 4}" for y in t.index]
    return t


def print_comparison(rets, summaries, spy_prices, include_test):
    """rets: {name: daily return Series}; summaries: {name: formation_summary table}."""
    last_day = rets[next(iter(rets))].index[-1] if include_test else pd.Timestamp(config.TRAIN_END)
    shown = "train + test" if include_test else f"TRAIN only (to {pd.Timestamp(config.TRAIN_END).date()})"
    market = spy_prices.pct_change().fillna(0.0)
    print(f"\n{'=' * 78}\nCOMPARISON: pairs on STOCKS vs pairs on SECTOR ETFs   [{shown}]\n{'=' * 78}")

    print("\n1. Overall (each strategy starts on its own first trading day):")
    print(_fmt(overall_table(rets, last_day)))
    names = list(rets)
    if len(names) == 2:
        a, b = (rets[n].loc[:last_day] for n in names)
        j = a.index.intersection(b.index)
        print(f"\n   Correlation of the two daily return streams: {a.loc[j].corr(b.loc[j]):.2f}"
              "   (near 0 = they are different bets; near 1 = same bet twice)")

    print("\n2. How many pairs were tested / passed the test / chosen each year (averages):")
    for n, s in summaries.items():
        if len(s):
            exp = 0.05 * s["tested"].mean()
            print(f"   {n:14} tested {s['tested'].mean():5.1f}  passed {s['passed'].mean():4.1f}  chosen {s['chosen'].mean():4.1f}"
                  f"   (pure luck alone would pass about {exp:.1f} of the tested pairs at p < 0.05)")

    print("\n3. WHEN does each work? Sharpe (and yearly return %) under different market conditions:")
    print(_fmt(regime_table(rets, spy_prices.loc[:last_day], last_day)))

    print("\n4. Sharpe by 5-year block:")
    print(_fmt(decade_table(rets, last_day)))

    print("\n5. Calendar years (return %, SPY for reference, number of pairs traded):")
    print(_fmt(yearly_table(rets, market, summaries, last_day)))
