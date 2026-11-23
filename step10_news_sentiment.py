"""STEP 10 - Strategy: news sentiment on 49 stocks (Alpaca/Benzinga headlines scored by FinBERT on your GPU).

Run the commands IN THIS ORDER (read strategies/news_sentiment/description.txt first):

   python step10_news_sentiment.py fetch              # download headlines 2015-> (resumable, needs Alpaca keys)
   python step10_news_sentiment.py score              # FinBERT on the GPU, cached (resumable)
   python step10_news_sentiment.py ic                 # PHASE A: does sentiment predict returns at all? (train only)
   python step10_news_sentiment.py run                # PHASE B: backtest, only worth reading if Phase A showed something
   python step10_news_sentiment.py run --test         # one logged look at the locked test period

Options: fetch --start 2015-01-01 | score --batch 128 | score --cpu (slow, only if the GPU fails)
"""
import argparse
import sys
from functools import partial

import pandas as pd

import config
import data_news
from data import get_prices
from lab.experiment import run_experiment
from lab.ic import daily_ic, forward_returns, quintile_spread, summarize
from strategies.news_sentiment.scorer import SCORES_FILE, finbert_scorer, score_articles
from strategies.news_sentiment.signal import MAX_SYMBOLS, daily_sentiment
from strategies.news_sentiment.strategy import build_weights
from strategies.sma.strategy import TICKERS, equal_weight_index

VERSION = "v1"
COST_BPS = 5
WINDOWS = [1, 5]            # days of headlines averaged
K = 10                      # stocks held


def cmd_fetch(a):
    data_news.download_news(TICKERS, start=a.start)
    news = data_news.load_news(TICKERS)
    news["year"] = pd.to_datetime(news["created_at"], utc=True).dt.year
    print("\nArticles per year (all 49 tickers). A big jump or gap between years means the data's coverage changed:")
    print(news.groupby("year").size().to_string())
    print(f"Articles tagged with at most {MAX_SYMBOLS} tickers (the ones we use): "
          f"{(news['n_symbols'] <= MAX_SYMBOLS).mean():.0%} of rows")


def cmd_score(a):
    news = data_news.load_news(TICKERS)
    news = news[news["n_symbols"] <= MAX_SYMBOLS]
    print(f"{news['id'].nunique():,} unique articles to consider (cached scores are reused).")
    scorer = finbert_scorer(device="cpu" if a.cpu else "cuda", batch_size=a.batch)
    scores = score_articles(news, scorer)
    print(f"Done. {len(scores):,} scores in {SCORES_FILE}")


def load_inputs():
    prices = get_prices(TICKERS)
    stocks = prices[[t for t in TICKERS if t in prices.columns]]
    news = data_news.load_news(TICKERS)
    scores = pd.read_csv(SCORES_FILE).drop_duplicates("id").set_index("id")["score"]
    s, c = daily_sentiment(news, scores, stocks.index)
    return stocks, s, c


def cmd_ic(a):
    stocks, s, c = load_inputs()
    stocks_tr = stocks.loc[: config.TRAIN_END]
    s, c = s.loc[: config.TRAIN_END], c.loc[: config.TRAIN_END]
    first = c.sum(axis=1).gt(0).idxmax()
    print(f"PHASE A - does news sentiment predict next-period relative returns? (train {first.date()} to {config.TRAIN_END.date()}; test locked)")
    print(f"Stock-days with at least one article: {(c > 0).sum().sum():,} of {c.loc[first:].size:,}")
    for window in WINDOWS:
        rs, rc = s.rolling(window).sum(), c.rolling(window).sum()
        sig = (rs / rc).where(rc >= 3).loc[first:]
        print(f"\n--- signal = average sentiment of the last {window} trading day(s) (>= 3 articles) ---")
        n_opinion = sig.notna().sum(axis=1)
        print(f"Coverage: on an average day {n_opinion.mean():.1f} of {sig.shape[1]} stocks have an opinion "
              f"(the strategy needs at least 15 on a week-end day to trade).")
        rows = {}
        for h in (1, 5):
            fwd = forward_returns(stocks_tr, h)
            ic, sp = daily_ic(sig, fwd), quintile_spread(sig, fwd)
            r, q = summarize(ic, h), summarize(sp, h)
            if r:
                rows[f"{h}-day ahead"] = {"mean IC": f"{r['mean']:+.4f}", "t-stat": f"{r['t_stat']:.2f}", "% days IC>0": f"{r['pct_positive']:.0%}",
                                          "days": r["days"], "top-minus-bottom fifth (bp)": f"{q['mean'] * 1e4:+.1f}", "its t-stat": f"{q['t_stat']:.2f}"}
        if not rows:
            print("Too few stock-days with enough articles to measure anything for this window. (This is a COVERAGE finding, "
                  "not a result about returns.)")
            continue
        print(pd.DataFrame(rows).T.to_string())
        ic1 = daily_ic(sig, forward_returns(stocks_tr, 1))
        by_year = ic1.groupby(ic1.index.year).mean().dropna()
        print("Mean 1-day IC by year:", "  ".join(f"{y}:{v:+.3f}" for y, v in by_year.items()))
    print("\nHow to read it: mean IC near 0 or |t| < 2 = no usable signal (stop here). A real signal needs |t| >= 2 AND a "
          "positive IC in most years, AND the fifth-minus-fifth spread large enough to beat costs (5 bp per trade each way).")


def cmd_run(a):
    stocks, s, c = load_inputs()
    bench = pd.DataFrame({"EqualWeight49": equal_weight_index(stocks)})
    print("WARNING: the 49 stocks are TODAY's biggest companies -> survivorship bias flatters everything, and megacaps are the most-read stocks (weakest case for news).")
    run_experiment(
        name="news_sentiment", version=VERSION, prices=stocks,
        build_weights=partial(build_weights, score_sum=s, count=c),
        grid={"window": WINDOWS, "k": [K]},
        cost_bps=COST_BPS, borrow_bps=0.0, limits={"max_position": 0.25, "max_gross": 1.0},
        benchmark_prices=bench, benchmark_name="equal-weight buy & hold",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--start", default="2015-01-01"); f.set_defaults(fn=cmd_fetch)
    sc = sub.add_parser("score"); sc.add_argument("--batch", type=int, default=128); sc.add_argument("--cpu", action="store_true"); sc.set_defaults(fn=cmd_score)
    sub.add_parser("ic").set_defaults(fn=cmd_ic)
    r = sub.add_parser("run"); r.add_argument("--test", action="store_true"); r.set_defaults(fn=cmd_run)
    args, _ = p.parse_known_args()
    args.fn(args)
