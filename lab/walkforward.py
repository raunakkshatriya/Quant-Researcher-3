"""Walk-forward validation: a more honest 'train' score.

Picking the best setting on ALL of the train period and then scoring it on that same period
flatters it. Walk-forward instead replays history like a real trader:
  for each year Y:  choose the best setting using only years BEFORE Y, then use it during Y.
Stitching the Y years together gives returns that were never selected on their own data.
If the stitched Sharpe is far below the best in-sample Sharpe, the setting-picking is overfit.
"""
import pandas as pd

from lab.stats import sharpe


def walk_forward(returns_by_setting, min_years=5):
    """returns_by_setting: DataFrame (day x setting) of daily returns.

    Returns (stitched_returns Series, picks dict {year: setting_label}).
    """
    R = returns_by_setting
    pieces, picks = [], {}
    for year in sorted(set(R.index.year)):
        past = R[R.index.year < year]
        if len(past) < min_years * 252:
            continue
        best = sharpe(past).idxmax()
        picks[year] = best
        pieces.append(R.loc[R.index.year == year, best])
    if not pieces:
        return pd.Series(dtype=float), picks
    return pd.concat(pieces), picks
