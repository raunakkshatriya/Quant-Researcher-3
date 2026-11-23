import numpy as np
import pandas as pd
import pytest

import data_news
from lab.ic import daily_ic, forward_returns, quintile_spread, summarize
from strategies.news_sentiment.scorer import score_articles
from strategies.news_sentiment.signal import daily_sentiment, signal_dates
from strategies.news_sentiment.strategy import MIN_ARTICLES, build_weights

DAYS = pd.bdate_range("2024-01-02", "2024-12-31")          # trading days (holidays ignored in tests)


def sd(ts_list):
    return signal_dates(pd.Series(pd.to_datetime(ts_list, utc=True)), DAYS)


def test_signal_date_cutoff_is_4pm_new_york_summer_and_winter():
    # July (EDT, UTC-4): 15:59 NY = 19:59 UTC -> same day;  16:00 NY = 20:00 UTC -> next trading day
    r = sd(["2024-07-09 19:59:00", "2024-07-09 20:00:00"])
    assert list(r) == [pd.Timestamp("2024-07-09"), pd.Timestamp("2024-07-10")]
    # January (EST, UTC-5): 15:59 NY = 20:59 UTC;  16:00 NY = 21:00 UTC. A fixed UTC-4 assumption would get this wrong.
    r = sd(["2024-01-09 20:59:00", "2024-01-09 21:00:00"])
    assert list(r) == [pd.Timestamp("2024-01-09"), pd.Timestamp("2024-01-10")]


def test_signal_date_weekend_and_friday_evening_roll_to_monday():
    # Fri 2024-07-12 evening, Saturday and Sunday news are all first usable on Monday 2024-07-15
    r = sd(["2024-07-12 21:00:00", "2024-07-13 15:00:00", "2024-07-14 15:00:00"])
    assert set(r) == {pd.Timestamp("2024-07-15")}
    assert sd(["2024-07-13 15:00:00"]).iloc[0] > pd.Timestamp("2024-07-12")


def test_signal_date_after_last_trading_day_is_nat():
    assert pd.isna(sd(["2025-03-03 15:00:00"]).iloc[0])


def _articles():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
        "created_at": ["2024-07-09 15:00:00+00:00", "2024-07-09 15:30:00+00:00", "2024-07-09 21:00:00+00:00",
                       "2024-07-09 15:00:00+00:00", "2024-07-09 15:00:00+00:00"],
        "headline": ["Good news", "good news ", "Later news", "Roundup", "Other"],
        "n_symbols": [1, 1, 1, 9, 1]})


def test_daily_sentiment_dedupes_filters_roundups_and_respects_close():
    scores = pd.Series({1: 0.5, 2: 0.5, 3: -1.0, 4: 1.0, 5: 0.2})
    s, c = daily_sentiment(_articles(), scores, DAYS)
    d1, d2 = pd.Timestamp("2024-07-09"), pd.Timestamp("2024-07-10")
    assert c.loc[d1, "AAA"] == 1 and s.loc[d1, "AAA"] == 0.5        # duplicate headline counted once
    assert c.loc[d2, "AAA"] == 1 and s.loc[d2, "AAA"] == -1.0       # 21:00 UTC (5pm NY) news counts for the NEXT day
    assert c.loc[d1, "BBB"] == 1 and s.loc[d1, "BBB"] == 0.2        # 9-ticker roundup ignored


class FakeResponse:
    def __init__(self, code, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.text = code, payload or {}, headers or {}, "x"

    def json(self):
        return self._p


class FakeSession:
    """Serves a list of articles, 2 per page, honouring the start parameter and page_token."""
    def __init__(self, items, code=200):
        self.items, self.code, self.calls = items, code, 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        if self.code != 200:
            return FakeResponse(self.code)
        start = pd.Timestamp(params["start"])
        rows = [a for a in self.items if pd.Timestamp(a["created_at"]) >= start]
        off = int(params.get("page_token", 0))
        page = rows[off: off + 2]
        nxt = str(off + 2) if off + 2 < len(rows) else None
        return FakeResponse(200, {"news": page, "next_page_token": nxt})


def make_items(n, t0="2024-03-01T14:00:00Z"):
    base = pd.Timestamp(t0)
    return [{"id": i, "created_at": (base + pd.Timedelta(hours=i)).isoformat().replace("+00:00", "Z"),
             "headline": f"headline {i}", "symbols": ["AAA"]} for i in range(n)]


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "PKTEST")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")


def test_download_paginates_then_resumes_without_duplicates(tmp_path, keys):
    items = make_items(5)
    data_news.download_news(["AAA"], start="2024-01-01", session=FakeSession(items), news_dir=tmp_path, pause=0)
    assert len(data_news.load_cached("AAA", tmp_path)) == 5
    items += make_items(8)[5:]                                        # 3 newer articles appear
    sess = FakeSession(items)
    data_news.download_news(["AAA"], start="2024-01-01", session=sess, news_dir=tmp_path, pause=0)
    got = data_news.load_cached("AAA", tmp_path)
    assert len(got) == 8 and got["id"].is_unique
    assert got["n_symbols"].eq(1).all()


def test_missing_keys_and_bad_keys_fail_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="keys not found"):
        data_news.download_news(["AAA"], session=FakeSession([]), news_dir=tmp_path, pause=0)
    monkeypatch.setenv("APCA_API_KEY_ID", "bad")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "bad")
    with pytest.raises(RuntimeError, match="refused the keys"):
        data_news.download_news(["AAA"], session=FakeSession([], code=401), news_dir=tmp_path, pause=0)


def test_scoring_is_resumable_and_scores_each_article_once(tmp_path):
    arts = pd.DataFrame({"id": [1, 2, 3, 3], "headline": ["a", "bb", "ccc", "ccc"], "ticker": ["X", "X", "X", "Y"]})
    calls = []

    def fake(texts):
        calls.append(list(texts))
        return np.array([len(t) / 10 for t in texts])
    out = score_articles(arts, fake, cache_path=tmp_path / "s.csv", chunk=2, progress=False)
    assert out.to_dict() == {1: 0.1, 2: 0.2, 3: 0.3}
    n = sum(len(c) for c in calls)
    assert n == 3                                                      # id 3 appears twice but is scored once
    score_articles(arts, fake, cache_path=tmp_path / "s.csv", chunk=2, progress=False)
    assert sum(len(c) for c in calls) == n                             # second run scores nothing


def _tables():
    idx = pd.bdate_range("2020-01-01", periods=60)
    cols = list("ABCDEF")
    prices = pd.DataFrame(100.0, index=idx, columns=cols)
    rng = np.random.default_rng(0)
    s = pd.DataFrame(rng.normal(0, 1, (60, 6)), index=idx, columns=cols)
    c = pd.DataFrame(5.0, index=idx, columns=cols)
    return prices, s, c


def test_weights_pick_most_positive_and_never_use_the_future(monkeypatch):
    import strategies.news_sentiment.strategy as st
    monkeypatch.setattr(st, "MIN_STOCKS", 3)
    prices, s, c = _tables()
    w = build_weights(prices, s, c, window=1, k=2)
    last_fri = [d for d in w.dropna().index if d.weekday() == 4][-1]
    row = (s.loc[last_fri] / c.loc[last_fri]).sort_values()
    assert set(w.loc[last_fri][w.loc[last_fri] > 0].index) == set(row.index[-2:])
    assert np.isclose(w.loc[last_fri].sum(), 1.0)
    cut = prices.index[30]
    s2 = s.copy()
    s2.loc[cut + pd.Timedelta(days=1):] = 99.0                          # rewrite the future
    w2 = build_weights(prices, s2, c, window=1, k=2)
    pd.testing.assert_frame_equal(w.loc[:cut].fillna(-1), w2.loc[:cut].fillna(-1))


def test_too_few_articles_means_no_opinion(monkeypatch):
    import strategies.news_sentiment.strategy as st
    monkeypatch.setattr(st, "MIN_STOCKS", 3)
    prices, s, c = _tables()
    c[:] = MIN_ARTICLES - 1
    assert build_weights(prices, s, c, window=1, k=2).fillna(0).abs().sum().sum() == 0


def _ic_inputs(planted):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    ret = pd.DataFrame(rng.normal(0, 0.01, (1500, 25)), index=idx, columns=[f"S{i}" for i in range(25)])
    prices = 100 * (1 + ret).cumprod()
    return idx, ret, prices


def test_ic_detects_a_tradable_signal_and_ignores_known_past_returns():
    idx, ret, prices = _ic_inputs(True)
    fwd = forward_returns(prices, 1, delay=1)                           # close(t+2)/close(t+1) - 1
    noise = pd.DataFrame(np.random.default_rng(2).normal(0, 1, ret.shape), index=idx, columns=ret.columns)
    good = fwd.fillna(0) * 50 + noise                                   # genuinely predicts the tradable return
    assert summarize(daily_ic(good, fwd))["mean"] > 0.1
    past = prices.pct_change()                                          # return up to today: already known, predicts nothing
    r = summarize(daily_ic(past, fwd))
    assert abs(r["t_stat"]) < 3
    assert summarize(quintile_spread(good, fwd))["mean"] > 0


def test_ic_needs_enough_stocks():
    idx, ret, prices = _ic_inputs(False)
    fwd = forward_returns(prices, 1)
    sig = prices.pct_change().copy()
    sig.iloc[:, 5:] = np.nan                                            # only 5 stocks have a signal
    assert daily_ic(sig, fwd, min_names=15).isna().all()
