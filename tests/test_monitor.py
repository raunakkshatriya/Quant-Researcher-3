import datetime as dt

import numpy as np
import pandas as pd
import pytest

import config
from monitor import daily, ledger


def test_last_final_date_waits_for_the_close():
    mon = dt.datetime(2026, 11, 23, 10, 0, tzinfo=dt.timezone.utc)
    assert ledger.last_final_date(mon) == pd.Timestamp("2026-11-22")          # before the close: yesterday
    assert ledger.last_final_date(mon.replace(hour=22, minute=45)) == pd.Timestamp("2026-11-23")
    sat = dt.datetime(2026, 11, 21, 23, 0, tzinfo=dt.timezone.utc)
    assert ledger.last_final_date(sat) == pd.Timestamp("2026-11-20")         # weekend: never "today"


def test_forward_nav_starts_at_100_and_ignores_the_past():
    idx = pd.bdate_range("2026-11-16", periods=8)
    ret = pd.Series(0.01, index=idx)
    start = pd.Timestamp("2026-11-20")
    nav = ledger.forward_nav(ret, start)
    assert nav.iloc[0] == 100.0 and nav.index[0] == start
    after = (idx > start).sum()
    assert len(nav) == after + 1
    assert nav.iloc[-1] == pytest.approx(100 * 1.01 ** after)


def test_replace_rows_never_duplicates_a_day_and_drops_stale_tickers():
    old = pd.DataFrame({"date": ["2026-11-20"] * 2, "book": ["m"] * 2, "ticker": ["A", "B"], "weight": [.5, .5]})
    new = pd.DataFrame({"date": ["2026-11-20"], "book": ["m"], "ticker": ["A"], "weight": [1.0]})
    out = ledger.replace_rows(old, new, ["date", "book"])
    assert out["ticker"].tolist() == ["A"]                       # B was removed because the day was recomputed
    other = pd.DataFrame({"date": ["2026-11-19"], "book": ["m"], "ticker": ["C"], "weight": [1.0]})
    assert len(ledger.replace_rows(out, other, ["date", "book"])) == 2   # a different day is kept


def test_data_health_flags_lagging_and_jumping_tickers():
    idx = pd.bdate_range("2026-10-01", periods=40)
    px = pd.DataFrame({"OK": np.linspace(100, 110, 40), "OLD": np.linspace(50, 55, 40), "JUMP": np.linspace(10, 11, 40)}, index=idx)
    px.loc[idx[-10:], "OLD"] = np.nan
    px.loc[idx[-2], "JUMP"] = 20.0
    msgs = " | ".join(ledger.data_health(px))
    assert "OLD" in msgs and "behind" in msgs and "JUMP" in msgs and "OK" not in msgs


def test_weight_changes():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    cur = pd.Series({"A": 0.5, "C": 0.5, "B": 0.0})
    entries, exits, turn = ledger.weight_changes(prev, cur)
    assert entries == ["C"] and exits == ["B"] and turn == pytest.approx(0.5)


def _fake_book(last="2026-11-20", n=10):
    idx = pd.bdate_range(end=last, periods=n)
    return {"ret": pd.Series(0.001, index=idx), "bench": pd.Series(0.0005, index=idx),
            "weights": pd.Series({"AAA": 0.6, "BBB": 0.4, "CCC": 0.0}), "label": "Fake book"}


def test_write_outputs_is_idempotent_and_flags_problems(tmp_path):
    end = pd.Timestamp("2026-11-20")
    books = {"fake": _fake_book()}
    p1 = daily.write_outputs(books, [], [], end, out=tmp_path)
    first = (tmp_path / "paper_nav.csv").read_text()
    p2 = daily.write_outputs(books, [], [], end, out=tmp_path)               # same day again
    assert p1 == [] and p2 == []
    assert (tmp_path / "paper_nav.csv").read_text() == first
    sig = pd.read_csv(tmp_path / "signals.csv")
    assert sig["ticker"].tolist() == ["AAA", "BBB"]                          # zero weights are not logged
    assert "Fake book" in (tmp_path / "latest.md").read_text()
    assert "stale" in " ".join(daily.write_outputs(books, [], [], pd.Timestamp("2026-12-20"), out=tmp_path))
    assert daily.write_outputs({}, ["Momentum: skipped (boom)"], [], end, out=tmp_path)    # a skipped book is a problem
