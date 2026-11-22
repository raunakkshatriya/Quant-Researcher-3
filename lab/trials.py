"""The lab notebook: every setting we try is written down, forever.

experiments/trials.csv      one row per (strategy, version, settings) we ever evaluated
experiments/test_looks.csv  one row every time we looked at the TEST period

Why: the more things you try, the more a good-looking result is just luck. The trial count
feeds the Deflated Sharpe calculation. COMMIT these two files to git so the count survives.
Whenever you change the strategy's LOGIC (not just its settings), change `version` in your
run script, e.g. "v1" -> "v2": that counts as a new batch of trials.
"""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "experiments"
TRIALS = DIR / "trials.csv"
LOOKS = DIR / "test_looks.csv"


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=DIR.parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def _append(path, rows):
    DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_trials(strategy, version, rows):
    """rows: list of dicts with keys params, train_sharpe, n_obs, data_end."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    commit = _git_hash()
    _append(TRIALS, [{"time": now, "strategy": strategy, "version": version, "commit": commit, **r}
                     for r in rows])


def trial_count(strategy):
    """Number of DISTINCT (version, settings) combinations ever tried for this strategy."""
    if not TRIALS.exists():
        return 0
    df = pd.read_csv(TRIALS)
    df = df[df["strategy"] == strategy]
    return int(df[["version", "params"]].drop_duplicates().shape[0])


def log_test_look(strategy, version, params, test_sharpe, test_return):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _append(LOOKS, [{"time": now, "strategy": strategy, "version": version, "commit": _git_hash(),
                     "params": params, "test_sharpe": test_sharpe, "test_return": test_return}])


def test_look_count(strategy):
    if not LOOKS.exists():
        return 0
    return int((pd.read_csv(LOOKS)["strategy"] == strategy).sum())
