"""One chart function used by every strategy: full history on the left, test period on the right."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def plot_equity(curves, title, filename):
    """curves: {"label": result_table_with_ret_column}. Saves a PNG in output/.

    Left : growth of $100 from each strategy's first day (log scale).
    Right: growth of $100 starting on the first TEST day (2026-01-01), plain scale.
    """
    Path("output").mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    for label, res in curves.items():
        eq = config.START_CAPITAL * (1 + res["ret"]).cumprod()
        a1.plot(eq.index, eq, label=label)
        test = res.loc[config.TEST_START:, "ret"]
        if len(test):
            a2.plot(test.index, config.START_CAPITAL * (1 + test).cumprod(), label=label)
    a1.set_yscale("log")
    a1.axvline(config.TEST_START, color="grey", linestyle="--")
    a1.set_title("Full history: $100 from each start date (log scale)")
    a2.set_title(f"Test period only: $100 on {config.TEST_START.date()}")
    a2.tick_params(axis="x", rotation=30)
    a1.legend(); a2.axhline(100, color="grey", linewidth=0.8)
    fig.suptitle(title)
    fig.savefig(f"output/{filename}", dpi=120, bbox_inches="tight")
    print(f"Saved picture to output/{filename}")
