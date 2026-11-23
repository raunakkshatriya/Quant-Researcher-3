"""Settings shared by every strategy. Change them here, once."""
import pandas as pd

START_CAPITAL = 100.0                    # every strategy starts with $100
TRAIN_END = pd.Timestamp("2025-12-31")   # up to here = TRAIN (we may look at it to choose settings)
TEST_START = pd.Timestamp("2026-01-01")  # from here  = TEST  (honest check, never used to choose settings)

# Execution delay: signal is computed with the close of day t; we trade at the close of day
# t + EXEC_DELAY_DAYS. 1 = "see the close, place the order, it fills at tomorrow's close".
# 0 = "trade at the very close we used for the signal" (too optimistic, only for teaching demos).
EXEC_DELAY_DAYS = 1

# Crisis windows: "black swan" periods we can report WITH and WITHOUT (dates are approximate:
# roughly the peak-to-trough of each event). These are used for REPORTING ONLY - never to rank
# or select settings, and never to hide results. Add your own, e.g. "2022 rate shock".
CRISIS_WINDOWS = {
    "dot-com bust": ("2000-03-10", "2002-10-09"),
    "global financial crisis": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-04-07"),
}

# Uninvested cash earns the 3-month T-bill rate (FRED). Set False to compare with the old 0% assumption.
USE_CASH_YIELD = True

# Daily monitor (monitor/ folder, GitHub Actions): forward paper tracking of the ACTIVE books.
# Every book is marked at $100 at the close of PAPER_START and then followed day by day. The monitor only RECORDS;
# nothing is tuned or judged from it between scheduled reviews (see monitor/README.md).
PAPER_START = pd.Timestamp("2026-11-20")
