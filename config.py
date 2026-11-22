"""Settings shared by every strategy. Change them here, once."""
import pandas as pd

START_CAPITAL = 100.0                    # every strategy starts with $100
TRAIN_END = pd.Timestamp("2025-12-31")   # up to here = TRAIN (we may look at it to choose settings)
TEST_START = pd.Timestamp("2026-01-01")  # from here  = TEST  (honest check, never used to choose settings)

# Execution delay: signal is computed with the close of day t; we trade at the close of day
# t + EXEC_DELAY_DAYS. 1 = "see the close, place the order, it fills at tomorrow's close".
# 0 = "trade at the very close we used for the signal" (too optimistic, only for teaching demos).
EXEC_DELAY_DAYS = 1
