"""Settings shared by every strategy. Change them here, once."""
import pandas as pd

START_CAPITAL = 100.0                    # every strategy starts with $100
TRAIN_END = pd.Timestamp("2025-12-31")   # up to here = TRAIN (we may look at it to choose settings)
TEST_START = pd.Timestamp("2026-01-01")  # from here  = TEST  (honest check, never used to choose settings)
