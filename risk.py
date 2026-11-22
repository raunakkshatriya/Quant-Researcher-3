"""Pre-trade risk limits. Every strategy's weights pass through here before being traded.

Industry habit: the strategy proposes, the risk layer disposes. Limits live in one place so a
bug in a strategy cannot silently take a giant position.
"""
import numpy as np


def apply_limits(weights, max_position=None, max_gross=1.0):
    """Clip each weight to +-max_position, then scale each day so total |weights| <= max_gross."""
    w = weights.copy()
    if max_position is not None:
        w = w.clip(lower=-max_position, upper=max_position)
    gross = w.abs().sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / gross.replace(0, np.nan), 1.0)
    return w.mul(np.nan_to_num(scale, nan=1.0), axis=0)
