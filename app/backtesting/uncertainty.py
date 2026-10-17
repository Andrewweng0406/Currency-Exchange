from __future__ import annotations

import math

import numpy as np


def block_mean_interval(values, block_size: int, repeats: int = 1000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or block_size < 1:
        raise ValueError("A block interval requires at least two finite observations and positive block size")
    block_size = min(block_size, len(values))
    rng = np.random.default_rng(20261016)
    estimates = []
    for _ in range(repeats):
        starts = rng.integers(0, len(values), math.ceil(len(values) / block_size))
        indices = ((starts[:, None] + np.arange(block_size)) % len(values)).ravel()[:len(values)]
        estimates.append(values[indices].mean())
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)
