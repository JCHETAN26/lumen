"""Percentile bootstrap confidence intervals over questions (fixed seed, recorded)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 20260928


def bootstrap_ci(
    n: int,
    statistic: Callable[[np.ndarray], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> dict[str, float | int]:
    """CI for `statistic(indices)`, where `indices` resamples range(n) with replacement.

    Resamples where the statistic is undefined (it raises ValueError, e.g. average precision on a
    resample with one class) are skipped and counted.
    """
    if n <= 0:
        raise ValueError("need at least one question")
    rng = np.random.default_rng(seed)
    values, skipped = [], 0
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(statistic(idx))
        except ValueError:
            skipped += 1
    if not values:
        raise ValueError("statistic undefined on every resample")
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {
        "low": float(lo),
        "high": float(hi),
        "level": level,
        "resamples": resamples,
        "skipped": skipped,
        "seed": seed,
    }


def mean_ci(values: Sequence[float], **kwargs) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return bootstrap_ci(len(arr), lambda idx: float(arr[idx].mean()), **kwargs)
