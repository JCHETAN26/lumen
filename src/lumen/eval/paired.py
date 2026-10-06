"""Paired comparisons of two systems on the same questions, and summaries across training seeds.

Every comparison resamples *questions* once and scores both systems on the same resample, so the
CI is on the difference itself (paired), not two overlapping marginal CIs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lumen.eval.answerability import evaluate_answerability
from lumen.eval.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci


def paired_difference(
    answerable: Sequence[int],
    acc_a: Sequence[float],
    acc_b: Sequence[float],
    score_a: Sequence[float],
    score_b: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """System A minus system B on the same questions, in the same order.

    `acc_*`: per-question official accuracy (0-100). `score_*`: answerable scores (1 -
    P(unanswerable)). Returns accuracy and AP differences with 95% paired bootstrap CIs, plus the
    accuracy difference on answerable and unanswerable questions separately.
    """
    y = np.asarray(answerable, dtype=int)
    a, b = np.asarray(acc_a, dtype=float), np.asarray(acc_b, dtype=float)
    sa, sb = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    n = len(y)
    if not n or any(len(x) != n for x in (a, b, sa, sb)):
        raise ValueError("both systems must be scored on the same, non-empty question set")

    def ap(labels: np.ndarray, s: np.ndarray) -> float:
        return evaluate_answerability(labels, s).average_precision

    kw = dict(resamples=resamples, seed=seed)
    out: dict[str, Any] = {
        "n": n,
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "accuracy_diff": float(a.mean() - b.mean()),
        "accuracy_diff_ci95": bootstrap_ci(n, lambda i: float(a[i].mean() - b[i].mean()), **kw),
        "ap_a": ap(y, sa),
        "ap_b": ap(y, sb),
        "ap_diff": ap(y, sa) - ap(y, sb),
        "ap_diff_ci95": bootstrap_ci(n, lambda i: ap(y[i], sa[i]) - ap(y[i], sb[i]), **kw),
    }
    for name, mask in (("answerable", y == 1), ("unanswerable", y == 0)):
        if mask.any():
            d = a[mask] - b[mask]
            out[f"accuracy_diff_{name}"] = float(d.mean())
            out[f"accuracy_diff_{name}_ci95"] = bootstrap_ci(
                len(d), lambda i, d=d: float(d[i].mean()), **kw
            )
    return out


def seed_summary(values: Sequence[float]) -> dict[str, Any]:
    """Mean and sample standard deviation over training seeds (std is None for a single seed)."""
    v = np.asarray(values, dtype=float)
    if not v.size:
        raise ValueError("no values")
    return {
        "n_seeds": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else None,
        "min": float(v.min()),
        "max": float(v.max()),
        "values": [float(x) for x in v],
    }
