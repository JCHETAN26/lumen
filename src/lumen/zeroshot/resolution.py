"""Choosing the image resolution for fine-tuning from the dev probe (all 2,052 dev questions).

Rule, fixed 2026-10-03 before any probe result: take the arm with the best accuracy, then choose
the lowest-token resolution (smallest long edge) that is not meaningfully worse than it: the lower
bound of the 95% paired bootstrap CI on its accuracy difference from the best arm is no lower than
-MARGIN_POINTS. Average precision is reported alongside but does not decide.

The 2-point margin is a product decision made before any result: 1,024 px uses ~2.6x fewer image
tokens than 2,048 px, cutting training cost and on-device latency, and up to 2 accuracy points is
an accepted price for that.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lumen.eval.bootstrap import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci

MARGIN_POINTS = 2.0
RULE = (
    "best = highest dev accuracy (all of dev; ties: fewer image tokens); chosen = the smallest "
    f"long edge whose 95% paired bootstrap CI on (accuracy - best accuracy) has a lower bound >= "
    f"-{MARGIN_POINTS:g} points. AP is reported, not decisive. Fixed 2026-10-03, before any probe "
    "result existed."
)


def choose_long_edge(
    per_question: Mapping[int, np.ndarray],
    *,
    margin: float = MARGIN_POINTS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """`per_question[long_edge]` = per-question accuracy (0-100) on the same questions, same order.

    Returns the best arm, the chosen arm, and each arm's difference from the best with its CI.
    """
    if not per_question:
        raise ValueError("no arms")
    sizes = {len(a) for a in per_question.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError(f"arms must score the same, non-empty question set; sizes {sorted(sizes)}")
    n = sizes.pop()
    means = {le: float(np.mean(a)) for le, a in per_question.items()}
    best = max(means, key=lambda le: (means[le], -le))
    a_best = np.asarray(per_question[best], dtype=float)
    vs_best: dict[int, dict[str, Any]] = {}
    for le, arr in per_question.items():
        if le == best:
            continue
        a = np.asarray(arr, dtype=float)
        ci = bootstrap_ci(
            n,
            lambda i, a=a: float(a[i].mean() - a_best[i].mean()),
            resamples=resamples,
            seed=seed,
        )
        vs_best[le] = {
            "accuracy_diff": means[le] - means[best],
            "accuracy_diff_ci95": ci,
            "within_margin": ci["low"] >= -margin,
        }
    qualifying = [best] + [le for le, v in vs_best.items() if v["within_margin"]]
    return {
        "best": best,
        "chosen": min(qualifying),
        "margin_points": margin,
        "vs_best": vs_best,
    }
