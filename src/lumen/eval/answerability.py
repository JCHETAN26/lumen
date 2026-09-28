"""The official VizWiz answerability metrics, plus the threshold selection Lumen does on dev.

Official definition (VQAEval.evaluate_unanswerability in the VizWiz API.zip): the model outputs
a score in [0, 1] that the question is *answerable*. The positive class is *unanswerable*, so
the evaluator scores `1 - score` against `1 - answerable`:

    average_precision = sklearn.average_precision_score(1 - answerable, 1 - score)
    f1                = sklearn.f1_score(1 - answerable, (1 - score) > 0.5)

Average precision is the headline number. The official F1 uses a fixed 0.5 threshold; Lumen also
reports F1 at a threshold chosen on the dev split (never on val or test), labeled as such.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


@dataclass
class AnswerabilityReport:
    """Metrics in percent; `*_official` values are rounded to 2 places like the official code."""

    n: int
    n_unanswerable: int
    average_precision: float
    average_precision_official: float
    f1_at_0_5: float
    f1_at_0_5_official: float


def _validate(answerable: Sequence[int], answerable_score: Sequence[float]) -> tuple:
    y = np.asarray(answerable, dtype=int)
    s = np.asarray(answerable_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1 or y.size == 0:
        raise ValueError(f"need two equal-length non-empty 1-D sequences, got {y.shape}, {s.shape}")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("answerable labels must be 0 or 1")
    if np.isnan(s).any() or (s < 0).any() or (s > 1).any():
        raise ValueError("answerable scores must be in [0, 1]")
    if y.min() == y.max():
        raise ValueError("average precision needs both answerable and unanswerable questions")
    return y, s


def evaluate_answerability(
    answerable: Sequence[int], answerable_score: Sequence[float]
) -> AnswerabilityReport:
    y, s = _validate(answerable, answerable_score)
    y_unans, s_unans = 1 - y, 1.0 - s
    ap = 100 * float(average_precision_score(y_unans, s_unans))
    f1 = 100 * float(f1_score(y_unans, s_unans > 0.5))
    return AnswerabilityReport(
        n=int(y.size),
        n_unanswerable=int(y_unans.sum()),
        average_precision=ap,
        average_precision_official=round(ap, 2),
        f1_at_0_5=f1,
        f1_at_0_5_official=round(f1, 2),
    )


def f1_at_threshold(
    answerable: Sequence[int], answerable_score: Sequence[float], threshold: float
) -> float:
    """F1 (percent) for flagging *unanswerable* when `1 - score > threshold`."""
    y, s = _validate(answerable, answerable_score)
    return 100 * float(f1_score(1 - y, (1.0 - s) > threshold))


def best_f1_threshold(
    answerable: Sequence[int], answerable_score: Sequence[float]
) -> tuple[float, float]:
    """The unanswerable-score threshold that maximizes F1. Call this on the dev split only.

    Candidates are midpoints between consecutive distinct scores (plus the ends), so the chosen
    threshold does not sit exactly on a dev example's score. Ties go to the lowest threshold.
    Returns (threshold, f1_percent).
    """
    y, s = _validate(answerable, answerable_score)
    u = np.unique(1.0 - s)
    candidates = np.concatenate([[0.0], (u[:-1] + u[1:]) / 2, [1.0]]) if u.size > 1 else u
    best_t, best_f1 = 0.0, -1.0
    for t in candidates:
        f1 = 100 * float(f1_score(1 - y, (1.0 - s) > t, zero_division=0))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1
