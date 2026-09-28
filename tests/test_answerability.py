from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.eval.answerability import (
    best_f1_threshold,
    evaluate_answerability,
    f1_at_threshold,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "official_vqaeval_golden.json").read_text()
)["answerability_case"]


def test_matches_official() -> None:
    r = evaluate_answerability(GOLDEN["answerable"], GOLDEN["answerable_score"])
    assert r.average_precision_official == GOLDEN["official"]["average_precision"]
    assert r.f1_at_0_5_official == GOLDEN["official"]["f1_score"]
    assert r.n_unanswerable == GOLDEN["answerable"].count(0)


def test_perfect_ranking_has_ap_100() -> None:
    r = evaluate_answerability([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert r.average_precision == pytest.approx(100.0)
    assert r.f1_at_0_5 == pytest.approx(100.0)


def test_positive_class_is_unanswerable() -> None:
    # Inverting the scores inverts the ranking of *unanswerable*: AP must drop.
    good = evaluate_answerability([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]).average_precision
    bad = evaluate_answerability([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]).average_precision
    assert bad < good


def test_best_threshold_beats_or_ties_fixed_0_5() -> None:
    y, s = GOLDEN["answerable"], GOLDEN["answerable_score"]
    t, f1 = best_f1_threshold(y, s)
    assert f1 >= f1_at_threshold(y, s, 0.5)
    assert f1 == pytest.approx(f1_at_threshold(y, s, t))


@pytest.mark.parametrize(
    ("y", "s"),
    [
        ([1, 0], [0.5]),
        ([1, 1], [0.5, 0.5]),
        ([1, 0], [1.5, 0.1]),
        ([1, 2], [0.1, 0.2]),
        ([1, 0], [float("nan"), 0.1]),
        ([], []),
    ],
)
def test_rejects_bad_inputs(y: list, s: list) -> None:
    with pytest.raises(ValueError):
        evaluate_answerability(y, s)
