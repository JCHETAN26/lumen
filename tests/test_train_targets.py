"""Training targets (lumen.train.targets): the most common human answer, quirks counted."""

from __future__ import annotations

import pytest

from lumen.train.targets import build_targets, target_answer


def rec(answers: list[str], answerable: int = 1, qid: str = "VizWiz_train_00000001") -> dict:
    return {
        "question_id": qid,
        "answers": [{"answer": a, "answer_confidence": "yes"} for a in answers],
        "answerable": answerable,
    }


def test_majority_answer() -> None:
    assert target_answer(rec(["dog"] * 6 + ["cat"] * 4)) == "dog"


def test_exact_tie_is_alphabetical() -> None:
    assert target_answer(rec(["red"] * 5 + ["blue"] * 5)) == "blue"


def test_unanswerable_label_always_gives_unanswerable() -> None:
    assert target_answer(rec(["dog"] * 6 + ["unanswerable"] * 4, answerable=0)) == "unanswerable"


def test_never_switches_to_a_minority_answer_for_metric_credit() -> None:
    """'the dog' normalizes to 'dog' and matches no raw ground truth, so the metric scores it 0
    while 'cat' would score 1. The target stays 'the dog' (right content), and is counted."""
    pairs, stats = build_targets([rec(["the dog"] * 6 + ["cat"] * 4)])
    assert pairs[0][1] == "the dog" and stats.zero_score_targets == 1


def test_build_targets_counts_and_ablation() -> None:
    records = [
        rec(["dog"] * 10, qid="q1"),
        rec(["red"] * 5 + ["blue"] * 5, qid="q2"),
        rec(["unanswerable"] * 10, answerable=0, qid="q3"),
        rec(["three"] * 10, qid="q4"),  # predicted "three" is scored as "3": no raw match
    ]
    pairs, stats = build_targets(records)
    assert [t for _, t in pairs] == ["dog", "blue", "unanswerable", "three"]
    assert stats.examples == 4 and stats.unanswerable_targets == 1
    assert stats.tied_majority == 1 and stats.zero_score_targets == 1
    pairs, stats = build_targets(records, include_unanswerable=False)
    assert [r["question_id"] for r, _ in pairs] == ["q1", "q2", "q4"]
    assert stats.dropped_unanswerable == 1 and stats.unanswerable_targets == 0


def test_empty_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="no training examples"):
        build_targets([])
    with pytest.raises(ValueError, match="empty training target"):
        build_targets([rec(["  "] * 10)])
