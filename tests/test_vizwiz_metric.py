"""The metric port must reproduce the official VizWiz vqaEval.py exactly.

tests/fixtures/official_vqaeval_golden.json was produced by running the official file (sha256
pinned in configs/pins.yaml) on the cases in tests/fixtures/make_official_golden.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.eval.vizwiz_metric import evaluate_accuracy, normalize_prediction, vqa_accuracy
from lumen.pins import load_pins

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "official_vqaeval_golden.json").read_text()
)
CASES = GOLDEN["accuracy_cases"]


def test_golden_comes_from_the_pinned_official_file() -> None:
    assert GOLDEN["source_sha256"] == load_pins().eval_code_sha256


@pytest.mark.parametrize("case", CASES, ids=[repr(c["prediction"]) for c in CASES])
def test_matches_official_per_question(case: dict) -> None:
    assert normalize_prediction(case["prediction"]) == case["normalized"]
    assert vqa_accuracy(case["prediction"], case["answers"]) == pytest.approx(case["accuracy"])


def test_matches_official_summary() -> None:
    anns = [
        {
            "question_id": f"q{i}",
            "answer_type": c["answer_type"],
            "answers": [{"answer": a} for a in c["answers"]],
        }
        for i, c in enumerate(CASES)
    ]
    preds = {f"q{i}": c["prediction"] for i, c in enumerate(CASES)}
    report = evaluate_accuracy(preds, anns)
    assert report.overall_official == GOLDEN["accuracy_summary"]["overall"]
    assert report.per_answer_type == GOLDEN["accuracy_summary"]["perAnswerType"]
    assert report.n == len(CASES)


def test_leave_one_out_arithmetic() -> None:
    # 2 of 10 say "red": leaving out a "red" annotator leaves 1 match (1/3), else 2 (2/3).
    answers = ["red", "red"] + ["blue"] * 8
    assert vqa_accuracy("red", answers) == pytest.approx((2 * (1 / 3) + 8 * (2 / 3)) / 10)
    # 4 or more matches saturate at 1 in every subset.
    assert vqa_accuracy("blue", answers) == 1.0


def test_ground_truth_is_not_normalized() -> None:
    # Official quirk kept on purpose: "two" -> "2" never equals a raw ground truth of "two".
    assert vqa_accuracy("two", ["two"] * 10) == 0.0
    assert vqa_accuracy("2", ["2"] * 10) == 1.0


def test_missing_prediction_is_an_error_not_a_zero() -> None:
    anns = [{"question_id": "a", "answer_type": "other", "answers": [{"answer": "x"}] * 10}]
    with pytest.raises(KeyError):
        evaluate_accuracy({}, anns)


def test_empty_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        vqa_accuracy("x", [])
    with pytest.raises(ValueError):
        evaluate_accuracy({}, [])
