from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumen.data.annotations import AnnotationError, load_annotations, parse_annotations, summarize


def rec(
    i: int,
    split: str = "train",
    answerable: int = 1,
    answer_type: str = "other",
    answers: list[str] | None = None,
) -> dict:
    answers = answers or ["dog"] * 10
    return {
        "image": f"VizWiz_{split}_{i:08d}.jpg",
        "question": "What is this?",
        "answers": [{"answer": a, "answer_confidence": "yes"} for a in answers],
        "answer_type": answer_type,
        "answerable": answerable,
    }


def test_parse_adds_question_id_and_split() -> None:
    [r] = parse_annotations([rec(7)], "train")
    assert r["question_id"] == "VizWiz_train_00000007"
    assert r["split"] == "train"
    assert r["answerable"] == 1 and len(r["answers"]) == 10


def test_load_from_file(tmp_path: Path) -> None:
    p = tmp_path / "val.json"
    p.write_text(json.dumps([rec(0, "val"), rec(1, "val", 0, "unanswerable")]))
    assert [r["answerable"] for r in load_annotations(p, "val")] == [1, 0]


def test_questions_only_file_needs_no_answers() -> None:
    [r] = parse_annotations(
        [{"image": "VizWiz_test_00000000.jpg", "question": "Q?"}], "test", require_answers=False
    )
    assert "answers" not in r


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(image="VizWiz_val_00000000.jpg"),  # wrong split prefix
        lambda r: r.update(image="x.png"),
        lambda r: r.update(question=" "),
        lambda r: r.update(answers=[]),
        lambda r: r["answers"][0].update(answer_confidence="certain"),
        lambda r: r["answers"][0].update(answer=None),
        lambda r: r.update(answer_type="color"),
        lambda r: r.update(answerable=2),
        lambda r: r.pop("answerable"),
    ],
)
def test_malformed_records_raise(mutate) -> None:
    r = copy.deepcopy(rec(0))
    mutate(r)
    with pytest.raises(AnnotationError):
        parse_annotations([r], "train")


def test_duplicate_image_raises() -> None:
    with pytest.raises(AnnotationError, match="more than one question"):
        parse_annotations([rec(0), rec(0)], "train")


def test_empty_and_unknown_split_raise() -> None:
    with pytest.raises(AnnotationError):
        parse_annotations([], "train")
    with pytest.raises(AnnotationError):
        parse_annotations([rec(0)], "dev")


def test_summarize_counts_unmatchable_ground_truth() -> None:
    recs = parse_annotations(
        [
            rec(0, answers=["two"] * 10),
            rec(1, answerable=0, answer_type="unanswerable", answers=["unanswerable"] * 10),
        ],
        "train",
    )
    s = summarize(recs)
    assert s["answerable_rate"] == 0.5
    assert s["answer_type_counts"] == {"other": 1, "unanswerable": 1}
    assert s["gt_answers_not_in_normalized_form"] == 10  # "two" normalizes to "2"
    assert s["answers_per_question"] == {"10": 2}
