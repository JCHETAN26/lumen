"""Load and validate VizWiz-VQA annotation files.

Official record schema (train.json, val.json, and the released VQA_test.json):

    {"image": "VizWiz_train_00000000.jpg", "question": "...",
     "answers": [{"answer": "...", "answer_confidence": "yes"|"maybe"|"no"}, ... x10],
     "answer_type": "other"|"yes/no"|"number"|"unanswerable", "answerable": 0|1}

Lumen adds `question_id` (the image file stem; VizWiz has one question per image) and `split`.
Loading is strict: a malformed record raises, so a changed upstream file can't slip through.

Test answers are sealed (see lumen.eval.test_seal): loading test records *with answers* requires a
`FinalTestRun`, which only the single, recorded final evaluation can obtain.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lumen.eval.test_seal import FinalTestRun, TestSealed
from lumen.eval.vizwiz_metric import normalize_prediction

ANSWER_TYPES = frozenset({"other", "yes/no", "number", "unanswerable"})
CONFIDENCES = frozenset({"yes", "maybe", "no"})
SPLITS = frozenset({"train", "val", "test"})

Record = dict[str, Any]


class AnnotationError(ValueError):
    pass


def _check(record: Any, index: int, split: str, require_answers: bool) -> Record:
    where = f"{split}[{index}]"
    if not isinstance(record, dict):
        raise AnnotationError(f"{where}: expected an object, got {type(record).__name__}")
    image = record.get("image")
    prefix = f"VizWiz_{split}_"
    if not isinstance(image, str) or not image.startswith(prefix) or not image.endswith(".jpg"):
        raise AnnotationError(f"{where}: image {image!r} is not a {prefix}*.jpg name")
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise AnnotationError(f"{where}: missing question")
    out: Record = {
        "question_id": image.removesuffix(".jpg"),
        "split": split,
        "image": image,
        "question": record["question"],
    }
    if not require_answers:
        return out
    answers = record.get("answers")
    if not isinstance(answers, list) or not answers:
        raise AnnotationError(f"{where}: missing answers")
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("answer"), str):
            raise AnnotationError(f"{where}: malformed answer {a!r}")
        if a.get("answer_confidence") not in CONFIDENCES:
            raise AnnotationError(f"{where}: bad answer_confidence {a.get('answer_confidence')!r}")
    if record.get("answer_type") not in ANSWER_TYPES:
        raise AnnotationError(f"{where}: bad answer_type {record.get('answer_type')!r}")
    if record.get("answerable") not in (0, 1):
        raise AnnotationError(f"{where}: answerable must be 0 or 1, got {record.get('answerable')!r}")
    out["answers"] = [
        {"answer": a["answer"], "answer_confidence": a["answer_confidence"]} for a in answers
    ]
    out["answer_type"] = record["answer_type"]
    out["answerable"] = int(record["answerable"])
    return out


def parse_annotations(
    records: Iterable[Any], split: str, *, require_answers: bool = True,
    final_test_run: FinalTestRun | None = None,
) -> list[Record]:
    """Validate raw records and return them with `question_id` and `split` added."""
    if split not in SPLITS:
        raise AnnotationError(f"unknown split {split!r}")
    if split == "test" and require_answers and not isinstance(final_test_run, FinalTestRun):
        raise TestSealed("test answers are sealed; they load only inside the final test run")
    out = [_check(r, i, split, require_answers) for i, r in enumerate(records)]
    if not out:
        raise AnnotationError(f"{split}: no records")
    dupes = [q for q, n in Counter(r["question_id"] for r in out).items() if n > 1]
    if dupes:
        raise AnnotationError(f"{split}: {len(dupes)} images have more than one question: {dupes[:3]}")
    return out


def load_annotations(
    path: Path, split: str, *, require_answers: bool = True,
    final_test_run: FinalTestRun | None = None,
) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return parse_annotations(json.load(f), split, require_answers=require_answers,
                                 final_test_run=final_test_run)


def summarize(records: list[Record]) -> dict[str, Any]:
    """Counts written into the data-stage result: the numbers a reviewer checks first."""
    n = len(records)
    if "answers" not in records[0]:
        return {"n": n, "has_answers": False}
    answer_counts = Counter(len(r["answers"]) for r in records)
    all_answers = [a["answer"] for r in records for a in r["answers"]]
    # The official metric compares normalized predictions to *raw* ground truth, so any ground
    # truth that normalization would change can never be matched exactly. Track how common it is.
    unmatchable = sum(1 for a in all_answers if normalize_prediction(a) != a)
    return {
        "n": n,
        "has_answers": True,
        "answerable_rate": round(sum(r["answerable"] for r in records) / n, 4),
        "answer_type_counts": dict(sorted(Counter(r["answer_type"] for r in records).items())),
        "answers_per_question": {str(k): v for k, v in sorted(answer_counts.items())},
        "gt_answers_not_in_normalized_form": unmatchable,
        "gt_answers_total": len(all_answers),
    }
