"""Training targets: one answer string per question, in VizWiz's own vocabulary.

The target is the most common human answer (the VizWiz convention; exact ties: alphabetical). A
metric-aware alternative, picking whichever human answer the official metric scores highest, was
rejected before any training: forms of the same answer ("two" / "2") always tie, so it only ever
helps by switching to a *different*, minority answer the metric happens to credit (6x "the dog"
vs 4x "cat" -> "cat", because only predictions are normalized). That teaches wrong content to get
metric credit. Instead the targets the metric can't credit are counted (`TargetStats`), so the
effect of the metric's quirks on training is measured.

Unanswerable questions (answerable == 0) always get the literal "unanswerable", so the model learns
to abstain in VizWiz's vocabulary; `include_unanswerable=False` drops them (the Stage 3 ablation).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from lumen.data.annotations import Record
from lumen.eval.vizwiz_metric import UNANSWERABLE, vqa_accuracy

TARGET_VERSION = "v1"  # recorded in every training result; bump if the rule changes


def majority_answer(record: Record) -> str:
    counts = Counter(a["answer"] for a in record["answers"])
    return min(counts, key=lambda a: (-counts[a], a))


def target_answer(record: Record) -> str:
    return UNANSWERABLE if record["answerable"] == 0 else majority_answer(record)


@dataclass
class TargetStats:
    examples: int = 0
    unanswerable_targets: int = 0
    dropped_unanswerable: int = 0
    tied_majority: int = 0  # answerable questions whose top answer count is shared (alphabetical)
    zero_score_targets: int = 0  # answerable questions whose target the official metric scores 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_targets(
    records: Iterable[Record], *, include_unanswerable: bool = True
) -> tuple[list[tuple[Record, str]], TargetStats]:
    """(record, target) pairs in input order, and counts describing them."""
    out: list[tuple[Record, str]] = []
    stats = TargetStats()
    for r in records:
        if r["answerable"] == 0 and not include_unanswerable:
            stats.dropped_unanswerable += 1
            continue
        t = target_answer(r)
        if not t.strip():
            raise ValueError(f"{r['question_id']}: empty training target")
        stats.examples += 1
        if t == UNANSWERABLE:
            stats.unanswerable_targets += 1
        if r["answerable"] == 1:
            counts = Counter(a["answer"] for a in r["answers"]).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                stats.tied_majority += 1
            if vqa_accuracy(t, [a["answer"] for a in r["answers"]]) == 0:
                stats.zero_score_targets += 1
        out.append((r, t))
    if not out:
        raise ValueError("no training examples")
    return out, stats
