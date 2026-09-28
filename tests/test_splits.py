from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from lumen.data.splits import (
    _allocate,
    make_dev_split,
    make_mac_sample,
    read_ids,
    write_ids,
)


def records(n: int, split: str = "train") -> list[dict]:
    types = ["other", "other", "yes/no", "number", "unanswerable"]
    out = []
    for i in range(n):
        t = types[i % len(types)]
        out.append(
            {
                "question_id": f"VizWiz_{split}_{i:08d}",
                "answer_type": t,
                "answerable": 0 if t == "unanswerable" else 1,
            }
        )
    return out


def test_dev_split_is_disjoint_complete_and_sized() -> None:
    train = records(1000)
    fit, dev = make_dev_split(train, 0.1, seed=1)
    assert len(dev) == 100
    assert not set(fit) & set(dev)
    assert set(fit) | set(dev) == {r["question_id"] for r in train}


def test_dev_split_preserves_answerable_rate() -> None:
    train = records(1000)
    _, dev = make_dev_split(train, 0.1, seed=1)
    by_id = {r["question_id"]: r for r in train}
    assert sum(1 - by_id[q]["answerable"] for q in dev) == 20  # 20% unanswerable, like the pool


def test_selection_is_deterministic_and_order_independent() -> None:
    train = records(500)
    shuffled = train[:]
    random.Random(0).shuffle(shuffled)
    assert make_dev_split(train, 0.2, 7) == make_dev_split(shuffled, 0.2, 7)
    assert make_dev_split(train, 0.2, 7)[1] != make_dev_split(train, 0.2, 8)[1]


def test_mac_sample_exact_size_and_all_strata() -> None:
    val = records(4319, "val")
    sample = make_mac_sample(val, 300, seed=3)
    assert len(sample) == len(set(sample)) == 300
    by_id = {r["question_id"]: r for r in val}
    got = Counter((by_id[q]["answerable"], by_id[q]["answer_type"]) for q in sample)
    assert set(got) == {(1, "other"), (1, "yes/no"), (1, "number"), (0, "unanswerable")}


def test_allocate_sums_exactly() -> None:
    alloc = _allocate({"a": 7, "b": 7, "c": 7}, 10)
    assert sum(alloc.values()) == 10 and max(alloc.values()) - min(alloc.values()) <= 1
    with pytest.raises(ValueError):
        _allocate({"a": 3}, 4)


def test_ids_round_trip_and_reject_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "ids.txt"
    write_ids(p, ["b", "a"])
    assert read_ids(p) == ["b", "a"]
    p.write_text("a\na\n")
    with pytest.raises(ValueError):
        read_ids(p)
