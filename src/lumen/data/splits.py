"""Deterministic, stratified splits.

- dev: carved from official *train*. All tuning (hyperparameters, prompts, the unanswerable
  threshold) happens on dev. Official val is the reporting split and official test is used once,
  at the end; neither is ever used to choose anything.
- mac_sample: a small stratified slice of *val* for on-device tests on the Mac.

Selection orders questions by sha256(seed:question_id), not by a stateful RNG, so the result is
independent of input order and of library versions. The ID lists are committed under
configs/splits/, and those committed files, not this code, are what later stages read.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path

from lumen.data.annotations import Record


def _rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def _allocate(sizes: dict[Hashable, int], total: int) -> dict[Hashable, int]:
    """Split `total` across strata proportionally to `sizes` (largest remainder, exact sum)."""
    n = sum(sizes.values())
    if not 0 <= total <= n:
        raise ValueError(f"cannot take {total} of {n}")
    exact = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(exact[k] - alloc[k]), str(k)))
    for k in by_remainder[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified_sample(
    records: Sequence[Record], total: int, stratum: Callable[[Record], Hashable], seed: int
) -> list[str]:
    """Pick `total` question IDs, proportionally per stratum. Returns sorted IDs."""
    groups: dict[Hashable, list[str]] = defaultdict(list)
    for r in records:
        groups[stratum(r)].append(r["question_id"])
    alloc = _allocate({k: len(v) for k, v in groups.items()}, total)
    chosen: list[str] = []
    for k, ids in groups.items():
        chosen += sorted(ids, key=lambda q: _rank(seed, q))[: alloc[k]]
    return sorted(chosen)


def make_dev_split(
    train: Sequence[Record], dev_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split official train into (train_fit IDs, dev IDs), stratified on answerability."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    dev = set(stratified_sample(train, round(len(train) * dev_fraction),
                                lambda r: r["answerable"], seed))
    fit = sorted(r["question_id"] for r in train if r["question_id"] not in dev)
    return fit, sorted(dev)


def make_mac_sample(val: Sequence[Record], size: int, seed: int) -> list[str]:
    """Stratified on answerable x answer_type, so the rare 'number' type is still represented."""
    return stratified_sample(val, size, lambda r: (r["answerable"], r["answer_type"]), seed)


def write_ids(path: Path, ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{q}\n" for q in ids), encoding="utf-8")


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate IDs")
    return ids
