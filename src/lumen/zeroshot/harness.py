"""Run a backend over VizWiz questions (resumable), and score the predictions officially.

Predictions go to a JSONL file: a header line fixing everything that determines the outputs
(model, revision, prompt and its version, extractor version, decoding), then one line per
question, flushed and fsynced as each batch completes. A rerun resumes from that file, and
refuses to if the header differs, so outputs from different settings are never mixed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lumen.data.annotations import Record
from lumen.data.images import ZipImages
from lumen.eval.answerability import best_f1_threshold, evaluate_answerability
from lumen.eval.bootstrap import bootstrap_ci, mean_ci
from lumen.eval.vizwiz_metric import evaluate_accuracy
from lumen.zeroshot.backends import Backend
from lumen.zeroshot.extract import EXTRACTOR_VERSION, UNANSWERABLE, extract_answer
from lumen.zeroshot.prompts import PROMPT_VERSION, PROMPTS, chat_messages


class HeaderMismatch(RuntimeError):
    pass


def question_ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def predictions_header(backend: Backend, prompt_key: str, split: str) -> dict[str, Any]:
    d = backend.describe()
    return {
        "model": {k: d.get(k) for k in ("model_key", "repo", "revision", "stand_in", "backend")},
        "generation": d.get("generation"),
        "prompt_key": prompt_key,
        "prompt_template": PROMPTS[prompt_key],
        "prompt_version": PROMPT_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "split": split,
    }


def read_predictions(path: Path) -> tuple[dict | None, dict[str, dict]]:
    if not path.exists():
        return None, {}
    header, preds = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # a torn last line from a killed session: drop it, rerun that batch
            if "_header" in row:
                header = row["_header"]
            else:
                preds[row["question_id"]] = row
    return header, preds


@dataclass
class RunStats:
    new: int
    resumed: int
    seconds: float
    batches: int


def run_predictions(
    backend: Backend,
    records: Sequence[Record],
    images: ZipImages,
    *,
    prompt_key: str,
    split: str,
    out_path: Path,
    batch_size: int,
    on_batch: Callable[[int, float], None] | None = None,
) -> tuple[dict[str, dict], RunStats]:
    """Answer every record not already in `out_path`. `on_batch(done, seconds)` after each batch.

    `on_batch` may raise (e.g. the budget guard after calibration) to stop cleanly: everything
    answered so far is already on disk.
    """
    header = predictions_header(backend, prompt_key, split)
    old_header, preds = read_predictions(out_path)
    if old_header is not None and old_header != header:
        raise HeaderMismatch(
            f"{out_path} was written with different settings; use a new file.\n"
            f"old: {json.dumps(old_header, sort_keys=True)}\n"
            f"new: {json.dumps(header, sort_keys=True)}"
        )
    wanted = {r["question_id"] for r in records}
    stale = set(preds) - wanted
    if stale:
        raise HeaderMismatch(f"{out_path} has {len(stale)} questions outside this split")
    todo = [r for r in records if r["question_id"] not in preds]
    resumed = len(preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0, done, batches = time.monotonic(), 0, 0
    # Rewrite a clean file first if the last line was torn, so appends stay valid JSONL.
    if old_header is not None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_header": header}) + "\n")
            for row in preds.values():
                f.write(json.dumps(row) + "\n")
    with open(out_path, "a", encoding="utf-8") as f:
        if old_header is None:
            f.write(json.dumps({"_header": header}) + "\n")
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            imgs = [images.open(r["image"]) for r in chunk]
            msgs = [chat_messages(r["question"], prompt_key) for r in chunk]
            outs = backend.run_batch(imgs, msgs)
            for r, o in zip(chunk, outs, strict=True):
                ex = extract_answer(o.raw)
                row = {
                    "question_id": r["question_id"],
                    "raw": o.raw,
                    "answer": ex.answer,
                    "abstained_by_rule": ex.abstained_by_rule,
                    "p_unanswerable": o.p_unanswerable,
                }
                preds[r["question_id"]] = row
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += len(chunk)
            batches += 1
            if on_batch is not None:
                on_batch(done, time.monotonic() - t0)
    return preds, RunStats(done, resumed, time.monotonic() - t0, batches)


def score_predictions(
    records: Sequence[Record], preds: dict[str, dict], *, bootstrap_resamples: int = 1000
) -> dict[str, Any]:
    """Official accuracy and answerability, with bootstrap CIs over questions."""
    acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
    ids = [r["question_id"] for r in records]
    per_q = np.array([acc.per_question[q] for q in ids])
    y = np.array([r["answerable"] for r in records])
    s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
    ans = evaluate_answerability(y, s)

    def ap(idx: np.ndarray) -> float:
        return evaluate_answerability(y[idx], s[idx]).average_precision

    said_unans = np.array([preds[q]["answer"].strip().lower() == UNANSWERABLE for q in ids])
    t, f1_best = best_f1_threshold(y, s)
    return {
        "n": acc.n,
        "question_ids_sha256": question_ids_sha256(ids),
        "accuracy": {
            "overall": acc.overall,
            "overall_official": acc.overall_official,
            "ci95": mean_ci(100 * per_q, resamples=bootstrap_resamples),
            "per_answer_type": acc.per_answer_type,
            "per_answer_type_n": acc.per_answer_type_n,
            "answerable_only": float(100 * per_q[y == 1].mean()) if (y == 1).any() else None,
            "unanswerable_only": float(100 * per_q[y == 0].mean()) if (y == 0).any() else None,
        },
        "answerability": {
            "average_precision": ans.average_precision,
            "average_precision_official": ans.average_precision_official,
            "ap_ci95": bootstrap_ci(len(ids), ap, resamples=bootstrap_resamples),
            "f1_at_0_5_official": ans.f1_at_0_5_official,
            "positive_rate": float((y == 0).mean()),  # AP of a random ranker
            # Descriptive only on this split; a threshold is *chosen* on dev and applied elsewhere.
            "oracle_threshold_this_split": {"threshold": t, "f1": f1_best},
        },
        "said_unanswerable_rate": float(said_unans.mean()),
        "abstained_by_rule_rate": float(np.mean([preds[q]["abstained_by_rule"] for q in ids])),
    }
