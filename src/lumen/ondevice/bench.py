"""Stage 4 measurements on the Mac: accuracy on the pinned sample, latency, and their summary.

Backend-agnostic (any `lumen.zeroshot.backends.Backend` with `timed_answer`), so the whole flow is
tested on Linux with a fake; on the Mac the backend is `lumen.ondevice.mlx_backend.MLXBackend`.

- Accuracy and answerability: the official metric on all 300 Mac-sample questions, through the
  same resumable harness as Kaggle. Prediction files stay in the gitignored `predictions/`; results
  record only scores and the file's sha256 (answers can repeat private details from the photos).
- Latency: a fixed subset (the first N questions by ID), answered once untimed to warm up, then
  `repeats` timed passes. Median and p90 per metric; a fanless M2 throttles, so the thermal state
  and power source before and after are recorded, and a run refuses to start while throttled.
- Summary: every quantized variant against MLX fp16 on the same questions (paired bootstrap CIs on
  accuracy and AP, answer agreement), with size on disk, latency, and memory.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lumen.data.annotations import Record
from lumen.eval.paired import paired_difference
from lumen.eval.vizwiz_metric import evaluate_accuracy
from lumen.ondevice.stats import agreement, latency_summary
from lumen.zeroshot.harness import read_predictions, run_predictions, score_predictions
from lumen.zeroshot.prompts import chat_messages

SPLIT = "mac_sample"
LATENCY_METRICS = ("ttft_s", "total_s", "preprocess_s", "generation_tps", "prompt_tps")


class Throttled(RuntimeError):
    pass


def wait_until_cool(
    read_limit: Callable[[], int | None],
    *,
    max_wait_s: float,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait until the CPU speed limit (`pmset -g therm`) is back to 100%, up to `max_wait_s`.
    None (no reading: not a Mac, or pmset said nothing) counts as not throttled."""
    waited = 0.0
    limit = read_limit()
    while limit is not None and limit < 100:
        if waited >= max_wait_s:
            raise Throttled(
                f"CPU speed limit is {limit}% after waiting {waited:.0f} s: let the Mac cool "
                "(lid open, nothing else running) and rerun, or pass --allow-throttled"
            )
        sleep(poll_s)
        waited += poll_s
        limit = read_limit()
    return {"cpu_speed_limit_pct": limit, "waited_s": waited}


def accuracy_run(
    backend: Any,
    records: Sequence[Record],
    images: Any,
    *,
    prompt_key: str,
    pred_path: Path,
) -> dict[str, Any]:
    """Answer every sample question (resumable) and score it officially."""
    preds, stats = run_predictions(
        backend,
        records,
        images,
        prompt_key=prompt_key,
        split=SPLIT,
        out_path=pred_path,
        batch_size=1,
    )
    return {
        "metrics": score_predictions(records, preds),
        "run": {
            "new": stats.new,
            "resumed": stats.resumed,
            "seconds": round(stats.seconds, 1),
            "sec_per_question": round(stats.seconds / stats.new, 4) if stats.new else None,
        },
        "predictions": {
            "local_file": str(pred_path.name),
            "sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
            "stored": "local only (Mac, gitignored predictions/; never uploaded or committed)",
        },
    }


def latency_run(
    backend: Any,
    records: Sequence[Record],
    images: Any,
    *,
    prompt_key: str,
    n: int,
    repeats: int,
    warmup: int = 2,
) -> dict[str, Any]:
    """Median and p90 of each latency metric over `repeats` passes of the first `n` questions."""
    subset = list(records)[:n]
    if not subset or repeats < 1:
        raise ValueError("latency needs at least one question and one repeat")

    def answer(r: Record) -> dict[str, Any]:
        return backend.timed_answer(
            images.open(r["image"]), chat_messages(r["question"], prompt_key)
        )

    for r in subset[:warmup]:  # compiles Metal kernels; not timed
        answer(r)
    runs = [{"repeat": rep, **answer(r)} for rep in range(repeats) for r in subset]
    out: dict[str, Any] = {
        "questions": len(subset),
        "repeats": repeats,
        "warmup_questions": min(warmup, len(subset)),
    }
    for m in LATENCY_METRICS:
        vals = [x[m] for x in runs if x.get(m) is not None]
        out[m] = latency_summary(vals) if vals else None
    out["prompt_tokens"] = latency_summary([x["prompt_tokens"] for x in runs])
    out["generation_tokens"] = latency_summary([x["generation_tokens"] for x in runs])
    mem = [x["peak_memory_gb"] for x in runs if x.get("peak_memory_gb") is not None]
    out["peak_memory_gb_max"] = max(mem) if mem else None
    # Repeat-to-repeat drift shows throttling mid-run (later passes slower on a hot fanless Mac).
    out["total_s_median_by_repeat"] = [
        float(np.median([x["total_s"] for x in runs if x["repeat"] == k])) for k in range(repeats)
    ]
    return out


def _per_question(records: Sequence[Record], preds: Mapping[str, dict]) -> tuple:
    ids = [r["question_id"] for r in records]
    if set(preds) != set(ids):
        raise ValueError("predictions don't cover exactly the sample's questions")
    acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
    a = np.array([100 * acc.per_question[q] for q in ids])
    s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
    return a, s, {q: preds[q]["answer"] for q in ids}


def summarize(
    records: Sequence[Record],
    benches: Mapping[str, dict[str, Any]],
    pred_files: Mapping[str, Path],
    *,
    reference: str = "fp16",
    kaggle_preds: Mapping[str, dict] | None = None,
) -> dict[str, Any]:
    """Each variant vs the MLX `reference` on the same questions; optionally vs Kaggle fp16."""
    if reference not in benches:
        raise ValueError(f"no {reference} benchmark to compare against")
    y = np.array([r["answerable"] for r in records])
    scored = {}
    for name, bench in benches.items():
        path = pred_files[name]
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != bench["accuracy"]["predictions"]["sha256"]
        ):
            raise ValueError(f"{path.name} changed since its benchmark result was written")
        scored[name] = _per_question(records, read_predictions(path)[1])
    a0, s0, ans0 = scored[reference]
    kaggle = None
    if kaggle_preds is not None:
        ids = {r["question_id"] for r in records}
        kaggle = _per_question(records, {q: p for q, p in kaggle_preds.items() if q in ids})
    rows = []
    for name, bench in benches.items():
        a, s, ans = scored[name]
        m = bench["accuracy"]["metrics"]
        lat = bench.get("latency") or {}
        row: dict[str, Any] = {
            "variant": name,
            "size_on_disk_bytes": bench.get("size_on_disk_bytes"),
            "accuracy": m["accuracy"]["overall"],
            "accuracy_ci95": m["accuracy"]["ci95"],
            "average_precision": m["answerability"]["average_precision"],
            "said_unanswerable_rate": m["said_unanswerable_rate"],
            "ttft_s": lat.get("ttft_s"),
            "total_s": lat.get("total_s"),
            "generation_tps": lat.get("generation_tps"),
            "peak_memory_gb_max": lat.get("peak_memory_gb_max"),
            "load_seconds": bench.get("load_seconds"),
        }
        if name != reference:
            row[f"vs_{reference}"] = paired_difference(y, a, a0, s, s0)
            row[f"agreement_with_{reference}"] = agreement(ans, ans0)
        if kaggle is not None:
            ka, ks, kans = kaggle
            row["vs_kaggle_fp16"] = paired_difference(y, a, ka, s, ks)
            row["agreement_with_kaggle_fp16"] = agreement(ans, kans)
        rows.append(row)
    return {"n": len(records), "reference": reference, "variants": rows}
