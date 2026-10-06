"""Latency summaries and answer agreement for the on-device benchmarks.

Latency is reported as median and p90 over repeated runs after warm-up (never a single run): the
M2 Air is fanless, so one number can't separate the model from the machine's thermal state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    if not v.size or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("latencies must be a non-empty list of finite, non-negative numbers")
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def agreement(a: Mapping[str, str], b: Mapping[str, str]) -> dict[str, Any]:
    """How often two systems give the same extracted answer on the same questions (case and
    surrounding space ignored). Only the rates are returned, never the answers themselves."""
    if set(a) != set(b) or not a:
        raise ValueError("agreement needs the same, non-empty question set on both sides")

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    same = [norm(a[q]) == norm(b[q]) for q in sorted(a)]
    unans_a = [norm(a[q]) == "unanswerable" for q in sorted(a)]
    unans_b = [norm(b[q]) == "unanswerable" for q in sorted(a)]
    return {
        "n": len(same),
        "exact_answer_agreement": float(np.mean(same)),
        "abstention_agreement": float(np.mean(np.array(unans_a) == np.array(unans_b))),
    }
