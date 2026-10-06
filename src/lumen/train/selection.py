"""Stage 3 choices, made on dev only: the training config and epoch, and the F1 threshold.

Rule, fixed 2026-10-06 before any fine-tuning result existed (`RULE`): among the sweep runs (the
default seed, unanswerable examples included), each evaluated at each epoch's adapter on all 2,052
dev questions, choose the (run, epoch) with the highest official dev accuracy; ties go to higher
dev AP, then fewer training steps, then the run name. Seed and ablation runs never compete: they
re-train the chosen config and are evaluated at the chosen epoch.

Why accuracy decides: it is VizWiz's headline metric, and "unanswerable" is a legitimate VizWiz
answer the model is trained to give. The resolution probe showed that accuracy can reward
abstaining, so each candidate's credit from real answers vs from "unanswerable" on answerable
questions is reported next to it (`lumen.zeroshot.resolution.abstention_breakdown`), not decisive.

The answerability threshold for F1 is chosen per (run, epoch) on that run's dev predictions
(`lumen.eval.answerability.best_f1_threshold`) and applied unchanged to val.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

RULE = (
    "chosen = the sweep (run, epoch) with the highest official accuracy on all of dev; ties: "
    "higher dev AP, then fewer steps, then run name. Sweep = default seed, unanswerable examples "
    "included. Seed and ablation runs re-train the chosen config. Fixed 2026-10-06, before any "
    "fine-tuning result existed."
)

# Fields of a run's spec that seed and ablation runs may change; everything else must match the
# chosen config for a run to count as "the chosen config" (and be allowed on val).
VARIES_ACROSS_REPLICATES = ("seed", "include_unanswerable")
# Operational settings that never change what a run learns, so they are not in the spec.
NOT_IN_SPEC = ("checkpoint_every", "keep_checkpoints", "dataloader_workers")


def spec_sha256(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# Derived from the example count, which the ablation changes (it drops unanswerable examples).
DERIVED_FROM_DATA = ("steps_per_epoch", "max_steps", "schedule_steps")


def family(spec: Mapping[str, Any]) -> dict[str, Any]:
    """A run's spec without the fields replicates vary: the config a seed or ablation re-trains."""
    skip = VARIES_ACROSS_REPLICATES + DERIVED_FROM_DATA
    cfg = {k: v for k, v in spec["config"].items() if k not in skip}
    rest = {k: v for k, v in spec.items() if k != "config" and k not in DERIVED_FROM_DATA}
    return {**rest, "config": cfg}


def is_sweep_run(spec: Mapping[str, Any], sweep_seed: int) -> bool:
    cfg = spec["config"]
    return cfg["seed"] == sweep_seed and bool(cfg["include_unanswerable"])


def choose_finetune(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`candidates`: one per (run, epoch) dev result, each with run, epoch, step, accuracy,
    average_precision. Returns the chosen one and the full ranking (best first)."""
    if not candidates:
        raise ValueError("no dev results for sweep runs: evaluate each sweep run's epochs on dev")
    keys = [(c["run"], c["epoch"]) for c in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate (run, epoch) candidates: {sorted(keys)}")
    ranking = sorted(
        candidates,
        key=lambda c: (-c["accuracy"], -c["average_precision"], c["step"], c["run"], c["epoch"]),
    )
    return {"rule": RULE, "chosen": dict(ranking[0]), "ranking": [dict(c) for c in ranking]}
