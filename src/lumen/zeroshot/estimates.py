"""Prior GPU-time estimates for the Stage 2 zero-shot runs on a Kaggle T4.

These are priors, used only to decide whether to *start*: every run then times a calibration
batch on the real GPU and re-checks the budget with the measured rate (lumen.budget). Both the
prior and the calibrated estimate are recorded in the result, next to the hours actually used,
so these numbers get replaced by measurements after the first runs.

Derivation (per question, batch-amortized). Cost ≈ 2 prefills (one for P(unanswerable), one
inside generate) + up to 16 greedy decode steps. Prefill FLOPs ≈ 2·P_vision·(image patches) +
2·P_lm·(sequence length). A T4 delivers ~65 TFLOP/s fp16 at peak; we assume ~20 effective.

| model | image tiling (processor) | vision | LM | ~TFLOP / question | s / question |
|---|---|---|---|---|---|
| SmolVLM2-256M | ≤17 tiles of 512², 64 tok each (~1.1k tokens) | 93M | 135M | 7.0 | 0.38 |
| SmolVLM2-500M | same as 256M | 93M | 360M | 8.1 | 0.45 |
| LFM2.5-VL-450M | ≤10 tiles + thumbnail, ≤256 tok each (~2.9k tokens) | 86M | 350M | 7.9 | 0.45 |
| SmolVLM2-2.2B | ≤17 tiles of 384², 81 tok each (~1.4k tokens) | 400M | 1.7B | 29 | 1.55 |

Overhead covers downloading and loading weights plus the one-time CUDA warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from lumen.budget import Estimate, prior_estimate


@dataclass(frozen=True)
class ModelCost:
    sec_per_question: float
    overhead_s: float
    batch_size: int  # default batch that fits a 15 GB T4 in fp16 with room to spare


# Measured on the Kaggle T4 (SmolVLM2-500M, fp16, batch 8, fresh dev questions, 2026-09-30):
# 0.666 s/question with prefetching on (2 workers, the default) vs 0.798 with it off, on different
# 64-96-question samples. The derived table above said 0.45. SmolVLM2-500M uses the measured
# prefetch-on rate; the other three are their derived values scaled by the same factor
# (0.666 / 0.45 = 1.48) until their own calibration batches measure them.
# Superseded: derived (256m 0.38, 500m 0.45, lfm 0.45, 2.2b 1.55); prefetch-off-based x1.78.
_MEASURED_SCALE = 0.666 / 0.45

MODEL_COSTS: dict[str, ModelCost] = {
    "smolvlm2-256m": ModelCost(round(0.38 * _MEASURED_SCALE, 2), 240, 8),  # 0.56
    "smolvlm2-500m": ModelCost(0.67, 300, 8),  # measured, prefetch on
    "lfm2.5-vl-450m": ModelCost(round(0.45 * _MEASURED_SCALE, 2), 300, 8),  # 0.67
    "smolvlm2-2.2b": ModelCost(round(1.55 * _MEASURED_SCALE, 2), 600, 4),  # 2.29
}


def estimate_run(model_key: str, split: str, prompt_key: str, n_questions: int) -> Estimate:
    cost = MODEL_COSTS[model_key]
    return prior_estimate(
        f"baseline:{model_key}:{split}:{prompt_key}",
        n_questions,
        cost.sec_per_question,
        overhead_s=cost.overhead_s,
    )
