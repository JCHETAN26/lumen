"""Prior GPU-time estimates for Stage 3 training on a Kaggle T4 (replaced by calibration).

Prior: a training example costs about what a zero-shot question did (SmolVLM2-500M, measured
0.67 s/question at 2,048 px). A training step skips generation (16 decode steps) but adds a
backward pass and gradient-checkpointing recompute, roughly 3x a forward; the two are taken to
cancel. That gives ~5.4 s per 8-example step at full resolution, in line with the earlier rough
2.9-3.7 h per 18,471-example epoch. Image resolution scales it like inference
(lumen.zeroshot.estimates.resolution_scale). The speed test's first steps replace this.
"""

from __future__ import annotations

from lumen.budget import Estimate, prior_estimate
from lumen.zeroshot.estimates import MODEL_COSTS, resolution_scale

TRAIN_OVERHEAD_S = 600.0  # weights download/load, LoRA setup, first-step CUDA warm-up


def sec_per_step_prior(batch_size: int, long_edge: int | None) -> float:
    return batch_size * MODEL_COSTS["smolvlm2-500m"].sec_per_question * resolution_scale(long_edge)


def estimate_train(stage: str, steps: int, batch_size: int, long_edge: int | None) -> Estimate:
    return prior_estimate(
        stage, steps, sec_per_step_prior(batch_size, long_edge), overhead_s=TRAIN_OVERHEAD_S
    )
