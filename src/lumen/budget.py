"""GPU-hour estimates and the Kaggle weekly-quota guard.

Kaggle does not expose the remaining weekly GPU quota through its API, so the runner requires it
as an input (`--gpu-hours-left`, read off kaggle.com -> Settings) for every GPU stage, estimates
the stage's GPU hours first, and refuses to start if the estimate plus a reserve exceeds it.

Estimates are two-phase:
1. Before loading anything: a prior from `sec_per_item` (conservative, per stage and model).
2. After a short timed calibration batch on the real GPU: a measured rate, checked again before
   the full run. Both estimates go into the stage's result file, next to the actual hours used,
   so later priors come from measurements rather than guesses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Held back from every run so a session never ends exactly at zero quota (debugging, the
# download cell, and the next session's setup all need a little GPU time).
DEFAULT_RESERVE_HOURS = 1.0
DEFAULT_SAFETY_FACTOR = 1.3


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Estimate:
    stage: str
    items: int
    sec_per_item: float
    overhead_s: float  # model download/load, warm-up, evaluation
    safety_factor: float
    source: str  # "prior" or "calibrated"

    @property
    def hours(self) -> float:
        return (self.items * self.sec_per_item + self.overhead_s) * self.safety_factor / 3600

    def as_dict(self) -> dict:
        return {**asdict(self), "hours": round(self.hours, 3)}


def prior_estimate(
    stage: str,
    items: int,
    sec_per_item: float,
    overhead_s: float = 600.0,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> Estimate:
    if items < 0 or sec_per_item <= 0 or overhead_s < 0 or safety_factor < 1:
        raise ValueError("invalid estimate inputs")
    return Estimate(stage, items, sec_per_item, overhead_s, safety_factor, "prior")


def calibrated_estimate(
    prior: Estimate,
    measured_items: int,
    measured_seconds: float,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> Estimate:
    """Replace the prior rate with one measured on a calibration batch."""
    if measured_items <= 0 or measured_seconds <= 0:
        raise ValueError("calibration needs a positive item count and duration")
    return Estimate(
        prior.stage,
        prior.items,
        measured_seconds / measured_items,
        prior.overhead_s,
        safety_factor,
        "calibrated",
    )


def check_budget(
    estimate: Estimate, gpu_hours_left: float | None, reserve_hours: float = DEFAULT_RESERVE_HOURS
) -> None:
    """Raise BudgetExceeded unless estimate + reserve fits in the remaining weekly quota."""
    if gpu_hours_left is None:
        raise BudgetExceeded(
            f"{estimate.stage}: pass --gpu-hours-left (Kaggle -> Settings -> GPU quota). "
            f"Estimated need: {estimate.hours:.2f} h + {reserve_hours:.1f} h reserve."
        )
    if gpu_hours_left < 0:
        raise ValueError("gpu_hours_left must be >= 0")
    need = estimate.hours + reserve_hours
    if need > gpu_hours_left:
        raise BudgetExceeded(
            f"{estimate.stage}: estimated {estimate.hours:.2f} h ({estimate.source}, "
            f"{estimate.items} items x {estimate.sec_per_item:.2f} s x {estimate.safety_factor} "
            f"+ overhead) + {reserve_hours:.1f} h reserve = {need:.2f} h > {gpu_hours_left:.2f} h "
            "left this week. Not starting."
        )
