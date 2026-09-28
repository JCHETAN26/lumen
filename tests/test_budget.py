from __future__ import annotations

import pytest

from lumen.budget import BudgetExceeded, calibrated_estimate, check_budget, prior_estimate


def test_hours_arithmetic() -> None:
    e = prior_estimate("baseline", items=3600, sec_per_item=1.0, overhead_s=0, safety_factor=1.0)
    assert e.hours == pytest.approx(1.0)
    e = prior_estimate("baseline", items=3600, sec_per_item=1.0, overhead_s=1800, safety_factor=1.5)
    assert e.hours == pytest.approx(2.25)


def test_refuses_when_over_quota() -> None:
    e = prior_estimate("train", items=36000, sec_per_item=1.0, overhead_s=0, safety_factor=1.0)
    check_budget(e, gpu_hours_left=11.0, reserve_hours=1.0)  # 10 + 1 fits exactly
    with pytest.raises(BudgetExceeded, match="Not starting"):
        check_budget(e, gpu_hours_left=10.5, reserve_hours=1.0)


def test_requires_quota_input() -> None:
    e = prior_estimate("train", items=10, sec_per_item=1.0)
    with pytest.raises(BudgetExceeded, match="--gpu-hours-left"):
        check_budget(e, gpu_hours_left=None)


def test_calibration_replaces_rate() -> None:
    prior = prior_estimate("eval", items=1000, sec_per_item=5.0, overhead_s=0, safety_factor=1.0)
    cal = calibrated_estimate(prior, measured_items=50, measured_seconds=25.0, safety_factor=1.0)
    assert cal.source == "calibrated" and cal.sec_per_item == 0.5
    assert cal.hours == pytest.approx(500 / 3600)
    assert cal.as_dict()["hours"] == round(cal.hours, 3)


@pytest.mark.parametrize("kw", [{"items": -1}, {"sec_per_item": 0}, {"safety_factor": 0.9}])
def test_invalid_inputs(kw: dict) -> None:
    args = {"stage": "x", "items": 1, "sec_per_item": 1.0} | kw
    with pytest.raises(ValueError):
        prior_estimate(**args)
