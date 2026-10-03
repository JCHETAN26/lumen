"""The pre-registered resolution rule (lumen.zeroshot.resolution) on synthetic scores."""

from __future__ import annotations

import numpy as np
import pytest

from lumen.zeroshot.resolution import MARGIN_POINTS, RULE, choose_long_edge

N = 1000


def arm(correct: set[int]) -> np.ndarray:
    return np.array([100.0 if i in correct else 0.0 for i in range(N)])


BASE = set(range(0, N, 2))  # 50% accuracy
# 40 right answers lost, 38 gained: 0.2 points worse, with many discordant questions.
NOISY = (BASE - set(range(0, 80, 2))) | set(range(1, 77, 2))


def test_near_identical_cheaper_arm_is_chosen_even_if_not_best() -> None:
    out = choose_long_edge({2048: arm(BASE), 1024: arm(BASE - {0, 2})})  # 2 of 1000 flip: -0.2
    assert out["best"] == 2048 and out["chosen"] == 1024
    v = out["vs_best"][1024]
    assert v["accuracy_diff"] == pytest.approx(-0.2) and v["within_margin"]
    assert v["accuracy_diff_ci95"]["low"] >= -MARGIN_POINTS


def test_clearly_worse_cheaper_arm_is_rejected() -> None:
    worse = BASE - set(range(0, 120, 2))  # 60 fewer right: -6 points
    out = choose_long_edge({2048: arm(BASE), 1024: arm(worse)})
    assert out["chosen"] == 2048 and not out["vs_best"][1024]["within_margin"]


def test_slightly_worse_but_noisy_arm_fails_the_margin() -> None:
    """Only 0.2 points worse, but 78 questions swap right/wrong: the paired CI is too wide to show
    it within 1 point, so the rule keeps the best arm (the cost of a strict margin at small n)."""
    out = choose_long_edge({2048: arm(BASE), 1024: arm(NOISY)})
    assert out["vs_best"][1024]["accuracy_diff"] == pytest.approx(-0.2)
    assert out["vs_best"][1024]["accuracy_diff_ci95"]["low"] < -MARGIN_POINTS
    assert out["chosen"] == 2048


def test_cheaper_better_arm_is_best() -> None:
    out = choose_long_edge({2048: arm(BASE), 1024: arm(BASE | {1, 3, 5})})
    assert out["best"] == out["chosen"] == 1024 and 1024 not in out["vs_best"]


def test_exact_tie_goes_to_fewer_tokens() -> None:
    out = choose_long_edge({2048: arm(BASE), 1536: arm(BASE), 512: arm(set())})
    assert out["best"] == 1536 and out["chosen"] == 1536


def test_smallest_qualifying_edge_wins_even_past_a_failing_one() -> None:
    out = choose_long_edge(
        {2048: arm(BASE), 1536: arm(NOISY), 1024: arm(BASE - {0}), 512: arm(set())}
    )
    assert not out["vs_best"][1536]["within_margin"] and out["vs_best"][1024]["within_margin"]
    assert out["chosen"] == 1024


def test_rule_text_records_the_margin() -> None:
    assert f"-{MARGIN_POINTS:g} point" in RULE and "not decisive" in RULE


@pytest.mark.parametrize("arms", [{}, {2048: arm(BASE), 1024: np.ones(5)}, {2048: np.array([])}])
def test_bad_input(arms) -> None:
    with pytest.raises(ValueError):
        choose_long_edge(arms)
