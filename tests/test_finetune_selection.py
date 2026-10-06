"""Stage 3 choices on dev (lumen.train.selection) and paired val comparisons (lumen.eval.paired)."""

from __future__ import annotations

import numpy as np
import pytest

from lumen.eval.paired import paired_difference, seed_summary
from lumen.train.selection import choose_finetune, family, is_sweep_run, spec_sha256


def cand(run: str, epoch: int, acc: float, ap: float = 50.0, step: int | None = None) -> dict:
    return {
        "run": run,
        "epoch": epoch,
        "step": step if step is not None else 100 * epoch,
        "accuracy": acc,
        "average_precision": ap,
    }


def test_highest_dev_accuracy_wins() -> None:
    c = choose_finetune([cand("a", 1, 40.0), cand("b", 1, 42.0), cand("a", 2, 41.0)])
    assert (c["chosen"]["run"], c["chosen"]["epoch"]) == ("b", 1)
    assert [r["run"] for r in c["ranking"]] == ["b", "a", "a"]
    assert "dev" in c["rule"] and "2026-10-06" in c["rule"]


def test_ties_break_on_ap_then_fewer_steps_then_name() -> None:
    c = choose_finetune([cand("a", 2, 40.0, 50.0), cand("b", 1, 40.0, 51.0)])
    assert c["chosen"]["run"] == "b"
    c = choose_finetune([cand("a", 2, 40.0), cand("a", 1, 40.0)])
    assert c["chosen"]["epoch"] == 1
    c = choose_finetune([cand("z", 1, 40.0), cand("y", 1, 40.0)])
    assert c["chosen"]["run"] == "y"


def test_no_or_duplicate_candidates_are_errors() -> None:
    with pytest.raises(ValueError, match="no dev results"):
        choose_finetune([])
    with pytest.raises(ValueError, match="duplicate"):
        choose_finetune([cand("a", 1, 1.0), cand("a", 1, 2.0)])


def spec(seed: int = 1, unans: bool = True, lr: float = 2e-4, steps: int = 10) -> dict:
    return {
        "model": {"revision": "x"},
        "config": {
            "lr": lr,
            "seed": seed,
            "include_unanswerable": unans,
            "schedule_steps": steps * 2,
        },
        "epochs": 2,
        "steps_per_epoch": steps,
        "max_steps": steps * 2,
    }


def test_replicates_share_a_family_and_other_configs_do_not() -> None:
    main = spec()
    assert family(spec(seed=2)) == family(main)  # seed replicate
    assert family(spec(unans=False, steps=8)) == family(main)  # ablation: fewer examples
    assert family(spec(lr=1e-4)) != family(main)
    assert spec_sha256(main) != spec_sha256(spec(seed=2))
    assert is_sweep_run(main, 1) and not is_sweep_run(spec(seed=2), 1)
    assert not is_sweep_run(spec(unans=False), 1)


def test_paired_difference_is_on_the_same_questions() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    a = np.array([100, 100, 100, 100, 0, 100, 100, 100.0])
    b = np.array([0, 100, 0, 100, 0, 100, 100, 0.0])
    sa = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.9, 0.8])  # ranks unanswerable perfectly
    sb = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3, 0.2, 0.8])
    d = paired_difference(y, a, b, sa, sb, resamples=200)
    assert d["accuracy_diff"] == pytest.approx(a.mean() - b.mean())
    assert d["ap_a"] == pytest.approx(100.0) and d["ap_diff"] > 0
    assert d["accuracy_diff_ci95"]["low"] <= d["accuracy_diff"] <= d["accuracy_diff_ci95"]["high"]
    assert d["accuracy_diff_unanswerable"] == pytest.approx(100 / 3)
    same = paired_difference(y, a, a, sa, sa, resamples=50)
    assert same["accuracy_diff_ci95"]["low"] == same["accuracy_diff_ci95"]["high"] == 0.0
    with pytest.raises(ValueError, match="same"):
        paired_difference(y, a[:-1], b, sa, sb)


def test_seed_summary() -> None:
    s = seed_summary([40.0, 42.0, 44.0])
    assert s["mean"] == 42.0 and s["std"] == pytest.approx(2.0) and s["n_seeds"] == 3
    assert seed_summary([40.0])["std"] is None
    with pytest.raises(ValueError):
        seed_summary([])
