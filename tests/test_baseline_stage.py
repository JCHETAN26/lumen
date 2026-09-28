"""The Stage 2 baseline stage end to end on fake VizWiz zips, before any GPU time is spent.

A fake backend (no ML dependencies, runs in CI) drives the full flow: dev runs for both prompts,
dev-only prompt selection, the val run, budget refusals, calibration stop + resume, and the
NaN stop. The last test runs the same stage with the tiny HF stand-in (needs torch).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from conftest import run_stages

from lumen.budget import Estimate
from lumen.zeroshot.backends import FakeBackend


def args(**kw) -> argparse.Namespace:
    base = dict(
        model=["smolvlm2-500m"],
        split="dev",
        prompt="both",
        gpu_hours_left=30.0,
        batch_size=4,
        calibration_items=4,
        limit=None,
        stand_in=False,
        device=None,
        dry_run=False,
    )
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def ready(env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake VizWiz after the data stage, with the dev split committed and a fake backend."""
    assert run_stages.cmd_data(argparse.Namespace()) == 0
    committed = env / "committed_splits"
    committed.mkdir(exist_ok=True)
    shutil.copy(env / "results" / "splits" / "dev.txt", committed / "dev.txt")
    fakes: dict[str, FakeBackend] = {}
    monkeypatch.setattr(
        run_stages, "make_backend", lambda key, a: fakes.setdefault(key, FakeBackend())
    )
    return env


def result(env: Path, model: str, name: str) -> dict:
    return json.loads((env / "results" / "baseline" / model / name).read_text())


def test_dry_run_prints_plan_without_loading(ready: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(run_stages, "make_backend", lambda *a: pytest.fail("loaded a model"))
    assert run_stages.cmd_baseline(args(model=["all"], dry_run=True, gpu_hours_left=2.0)) == 0
    out = capsys.readouterr().out
    assert "smolvlm2-2.2b" in out and "total" in out and "DOES NOT FIT" in out


def test_dev_both_prompts_then_val_with_dev_selected_prompt(ready: Path) -> None:
    assert run_stages.cmd_baseline(args()) == 0
    for p in ("plain", "abstain"):
        r = result(ready, "smolvlm2-500m", f"dev-{p}.json")
        assert r["status"] == "completed" and r["split"] == "dev" and r["metrics"]["n"] == 10
        # The fake answers "dog" (the ground truth) and says unanswerable on the blur questions.
        assert r["metrics"]["accuracy"]["overall"] == pytest.approx(100.0)
        assert r["metrics"]["answerability"]["average_precision"] == pytest.approx(100.0)
        assert r["gpu_hours"]["prior_estimate"]["hours"] > 0
        assert r["meta"]["git"] is not None and r["prompt"]["prompt_version"] == "v1"
    assert run_stages.cmd_baseline(args(split="val", prompt="dev-selected")) == 0
    [val] = list((ready / "results" / "baseline" / "smolvlm2-500m").glob("val-*.json"))
    v = json.loads(val.read_text())
    assert v["prompt_selection"]["chosen"] == v["prompt"]["prompt_key"]
    assert set(v["prompt_selection"]["dev"]) == {"plain", "abstain"}
    assert v["metrics"]["n"] == 24
    assert (ready / "download" / "baseline" / "smolvlm2-500m" / val.name).exists()


def test_val_prompt_must_come_from_dev(ready: Path) -> None:
    with pytest.raises(SystemExit, match="dev-selected"):
        run_stages.cmd_baseline(args(split="val", prompt="plain"))
    with pytest.raises(SystemExit, match="no dev result"):
        run_stages.cmd_baseline(args(split="val", prompt="dev-selected"))


def test_refuses_to_start_over_budget_or_without_quota(ready: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_stages, "make_backend", lambda *a: pytest.fail("loaded a model"))
    assert run_stages.cmd_baseline(args(gpu_hours_left=0.5)) == run_stages.EXIT_BUDGET
    assert run_stages.cmd_baseline(args(gpu_hours_left=None)) == run_stages.EXIT_BUDGET
    assert not (ready / "results" / "baseline").exists()


def test_calibration_stop_then_resume(ready: Path, monkeypatch, capsys) -> None:
    import lumen.budget as budget

    real = budget.calibrated_estimate
    monkeypatch.setattr(
        budget,
        "calibrated_estimate",  # the GPU turns out far slower than the prior
        lambda prior, n, s, **kw: Estimate(
            prior.stage, prior.items, 3600.0, 0.0, 1.0, "calibrated"
        ),
    )
    assert (
        run_stages.cmd_baseline(args(prompt="plain", gpu_hours_left=2.0)) == run_stages.EXIT_BUDGET
    )
    assert "STOPPED (budget)" in capsys.readouterr().out
    pred = ready / "results" / "baseline" / "smolvlm2-500m" / "dev-plain.predictions.jsonl"
    answered = len(pred.read_text().splitlines()) - 1  # minus header
    assert 4 <= answered < 10  # calibration batch kept, rest not run
    monkeypatch.setattr(budget, "calibrated_estimate", real)
    assert run_stages.cmd_baseline(args(prompt="plain")) == 0
    r = result(ready, "smolvlm2-500m", "dev-plain.json")
    assert r["run"]["resumed"] == answered and r["metrics"]["n"] == 10


def test_non_finite_output_stops_and_is_recorded(ready: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_stages, "make_backend", lambda *a: FakeBackend(nan_on="blurry"))
    assert run_stages.cmd_baseline(args(prompt="abstain")) == run_stages.EXIT_NON_FINITE
    r = result(ready, "smolvlm2-500m", "dev-abstain.json")
    assert r["status"] == "failed_non_finite" and "metrics" not in r


def test_stand_in_hf_model_end_to_end(ready: Path, monkeypatch) -> None:
    """The real HF code path (processor, chat template, generate, scoring) with a tiny model."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from conftest import skip_on_usage_cap

    monkeypatch.setattr(run_stages, "make_backend", run_stages._cached_backend)
    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_baseline(
            args(
                model=["smolvlm2-500m", "lfm2.5-vl-450m"],
                prompt="abstain",
                stand_in=True,
                device="cpu",
                limit=6,
                gpu_hours_left=None,
            )
        )
    assert code == 0
    for model in ("smolvlm2-500m", "lfm2.5-vl-450m"):
        r = json.loads(
            (ready / "results" / "baseline" / "smoke" / model / "dev-abstain.json").read_text()
        )
        assert r["smoke"] and r["model"]["stand_in"] and r["metrics"]["n"] == 6
        assert 0 <= r["metrics"]["accuracy"]["overall"] <= 100
    assert not (ready / "results" / "baseline" / "smolvlm2-500m").exists()  # smoke kept apart
