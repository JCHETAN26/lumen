"""The train_speed stage (Stage 3 prep) on the fake VizWiz from conftest.

Budget refusal, dry run, and split checks run in CI (no torch). The last test runs the whole stage
with the tiny SmolVLM2 stand-in on CPU (needs torch, transformers, peft).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from conftest import run_stages

from lumen.zeroshot.backends import FakeBackend


def targs(**kw) -> argparse.Namespace:
    base = dict(
        steps=4,
        micro_batch=2,
        grad_accum=1,
        calibration_steps=2,
        long_edge=512,
        gpu_hours_left=30.0,
        limit=None,
        stand_in=False,
        device=None,
        dry_run=False,
    )
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def chosen_prompt(env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Data stage done, splits committed, and SmolVLM2-500M's prompt chosen on dev_select."""
    assert run_stages.cmd_data(argparse.Namespace()) == 0
    committed = env / "committed_splits"
    committed.mkdir(exist_ok=True)
    for name in ("dev.txt", "train_fit.txt"):
        shutil.copy(env / "results" / "splits" / name, committed / name)
    monkeypatch.setattr(run_stages, "make_backend", lambda key, a: FakeBackend())
    base = dict(
        model=["smolvlm2-500m"],
        split="dev_select",
        prompt="both",
        gpu_hours_left=30.0,
        batch_size=4,
        calibration_items=4,
        limit=None,
        stand_in=False,
        device=None,
        dry_run=False,
        prefetch_workers=0,
        long_edge=None,
    )
    assert run_stages.cmd_baseline(argparse.Namespace(**base)) == 0
    return env


def no_model(*a, **kw):
    pytest.fail("loaded a model")


def test_dry_run_prints_the_prior_without_loading(chosen_prompt, monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_stages, "make_train_model", no_model)
    assert (
        run_stages.cmd_train_speed(targs(dry_run=True, steps=200, micro_batch=4, grad_accum=2)) == 0
    )
    out = capsys.readouterr().out
    assert "train_speed" in out and "512 px" in out and "200 steps x 8" in out and "GPU h" in out


def test_refuses_to_start_over_budget(chosen_prompt, monkeypatch) -> None:
    monkeypatch.setattr(run_stages, "make_train_model", no_model)
    assert run_stages.cmd_train_speed(targs(gpu_hours_left=0.5)) == run_stages.EXIT_BUDGET
    assert run_stages.cmd_train_speed(targs(gpu_hours_left=None)) == run_stages.EXIT_BUDGET
    assert not (chosen_prompt / "results" / "stage3_prep").exists()


def test_needs_a_resolution_choice_for_chosen(chosen_prompt) -> None:
    with pytest.raises(SystemExit, match="no resolution choice"):
        run_stages.cmd_train_speed(targs(long_edge="chosen"))


def test_overlapping_committed_splits_are_refused(chosen_prompt, monkeypatch) -> None:
    committed = chosen_prompt / "committed_splits"
    dev_ids = (committed / "dev.txt").read_text().split()
    with open(committed / "train_fit.txt", "a") as f:
        f.write(dev_ids[0] + "\n")
    monkeypatch.setattr(run_stages, "make_train_model", no_model)
    with pytest.raises(SystemExit, match="overlap"):
        run_stages.cmd_train_speed(targs())


def test_stand_in_end_to_end(chosen_prompt, monkeypatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from conftest import skip_on_usage_cap

    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_train_speed(
            targs(stand_in=True, device="cpu", limit=8, gpu_hours_left=None)
        )
    assert code == 0
    path = chosen_prompt / "results" / "stage3_prep" / "smoke" / "train_speed-le512.json"
    r = json.loads(path.read_text())
    assert r["status"] == "completed" and r["smoke"] and r["model"]["stand_in"]
    assert r["train"]["final_step"] == 4 and r["train"]["nonfinite_losses"] == []
    assert r["train"]["sec_per_step_median"] > 0 and r["projection"]["steps_per_epoch"] == 4
    assert r["targets"]["examples"] == 8 and r["config"]["image_longest_edge"] == 512
    assert r["config"]["prompt_key"] == r["prompt_selection"]["chosen"]
    assert r["gpu_hours"]["calibration"]["measured_steps"] == 2
    assert 0 < r["model"]["parameters"]["trainable"] < r["model"]["parameters"]["total"]
    assert (chosen_prompt / "download" / "stage3_prep" / "smoke" / path.name).exists()
    ckpts = list((chosen_prompt / "checkpoints" / "train_speed").rglob("trainer_state.pt"))
    assert len(ckpts) == 1  # the final state only; never in results/ or the download folder
    for root in (chosen_prompt / "results", chosen_prompt / "download"):
        assert not list(root.rglob("*.safetensors")) and not list(root.rglob("*.pt"))
