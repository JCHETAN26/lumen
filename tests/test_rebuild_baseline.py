"""rebuild_baseline: re-create result files lost with a Kaggle session from the Hub predictions.

Runs the fake pipeline (prompt choice on dev_select, then val), which uploads predictions to the
fake Hub; wipes results, local predictions, and the download folder like an ended session; then
rebuilds and checks the metrics match the originals exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from conftest import FakeHubApi, run_stages

from lumen.pins import load_pins
from lumen.zeroshot.backends import FakeBackend

PIN = load_pins().models["smolvlm2-500m"]


class PinnedFake(FakeBackend):
    """The fake backend, describing itself as the pinned SmolVLM2-500M."""

    def describe(self):
        return {
            "backend": "fake",
            "model_key": PIN.key,
            "repo": PIN.repo,
            "revision": PIN.revision,
            "stand_in": False,
        }


def bargs(**kw) -> argparse.Namespace:
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
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def lost(env: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, FakeHubApi, dict]:
    """The 500M runs done and uploaded, then the session's disk wiped. Returns the originals."""
    assert run_stages.cmd_data(argparse.Namespace()) == 0
    committed = env / "committed_splits"
    committed.mkdir(exist_ok=True)
    shutil.copy(env / "results" / "splits" / "dev.txt", committed / "dev.txt")
    monkeypatch.setattr(run_stages, "make_backend", lambda key, a: PinnedFake())
    assert run_stages.cmd_baseline(bargs()) == 0
    assert run_stages.cmd_baseline(bargs(split="val", prompt="dev-selected")) == 0
    run_dir = env / "results" / "baseline" / "smolvlm2-500m"
    originals = {f.stem: json.loads(f.read_text()) for f in run_dir.glob("*.json")}
    assert set(originals) == {"dev_select-plain", "dev_select-abstain", "val-abstain"} or set(
        originals
    ) == {"dev_select-plain", "dev_select-abstain", "val-plain"}
    for d in ("results/baseline", "predictions", "download"):
        shutil.rmtree(env / d)
    api = run_stages.make_hub_api()

    def fetch(repo, path, revision):
        assert len(revision) == 40
        out = env / "hub_cache" / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(api.files[path])
        return out

    monkeypatch.setattr(run_stages, "fetch_from_hub", fetch)
    return env, api, originals


def rebuild(model: str = "smolvlm2-500m") -> int:
    return run_stages.cmd_rebuild_baseline(argparse.Namespace(model=model))


def test_rebuild_reproduces_the_lost_metrics(lost, capsys) -> None:
    env, _, originals = lost
    assert rebuild() == 0
    run_dir = env / "results" / "baseline" / "smolvlm2-500m"
    for name, orig in originals.items():
        new = json.loads((run_dir / f"{name}.json").read_text())
        assert new["rebuilt"]["from"].startswith("predictions in the private")
        assert new["metrics"] == orig["metrics"]  # same predictions, same official scoring
        assert new["predictions"]["sha256"] == orig["predictions"]["sha256"]
        assert new["predictions"]["path"] == orig["predictions"]["path"]
        assert new["prompt"] == orig["prompt"] and new["gpu_hours"]["actual_this_run"] is None
        assert (env / "download" / "baseline" / "smolvlm2-500m" / f"{name}.json").exists()
    [val] = [n for n in originals if n.startswith("val-")]
    sel = json.loads((run_dir / f"{val}.json").read_text())["prompt_selection"]
    assert sel["split"] == "dev_select" and val == f"val-{sel['chosen']}"
    assert (env / "download" / "splits" / "dev_select.txt").exists()  # regenerated for commit
    assert run_stages.cmd_baseline_summary(argparse.Namespace()) == 0
    assert "nan" in capsys.readouterr().out  # GPU hours and s/q: lost, shown as nan


def test_existing_results_are_left_alone(lost, capsys) -> None:
    assert rebuild() == 0
    assert rebuild() == 0
    assert "already exists; left as is" in capsys.readouterr().out
    env = lost[0]
    assert not list((env / "results" / "baseline").rglob("*.1.json"))


def test_incomplete_predictions_are_refused(lost) -> None:
    env, api, _ = lost
    path = "baseline/smolvlm2-500m/dev_select-plain.predictions.jsonl"
    lines = api.files[path].decode().splitlines()
    api.files[path] = ("\n".join(lines[:-1]) + "\n").encode()  # drop one answered question
    assert rebuild() == run_stages.EXIT_REBUILD
    run_dir = env / "results" / "baseline" / "smolvlm2-500m"
    assert not (run_dir / "dev_select-plain.json").exists()
    assert (run_dir / "dev_select-abstain.json").exists()  # the good file still rebuilt


def test_wrong_model_revision_is_refused(lost, capsys) -> None:
    _, api, _ = lost
    path = "baseline/smolvlm2-500m/dev_select-abstain.predictions.jsonl"
    text = api.files[path].decode().replace(PIN.revision, "f" * 40)
    api.files[path] = text.encode()
    assert rebuild() == run_stages.EXIT_REBUILD
    out = capsys.readouterr().out
    assert "is not the pin" in out
    assert "no dev prompt choice to check it against" in out  # val refused too, not a crash


def test_nothing_on_the_hub(env, monkeypatch, capsys) -> None:
    assert rebuild() == run_stages.EXIT_REBUILD
    assert "must be rerun" in capsys.readouterr().out
    monkeypatch.setattr(run_stages, "make_hub_api", lambda: FakeHubApi(private=False))
    assert rebuild() == run_stages.EXIT_STORAGE
