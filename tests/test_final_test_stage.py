"""The sealed final_test stage on a fake VizWiz with a test split: everything checked before the
seal is claimed, exactly one completed run, the claim on the Hub surviving a wiped disk, and a
crashed run staying claimed. Fake backends; no torch."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pytest
from conftest import fake_records, jpeg, run_stages

from lumen import hub
from lumen.eval.test_seal import TestSealed
from lumen.pins import FilePin, Pins, load_pins
from lumen.train.selection import spec_sha256
from lumen.zeroshot.backends import FakeBackend

PIN = load_pins().models["smolvlm2-500m"]
CLEAN = {"commit": "a" * 40, "dirty": False}


def fargs(**kw) -> argparse.Namespace:
    base = dict(
        final_test_evaluation=True,
        gpu_hours_left=30.0,
        batch_size=4,
        dry_run=False,
        stand_in=False,
        device=None,
    )
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def sealed(env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake VizWiz with a test split, a committed dev choice, and fake backends."""
    data = env / "data"
    test = fake_records("test", 16)
    with zipfile.ZipFile(data / "Annotations.zip", "a") as z:
        z.writestr("Annotations/test.json", json.dumps([{**r, "answers": None} for r in test]))
    with zipfile.ZipFile(data / "test.zip", "w") as z:
        for r in test:
            z.writestr(f"test/{r['image']}", jpeg())
    (data / "VQA_test.json").write_text(json.dumps(test))
    pins = run_stages.load_pins()

    def pin(name: str, filename: str) -> FilePin:
        url = f"https://example.invalid/{filename}"
        return FilePin(name, url, (data / filename).stat().st_size, None)

    files = {
        **pins.files,
        "annotations": pin("annotations", "Annotations.zip"),
        "test_answers": pin("test_answers", "VQA_test.json"),
        "test_images": pin("test_images", "test.zip"),
    }
    monkeypatch.setattr(run_stages, "load_pins", lambda: Pins(**{**pins.__dict__, "files": files}))
    monkeypatch.setattr(run_stages, "final_test_git", lambda: CLEAN)

    spec = {
        "model": {"model_key": PIN.key, "repo": PIN.repo, "revision": PIN.revision},
        "config": {"prompt_key": "abstain", "image_longest_edge": 512, "seed": 1},
    }
    epoch_dir = env / "checkpoints" / "train" / "lr2e-4" / "epoch-1"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "adapter_model.safetensors").write_bytes(b"w")
    run_dir = env / "results" / "finetune" / "lr2e-4"
    run_dir.mkdir(parents=True)
    pointer = {
        "repo": "cjitendr/lumen-adapter",
        "repo_type": "model",
        "path": "runs/lr2e-4/epoch-1",
        "revision": "c" * 40,
        "files": hub.folder_digest(epoch_dir),
    }
    (run_dir / "train.json").write_text(
        json.dumps(
            {
                "spec": spec,
                "spec_sha256": spec_sha256(spec),
                "epochs_published": [{"epoch": 1, "step": 10, "adapter": pointer}],
            }
        )
    )
    (run_dir / "dev-e1.json").write_text(
        json.dumps({"status": "completed", "threshold_choice": {"threshold": 0.5}})
    )
    (env / "results" / "finetune" / "choice.json").write_text(
        json.dumps({"rule": "test rule", "chosen": {"run": "lr2e-4", "epoch": 1, "spec": spec}})
    )
    monkeypatch.setattr(
        run_stages,
        "make_finetuned_backend",
        lambda args, spec, d, adapter: FakeBackend(answer="dog", image_longest_edge=512),
    )
    monkeypatch.setattr(
        run_stages, "make_backend", lambda key, a: FakeBackend(answer="cat", image_longest_edge=512)
    )
    return env


def record(env: Path) -> dict:
    return json.loads((env / "results" / "test" / "FINAL_TEST_RUN.json").read_text())


def test_dry_run_and_refusals_claim_nothing(sealed: Path, monkeypatch) -> None:
    assert run_stages.cmd_final_test(fargs(dry_run=True)) == 0
    with pytest.raises(TestSealed, match="sealed"):
        run_stages.cmd_final_test(fargs(final_test_evaluation=False))
    monkeypatch.setattr(run_stages, "final_test_git", lambda: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(TestSealed, match="clean"):
        run_stages.cmd_final_test(fargs())
    monkeypatch.setattr(run_stages, "final_test_git", lambda: CLEAN)
    assert run_stages.cmd_final_test(fargs(gpu_hours_left=0.5)) == run_stages.EXIT_BUDGET
    assert not (sealed / "results" / "test").exists()
    assert "test/FINAL_TEST_RUN.json" not in run_stages.make_hub_api().files


def test_one_run_then_sealed_even_on_a_new_disk(sealed: Path) -> None:
    assert run_stages.cmd_final_test(fargs()) == 0
    r = record(sealed)
    assert r["status"] == "completed" and r["commit"] == "a" * 40
    res = r["results"]
    assert res["n"] == 16 and res["choice"]["run"] == "lr2e-4"
    ft, zs = res["systems"]["fine_tuned"], res["systems"]["zero_shot"]
    assert ft["metrics"]["accuracy"]["overall"] > zs["metrics"]["accuracy"]["overall"]
    assert ft["f1_at_dev_threshold"]["threshold"] == 0.5
    assert res["fine_tuned_vs_zero_shot"]["accuracy_diff"] > 0
    assert ft["predictions"]["stored"] == "private Hugging Face dataset"
    api = run_stages.make_hub_api()
    on_hub = json.loads(api.files["test/FINAL_TEST_RUN.json"])
    assert on_hub["status"] == "completed"
    assert "test/fine_tuned.predictions.jsonl" in api.files
    downloaded = json.loads((sealed / "download" / "test" / "FINAL_TEST_RUN.json").read_text())
    assert downloaded["status"] == "completed"
    with pytest.raises(TestSealed, match="already ran"):
        run_stages.cmd_final_test(fargs())
    # A new Kaggle session: the local record is gone, the Hub claim still seals test.
    (sealed / "results" / "test" / "FINAL_TEST_RUN.json").unlink()
    with pytest.raises(TestSealed, match="earlier session"):
        run_stages.cmd_final_test(fargs())


def test_a_crashed_run_stays_claimed(sealed: Path, monkeypatch) -> None:
    def broken(*a, **kw):
        raise RuntimeError("session died")

    monkeypatch.setattr(run_stages, "make_finetuned_backend", broken)
    with pytest.raises(RuntimeError, match="session died"):
        run_stages.cmd_final_test(fargs())
    assert record(sealed)["status"] == "started"
    claim = json.loads(run_stages.make_hub_api().files["test/FINAL_TEST_RUN.json"])
    assert claim["status"] == "started"
    with pytest.raises(TestSealed):
        run_stages.cmd_final_test(fargs())
