"""Stage 3 stages on the fake VizWiz from conftest: train, finetune_eval, finetune_choice,
finetune_compare.

The CI tests (no torch) drive eval -> choice -> val -> compare with fake backends and synthetic
smoke train results, and check every refusal: val before a dev choice, val for a config dev didn't
choose, a run name reused with another config, the quota guard. The last tests train the tiny
SmolVLM2 stand-in on CPU (needs torch, transformers, peft): two epochs with adapters published,
the session's disk wiped, and the run resumed from the (fake) Hub.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from conftest import run_stages

from lumen import hub
from lumen.pins import load_pins
from lumen.train.loop import TrainConfig
from lumen.train.selection import spec_sha256
from lumen.zeroshot.backends import FakeBackend

PIN = load_pins().models["smolvlm2-500m"]
SWEEP_SEED = TrainConfig().seed


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


def eargs(**kw) -> argparse.Namespace:
    base = dict(
        run="lr2e-4",
        epoch=1,
        split="dev",
        gpu_hours_left=30.0,
        batch_size=4,
        calibration_items=4,
        prefetch_workers=0,
        limit=6,
        stand_in=False,
        device=None,
        dry_run=False,
    )
    return argparse.Namespace(**(base | kw))


def targs(**kw) -> argparse.Namespace:
    base = dict(
        run="lr2e-4",
        epochs=2,
        stop_after_epoch=None,
        lr=2e-4,
        lora_r=4,
        lora_alpha=None,
        warmup_steps=1,
        seed=None,
        no_unanswerable=False,
        micro_batch=2,
        grad_accum=1,
        checkpoint_every=2,
        calibration_steps=2,
        long_edge=512,
        gpu_hours_left=30.0,
        session_hours=11.0,
        limit=8,
        stand_in=True,
        device="cpu",
        dry_run=False,
    )
    return argparse.Namespace(**(base | kw))


@pytest.fixture
def ready(env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Data stage done, dev + train_fit committed, SmolVLM2-500M's prompt chosen on dev_select."""
    assert run_stages.cmd_data(argparse.Namespace()) == 0
    committed = env / "committed_splits"
    committed.mkdir(exist_ok=True)
    for name in ("dev.txt", "train_fit.txt"):
        shutil.copy(env / "results" / "splits" / name, committed / name)
    monkeypatch.setattr(run_stages, "make_backend", lambda key, a: FakeBackend())
    assert run_stages.cmd_baseline(bargs()) == 0
    return env


def fake_train(
    env: Path, run: str, *, seed: int = SWEEP_SEED, unans: bool = True, lr: float = 2e-4
) -> dict:
    """A smoke train result with one locally published epoch adapter (what `train` leaves)."""
    spec = {
        "model": {"model_key": PIN.key, "repo": PIN.repo, "revision": PIN.revision},
        "config": {
            "prompt_key": "abstain",
            "image_longest_edge": 512,
            "include_unanswerable": unans,
            "lr": lr,
            "lora_r": 16,
            "seed": seed,
            "schedule_steps": 10 if unans else 8,
        },
        "epochs": 1,
        "steps_per_epoch": 10 if unans else 8,
        "max_steps": 10 if unans else 8,
    }
    epoch_dir = env / "checkpoints" / "train" / "smoke" / run / "epoch-1"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "adapter_model.safetensors").write_bytes(f"{run}".encode())
    doc = {
        "status": "completed",
        "run": run,
        "spec": spec,
        "spec_sha256": spec_sha256(spec),
        "epochs_published": [
            {
                "epoch": 1,
                "step": spec["max_steps"],
                "adapter": {
                    "stored": "local only (smoke run; not uploaded)",
                    "local": str(epoch_dir),
                    "files": hub.folder_digest(epoch_dir),
                },
            }
        ],
    }
    out = env / "results" / "finetune" / "smoke" / run / "train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc))
    return doc


class AdapterFake(FakeBackend):
    """A fake fine-tuned model: records the adapter it was built with, like HFBackend does."""

    def __init__(self, adapter: dict, answer: str) -> None:
        super().__init__(answer=answer, image_longest_edge=512)
        self.adapter = adapter

    def describe(self):
        return {**super().describe(), "model_key": PIN.key, "adapter": self.adapter}


@pytest.fixture
def fakes(ready: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    built = []

    def make(args, spec, adapter_dir, adapter):
        assert (adapter_dir / "adapter_model.safetensors").exists()
        # The "cat" run is worse than the zero-shot fake (which answers the ground truth "dog").
        built.append(adapter)
        return AdapterFake(adapter, "cat" if adapter["run"] == "lr1e-4" else "dog")

    monkeypatch.setattr(run_stages, "make_finetuned_backend", make)
    return ready


def result(env: Path, run: str, name: str) -> dict:
    return json.loads((env / "results" / "finetune" / "smoke" / run / name).read_text())


def test_dev_eval_records_the_adapter_and_picks_a_threshold(fakes: Path) -> None:
    fake_train(fakes, "lr2e-4")
    assert run_stages.cmd_finetune_eval(eargs()) == 0
    r = result(fakes, "lr2e-4", "dev-e1.json")
    assert r["status"] == "completed" and r["smoke"] and r["metrics"]["n"] == 6
    assert r["adapter"]["run"] == "lr2e-4" and r["adapter"]["epoch"] == 1
    assert r["adapter"]["spec_sha256"] == spec_sha256(r["spec"])
    t = r["threshold_choice"]
    assert 0.0 <= t["threshold"] <= 1.0 and "dev" in t["rule"]
    assert r["answerable_breakdown"]["n"] > 0
    assert r["predictions"]["stored"].startswith("local only")
    preds = fakes / "predictions" / "finetune" / "smoke" / "lr2e-4" / "dev-e1.predictions.jsonl"
    header = json.loads(preds.read_text().splitlines()[0])["_header"]
    assert header["model"]["adapter"]["run"] == "lr2e-4"  # never mixed with another adapter
    assert (fakes / "download" / "finetune" / "smoke" / "lr2e-4" / "dev-e1.json").exists()


def test_val_needs_a_dev_choice_and_only_for_the_chosen_config(fakes: Path) -> None:
    fake_train(fakes, "lr2e-4")
    fake_train(fakes, "lr1e-4", lr=1e-4)
    with pytest.raises(SystemExit, match="no fine-tuning choice"):
        run_stages.cmd_finetune_eval(eargs(split="val"))
    for run in ("lr2e-4", "lr1e-4"):
        assert run_stages.cmd_finetune_eval(eargs(run=run)) == 0
    assert run_stages.cmd_finetune_choice(argparse.Namespace(smoke=True)) == 0
    choice = json.loads((fakes / "results" / "finetune" / "smoke" / "choice.json").read_text())
    assert choice["chosen"]["run"] == "lr2e-4"  # "dog" beats "cat" on dev
    assert [r["run"] for r in choice["ranking"]] == ["lr2e-4", "lr1e-4"]
    assert choice["data_used"].startswith("dev only")
    with pytest.raises(SystemExit, match="val is only for the config and epoch chosen"):
        run_stages.cmd_finetune_eval(eargs(run="lr1e-4", split="val"))
    assert run_stages.cmd_finetune_eval(eargs(split="val")) == 0
    v = result(fakes, "lr2e-4", "val-e1.json")
    assert v["choice"]["run"] == "lr2e-4"
    dev = result(fakes, "lr2e-4", "dev-e1.json")
    assert v["f1_at_dev_threshold"]["threshold"] == dev["threshold_choice"]["threshold"]


def test_replicates_are_not_candidates_but_are_allowed_on_val(fakes: Path) -> None:
    fake_train(fakes, "lr2e-4")
    fake_train(fakes, "lr2e-4-s2", seed=2)
    fake_train(fakes, "lr2e-4-nounans", unans=False)
    for run in ("lr2e-4", "lr2e-4-s2", "lr2e-4-nounans"):
        assert run_stages.cmd_finetune_eval(eargs(run=run)) == 0
    assert run_stages.cmd_finetune_choice(argparse.Namespace(smoke=True)) == 0
    choice = json.loads((fakes / "results" / "finetune" / "smoke" / "choice.json").read_text())
    assert [r["run"] for r in choice["ranking"]] == ["lr2e-4"]
    for run in ("lr2e-4", "lr2e-4-s2", "lr2e-4-nounans"):
        assert run_stages.cmd_finetune_eval(eargs(run=run, split="val")) == 0


def test_compare_against_the_matched_zero_shot_baseline(fakes: Path, capsys) -> None:
    fake_train(fakes, "lr2e-4")
    fake_train(fakes, "lr2e-4-s2", seed=2)
    fake_train(fakes, "lr2e-4-nounans", unans=False)
    for run in ("lr2e-4", "lr2e-4-s2", "lr2e-4-nounans"):
        assert run_stages.cmd_finetune_eval(eargs(run=run)) == 0
    assert run_stages.cmd_finetune_choice(argparse.Namespace(smoke=True)) == 0
    for run in ("lr2e-4", "lr2e-4-s2", "lr2e-4-nounans"):
        assert run_stages.cmd_finetune_eval(eargs(run=run, split="val")) == 0
    with pytest.raises(SystemExit, match="no matched zero-shot baseline"):
        run_stages.cmd_finetune_compare(argparse.Namespace(smoke=True))
    base = bargs(split="val", prompt="dev-selected", long_edge=512, limit=6)
    assert run_stages.cmd_baseline(base) == 0
    assert run_stages.cmd_finetune_compare(argparse.Namespace(smoke=True)) == 0
    c = json.loads((fakes / "results" / "finetune" / "smoke" / "comparison.json").read_text())
    assert c["n"] == 6 and c["zero_shot"]["result"] == "val-abstain-le512"
    assert [m["run"] for m in c["fine_tuned"]] == ["lr2e-4", "lr2e-4-s2"]
    assert c["across_seeds"]["accuracy"]["n_seeds"] == 2
    v = c["fine_tuned"][0]["vs_zero_shot"]
    assert v["n"] == 6 and v["accuracy_diff"] == pytest.approx(0.0)  # same fake answers
    [ab] = c["ablation_no_unanswerable"]
    assert ab["run"] == "lr2e-4-nounans" and ab["vs_full_training"]["partner"] == "lr2e-4"
    assert "across 2 seed(s)" in capsys.readouterr().out


def test_compare_refuses_different_question_sets(fakes: Path) -> None:
    fake_train(fakes, "lr2e-4")
    assert run_stages.cmd_finetune_eval(eargs()) == 0
    assert run_stages.cmd_finetune_choice(argparse.Namespace(smoke=True)) == 0
    assert run_stages.cmd_finetune_eval(eargs(split="val", limit=5)) == 0
    base = bargs(split="val", prompt="dev-selected", long_edge=512, limit=6)
    assert run_stages.cmd_baseline(base) == 0
    with pytest.raises(SystemExit, match="different val questions"):
        run_stages.cmd_finetune_compare(argparse.Namespace(smoke=True))


def test_changed_adapter_files_are_refused(fakes: Path) -> None:
    fake_train(fakes, "lr2e-4")
    adapter = fakes / "checkpoints/train/smoke/lr2e-4/epoch-1/adapter_model.safetensors"
    adapter.write_bytes(b"something else")
    with pytest.raises(SystemExit, match="missing or changed"):
        run_stages.cmd_finetune_eval(eargs())


def test_train_dry_run_and_budget_guard(ready: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_stages, "make_train_model", lambda *a: pytest.fail("loaded a model"))
    a = targs(stand_in=False, limit=None, device=None, dry_run=True, epochs=3)
    assert run_stages.cmd_train(a) == 0
    out = capsys.readouterr().out
    assert "train | lr2e-4 | 512 px" in out and "x 3 epochs" in out and "GPU h (prior" in out
    real = targs(stand_in=False, limit=None, device=None, gpu_hours_left=0.5)
    assert run_stages.cmd_train(real) == run_stages.EXIT_BUDGET
    assert not (ready / "results" / "finetune").exists()


def test_train_uses_the_speed_test_when_there_is_one(ready: Path, capsys) -> None:
    out = ready / "results" / "stage3_prep" / "train_speed-le512.json"
    out.parent.mkdir(parents=True)
    out.write_text(
        json.dumps(
            {
                "status": "completed",
                "smoke": False,
                "config": {"effective_batch_size": 2},
                "train": {"sec_per_step_median": 1.5},
            }
        )
    )
    assert run_stages.cmd_train(targs(stand_in=False, limit=None, dry_run=True)) == 0
    assert "speed test results/stage3_prep/train_speed-le512.json" in capsys.readouterr().out


def test_bad_run_names_and_stop_epoch(ready: Path) -> None:
    for bad in ("Has-Caps", "smoke", "a/b", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            run_stages._run_name_arg(bad)
    with pytest.raises(SystemExit, match="stop-after-epoch"):
        run_stages.cmd_train(targs(stop_after_epoch=3, dry_run=True))


# ------------------------------------------------------------------ tiny stand-in, real code


def needs_torch() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")


def test_stand_in_train_then_eval(ready: Path, monkeypatch) -> None:
    needs_torch()
    from conftest import skip_on_usage_cap

    with skip_on_usage_cap("Hugging Face"):
        assert run_stages.cmd_train(targs()) == 0
    r = result(ready, "lr2e-4", "train.json")
    assert r["status"] == "completed" and r["smoke"] and r["spec"]["steps_per_epoch"] == 4
    assert [e["epoch"] for e in r["epochs_published"]] == [1, 2]
    assert [e["step"] for e in r["epochs_published"]] == [4, 8]
    assert r["spec"]["config"]["schedule_steps"] == 8  # one schedule over both epochs
    assert r["session"] == {"start_step": 0, "final_step": 8, "target_step": 8}
    assert r["config"]["lora_alpha"] == 8 and r["checkpoint"] is None  # smoke: never uploaded
    # A rerun has nothing left to do: it never loads the model or writes another result.
    with monkeypatch.context() as m:
        m.setattr(run_stages, "make_train_model", lambda *a: pytest.fail("loaded a model"))
        assert run_stages.cmd_train(targs()) == 0
    assert not (ready / "results/finetune/smoke/lr2e-4/train.1.json").exists()
    # The same name with another config is refused.
    with pytest.raises(SystemExit, match="different spec"):
        run_stages.cmd_train(targs(lr=1e-3))
    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_finetune_eval(
            eargs(epoch=2, stand_in=True, device="cpu", limit=4, gpu_hours_left=None)
        )
    assert code == 0
    ev = result(ready, "lr2e-4", "dev-e2.json")
    assert ev["status"] == "completed" and ev["model"]["adapter"]["step"] == 8
    assert ev["model"]["stand_in"] and ev["image_longest_edge"] == 512
    for root in (ready / "results", ready / "download"):
        assert not list(root.rglob("*.safetensors")) and not list(root.rglob("*.pt"))


def test_stand_in_resumes_from_the_hub_after_the_disk_is_wiped(ready: Path, monkeypatch) -> None:
    needs_torch()
    from conftest import skip_on_usage_cap

    # The real (non-smoke) storage path, with the tiny model and the fake Hub.
    monkeypatch.setattr(run_stages, "_finetune_kind", lambda a: "")
    api = run_stages.make_hub_api()

    def fetch(repo, path, revision):
        out = ready / "hub_cache" / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(api.model_files[path])
        return out

    monkeypatch.setattr(run_stages, "fetch_model_file", fetch)
    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_train(targs(stop_after_epoch=1, gpu_hours_left=None))
    assert code == 0
    first = json.loads((ready / "results/finetune/lr2e-4/train.json").read_text())
    [e1] = first["epochs_published"]
    assert e1["adapter"]["stored"] == "private Hugging Face model repo"
    assert e1["adapter"]["path"] == "runs/lr2e-4/epoch-1"
    assert first["checkpoint"]["path"] == "runs/lr2e-4/checkpoint"
    assert "runs/lr2e-4/checkpoint/trainer_state.pt" in api.model_files
    assert "runs/lr2e-4/epoch-1/lumen_adapter.json" in api.model_files

    shutil.rmtree(ready / "checkpoints")  # a new Kaggle session: empty disk
    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_train(targs(gpu_hours_left=None))
    assert code == 0
    second = json.loads((ready / "results/finetune/lr2e-4/train.1.json").read_text())
    assert second["session"]["start_step"] == 4 and second["session"]["final_step"] == 8
    assert [e["epoch"] for e in second["epochs_published"]] == [2]
    # The epoch-1 adapter is found from the train results, fetched from the Hub, verified.
    spec, pointer = run_stages._epoch_adapter("lr2e-4", 1, "")
    assert pointer["revision"] == e1["adapter"]["revision"] and spec == first["spec"]
    shutil.rmtree(ready / "checkpoints")
    d = run_stages._adapter_dir("lr2e-4", 1, "", pointer)
    assert hub.folder_digest(d) == e1["adapter"]["files"]
    # With the results gone too, the Hub listing alone still locates it.
    shutil.rmtree(ready / "results" / "finetune")
    spec2, pointer2 = run_stages._epoch_adapter("lr2e-4", 1, "")
    assert spec2 == spec and pointer2["files"] == e1["adapter"]["files"]


def test_stand_in_stops_at_a_shared_deadline(ready: Path, monkeypatch) -> None:
    needs_torch()
    import time

    from conftest import skip_on_usage_cap

    monkeypatch.setenv("LUMEN_DEADLINE_UNIX", str(time.time() - 1))  # already past
    with skip_on_usage_cap("Hugging Face"):
        code = run_stages.cmd_train(targs())
    assert code == run_stages.EXIT_BUDGET
    r = result(ready, "lr2e-4", "train.json")
    assert r["status"] == "stopped:session limit" and r["session"]["final_step"] == 1
    monkeypatch.delenv("LUMEN_DEADLINE_UNIX")
    with skip_on_usage_cap("Hugging Face"):
        assert run_stages.cmd_train(targs()) == 0  # resumes from step 1
    assert result(ready, "lr2e-4", "train.1.json")["session"]["start_step"] == 1
