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
from conftest import FakeHubApi, run_stages

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
        prefetch_workers=2,  # every stage test also exercises the prefetch path
        long_edge=None,
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
        run_stages,
        "make_backend",
        lambda key, a: fakes.setdefault(
            f"{key}@{a.long_edge}", FakeBackend(image_longest_edge=a.long_edge)
        ),
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
    pred = ready / "predictions" / "baseline" / "smolvlm2-500m" / "dev-plain.predictions.jsonl"
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
            (ready / "results" / "baseline" / "stand-in" / model / "dev-abstain.json").read_text()
        )
        assert r["smoke"] and r["model"]["stand_in"] and r["metrics"]["n"] == 6
        assert 0 <= r["metrics"]["accuracy"]["overall"] <= 100
    assert not (ready / "results" / "baseline" / "smolvlm2-500m").exists()  # smoke kept apart


def test_predictions_go_to_private_hub_never_to_results(ready: Path) -> None:
    assert run_stages.cmd_baseline(args(prompt="plain")) == 0
    r = result(ready, "smolvlm2-500m", "dev-plain.json")
    ptr = r["predictions"]
    assert ptr["stored"] == "private Hugging Face dataset" and ptr["repo_type"] == "dataset"
    assert len(ptr["revision"]) == 40 and ptr["lines"] == 11  # header + 10 questions
    assert ptr["path"] == "baseline/smolvlm2-500m/dev-plain.predictions.jsonl"
    api = run_stages.make_hub_api()
    assert ptr["path"] in api.files
    for root in (ready / "results", ready / "download"):
        assert not list(root.rglob("*.predictions.jsonl")), f"predictions leaked into {root}"
    for f in (ready / "results").rglob("*.json"):  # no per-question rows in any result
        text = f.read_text()
        assert '"question_id"' not in text and '"raw":' not in text, f


@pytest.mark.parametrize(
    "api", [FakeHubApi(private=False), FakeHubApi(exists=False), FakeHubApi(role="read")]
)
def test_storage_preflight_refuses_before_any_gpu_time(ready: Path, monkeypatch, api) -> None:
    monkeypatch.setattr(run_stages, "make_hub_api", lambda: api)
    monkeypatch.setattr(run_stages, "make_backend", lambda *a: pytest.fail("loaded a model"))
    assert run_stages.cmd_baseline(args(prompt="plain")) == run_stages.EXIT_STORAGE
    assert not api.files


def test_upload_failure_is_recorded_then_rerun_uploads(ready: Path, monkeypatch) -> None:
    api = FakeHubApi()
    monkeypatch.setattr(run_stages, "make_hub_api", lambda: api)
    api.fail_upload = True  # preflight passes; the upload itself fails
    assert run_stages.cmd_baseline(args(prompt="plain")) == run_stages.EXIT_STORAGE
    failed = result(ready, "smolvlm2-500m", "dev-plain.json")["predictions"]
    assert failed["stored"] == "upload failed" and "hub unreachable" in failed["error"]
    api.fail_upload = False
    assert run_stages.cmd_baseline(args(prompt="plain")) == 0
    ok = result(ready, "smolvlm2-500m", "dev-plain.1.json")
    assert ok["run"]["new"] == 0 and ok["run"]["resumed"] == 10  # nothing re-answered
    assert ok["predictions"]["stored"] == "private Hugging Face dataset"


def test_smoke_runs_are_not_uploaded(ready: Path) -> None:
    assert run_stages.cmd_baseline(args(prompt="plain", limit=4)) == 0
    r = json.loads(
        (ready / "results" / "baseline" / "smoke" / "smolvlm2-500m" / "dev-plain.json").read_text()
    )
    assert r["predictions"]["stored"].startswith("local only")
    assert not run_stages.make_hub_api().files


def test_storage_check_covers_both_private_repos(env: Path, monkeypatch, capsys) -> None:
    assert run_stages.cmd_storage_check(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "cjitendr/lumen-predictions (dataset)" in out and "cjitendr/lumen-adapter (model)" in out
    monkeypatch.setattr(run_stages, "make_hub_api", lambda: FakeHubApi(private=False))
    assert run_stages.cmd_storage_check(argparse.Namespace()) == run_stages.EXIT_STORAGE


def test_stand_in_and_real_smoke_runs_never_share_files(ready: Path, monkeypatch) -> None:
    """Regression (Kaggle, 2026-09-28): a stand-in run and a --limit run of the same model wrote
    the same smoke predictions file; the second run hit HeaderMismatch."""
    from lumen.zeroshot.backends import FakeBackend

    class StandIn(FakeBackend):
        def describe(self):
            return {
                "backend": "transformers",
                "model_key": "smolvlm2-500m-stand-in",
                "stand_in": True,
            }

    monkeypatch.setattr(
        run_stages, "make_backend", lambda key, a: StandIn() if a.stand_in else FakeBackend()
    )
    assert (
        run_stages.cmd_baseline(
            args(prompt="abstain", stand_in=True, limit=4, device="cpu", gpu_hours_left=None)
        )
        == 0
    )
    assert run_stages.cmd_baseline(args(prompt="abstain", limit=8)) == 0
    base = ready / "predictions" / "baseline"
    assert (base / "stand-in" / "smolvlm2-500m" / "dev-abstain.predictions.jsonl").exists()
    assert (base / "smoke" / "smolvlm2-500m" / "dev-abstain.predictions.jsonl").exists()
    real = json.loads(
        (
            ready / "results" / "baseline" / "smoke" / "smolvlm2-500m" / "dev-abstain.json"
        ).read_text()
    )
    assert real["run"]["resumed"] == 0 and real["metrics"]["n"] == 8


def test_measured_rate_and_summary_table(ready: Path, capsys) -> None:
    assert run_stages.cmd_baseline(args()) == 0
    assert run_stages.cmd_baseline(args(prompt="plain", limit=4)) == 0
    r = result(ready, "smolvlm2-500m", "dev-plain.json")
    assert r["run"]["sec_per_question"] > 0
    out = capsys.readouterr().out
    assert "measured" in out and "s/question over 10 questions" in out
    assert run_stages.cmd_baseline_summary(argparse.Namespace()) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split()[:4] == ["kind", "model", "split", "prompt"]
    rows = [line.split() for line in table[1:]]
    assert {(row[0], row[3]) for row in rows} == {
        ("real", "abstain"),
        ("real", "plain"),
        ("smoke", "plain"),
    }
    assert all(row[1] == "smolvlm2-500m" and row[4] == "completed" for row in rows)


def test_summary_without_results(env: Path, capsys) -> None:
    assert run_stages.cmd_baseline_summary(argparse.Namespace()) == 0
    assert "no baseline results" in capsys.readouterr().out


# ------------------------------------------------ dev_select subset and automatic prefetch A/B


def test_data_stage_builds_dev_select_from_dev(env: Path) -> None:
    assert run_stages.cmd_data(argparse.Namespace()) == 0
    split = env / "results" / "splits"
    sel = split.joinpath("dev_select.txt").read_text().split()
    dev = set(split.joinpath("dev.txt").read_text().split())
    assert len(sel) == 4 and set(sel) <= dev
    assert (env / "download" / "splits" / "dev_select.txt").exists()


def test_prompt_choice_on_dev_select_feeds_val(ready: Path) -> None:
    assert run_stages.cmd_baseline(args(split="dev_select")) == 0
    for p in ("plain", "abstain"):
        assert result(ready, "smolvlm2-500m", f"dev_select-{p}.json")["metrics"]["n"] == 4
    assert (ready / "download" / "splits" / "dev_select.txt").exists()
    assert run_stages.cmd_baseline(args(split="val", prompt="dev-selected")) == 0
    [val] = list((ready / "results" / "baseline" / "smolvlm2-500m").glob("val-*.json"))
    sel = json.loads(val.read_text())["prompt_selection"]
    assert sel["split"] == "dev_select" and sel["rule"].startswith("max dev_select accuracy")
    assert all("dev_select-" in v["file"] for v in sel["dev"].values())


def test_committed_dev_select_must_match(ready: Path) -> None:
    (ready / "committed_splits" / "dev_select.txt").write_text("VizWiz_train_00000001\n")
    with pytest.raises(SystemExit, match="dev_select"):
        run_stages.cmd_baseline(args(split="dev_select", prompt="plain"))


def test_prefetch_auto_measures_both_and_answers_everything_once(ready: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_stages, "AB_BATCHES", 1)  # arms of 2 questions: fits the fake dev
    assert run_stages.cmd_baseline(args(prompt="plain", batch_size=2, prefetch_workers="auto")) == 0
    r = result(ready, "smolvlm2-500m", "dev-plain.json")
    ab = r["prefetch_ab"]
    assert ab["questions_per_arm"] == 2 and ab["chosen"] in (0, run_stages.PREFETCH_ON)
    assert r["prefetch_workers"] == ab["chosen"]
    assert r["gpu_hours"]["calibration"]["source"] == "prefetch A/B, chosen arm"
    assert r["run"]["new"] == 10 and r["run"]["resumed"] == 0 and r["metrics"]["n"] == 10
    lines = (
        (ready / "predictions" / "baseline" / "smolvlm2-500m" / "dev-plain.predictions.jsonl")
        .read_text()
        .splitlines()
    )
    qids = [json.loads(line)["question_id"] for line in lines[1:]]
    assert len(qids) == len(set(qids)) == 10


def test_prefetch_auto_with_too_few_questions_uses_default(ready: Path) -> None:
    assert run_stages.cmd_baseline(args(prompt="plain", prefetch_workers="auto")) == 0
    r = result(ready, "smolvlm2-500m", "dev-plain.json")
    assert r["prefetch_ab"] is None and r["prefetch_workers"] == run_stages.PREFETCH_ON


def test_workers_arg() -> None:
    assert run_stages._workers_arg("auto") == "auto" and run_stages._workers_arg("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        run_stages._workers_arg("-1")


# ------------------------------------------------------- image resolution probe (Stage 3 prep)


def probe(edges=(1536, 1024, 512)) -> None:
    """Prompt choice on dev_select at the default resolution, then the abstain-prompt probe."""
    assert run_stages.cmd_baseline(args(split="dev_select")) == 0
    for le in edges:
        assert (
            run_stages.cmd_baseline(args(split="dev_select", prompt="abstain", long_edge=le)) == 0
        )


def test_long_edge_runs_get_their_own_files_and_headers(ready: Path) -> None:
    probe(edges=(1024,))
    base = ready / "predictions" / "baseline" / "smolvlm2-500m"
    default = json.loads((base / "dev_select-abstain.predictions.jsonl").read_text().split("\n")[0])
    capped = json.loads(
        (base / "dev_select-abstain-le1024.predictions.jsonl").read_text().split("\n")[0]
    )
    assert "image_longest_edge" not in default["_header"]["model"]  # old files stay valid
    assert capped["_header"]["model"]["image_longest_edge"] == 1024
    r = result(ready, "smolvlm2-500m", "dev_select-abstain-le1024.json")
    assert r["image_longest_edge"] == 1024 and r["image_tokens"]["questions"] == 4
    assert r["gpu_hours"]["prior_estimate"]["stage"].endswith(":le1024")
    assert result(ready, "smolvlm2-500m", "dev_select-abstain.json")["image_longest_edge"] is None


def test_prompt_choice_ignores_resolution_probe_runs(ready: Path) -> None:
    probe(edges=(1024,))
    _, sel = run_stages._dev_selected_prompt("smolvlm2-500m")
    assert all("-le" not in v["file"] for v in sel["dev"].values())


def test_default_long_edge_is_the_default_run(ready: Path) -> None:
    assert run_stages.cmd_baseline(args(split="dev_select", prompt="plain", long_edge=2048)) == 0
    r = result(ready, "smolvlm2-500m", "dev_select-plain.json")
    assert r["image_longest_edge"] is None


def test_long_edge_only_for_smolvlm_tiling(ready: Path) -> None:
    with pytest.raises(SystemExit, match=r"lfm2\.5-vl-450m"):
        run_stages.cmd_baseline(args(model=["lfm2.5-vl-450m"], long_edge=1024))
    with pytest.raises(argparse.ArgumentTypeError):
        run_stages._long_edge_arg("768")  # rounds up to 1,024 tiles: not a distinct setting
    assert run_stages._long_edge_arg("chosen") == "chosen"
    assert run_stages._long_edge_arg("512") == 512


def test_lower_resolution_has_a_cheaper_prior() -> None:
    from lumen.zeroshot.estimates import estimate_run

    full = estimate_run("smolvlm2-500m", "dev_select", "abstain", 512)
    low = estimate_run("smolvlm2-500m", "dev_select", "abstain", 512, long_edge=1024)
    assert low.sec_per_item < full.sec_per_item
    assert estimate_run("smolvlm2-500m", "v", "p", 10, long_edge=512).sec_per_item > 0


@pytest.fixture
def degrading(ready: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake that answers wrong at 512 px and right at every larger cap."""
    fakes: dict[str, FakeBackend] = {}
    monkeypatch.setattr(
        run_stages,
        "make_backend",
        lambda key, a: fakes.setdefault(
            f"{key}@{a.long_edge}",
            FakeBackend(
                answer="cat" if a.long_edge == 512 else "dog", image_longest_edge=a.long_edge
            ),
        ),
    )
    return ready


def test_resolution_choice_takes_best_accuracy_then_fewest_tokens(degrading: Path, capsys) -> None:
    probe()
    assert run_stages.cmd_resolution_choice(argparse.Namespace()) == 0
    doc = json.loads((degrading / "results" / "stage3_prep" / "resolution_choice.json").read_text())
    arms = {a["long_edge"]: a for a in doc["arms"]}
    assert set(arms) == {2048, 1536, 1024, 512}
    # 2,048 / 1,536 / 1,024 tie at 100: the best is the one with the fewest tokens, 1,024. 512 is
    # far worse (outside the 1-point margin), so 1,024 is chosen.
    assert doc["best"] == 1024 and doc["chosen"]["long_edge"] == 1024
    assert doc["margin_points"] == 1.0 and "vs_best" not in arms[1024]
    assert not arms[512]["vs_best"]["within_margin"] and "ap_diff_ci95" in arms[512]["vs_best"]
    assert arms[2048]["vs_best"]["within_margin"]  # identical answers: CI [0, 0]
    assert arms[512]["accuracy"] < arms[1024]["accuracy"] == 100.0
    assert arms[512]["vs_default"]["accuracy_diff"] < 0
    assert arms[1024]["vs_default"]["accuracy_diff"] == 0
    assert "vs_default" not in arms[2048]
    assert doc["rule"].startswith("best = highest dev_select accuracy") and doc["n"] == 4
    assert (degrading / "download" / "stage3_prep" / "resolution_choice.json").exists()
    assert "chosen: 1024 px" in capsys.readouterr().out
    # The matched-resolution val run picks the choice up by name.
    assert (
        run_stages.cmd_baseline(args(split="val", prompt="dev-selected", long_edge="chosen")) == 0
    )
    val = result(degrading, "smolvlm2-500m", "val-abstain-le1024.json")
    assert val["image_longest_edge"] == 1024 and val["metrics"]["n"] == 24


def test_resolution_choice_needs_every_arm(ready: Path) -> None:
    probe(edges=(1024,))
    with pytest.raises(SystemExit, match=r"\[1536, 512\]"):
        run_stages.cmd_resolution_choice(argparse.Namespace())


def test_long_edge_chosen_needs_a_choice(ready: Path) -> None:
    with pytest.raises(SystemExit, match="no resolution choice"):
        run_stages.cmd_baseline(args(split="val", prompt="dev-selected", long_edge="chosen"))


def test_resolution_choice_fetches_missing_predictions_from_the_hub(
    degrading: Path, monkeypatch
) -> None:
    probe()
    local = (
        degrading
        / "predictions"
        / "baseline"
        / "smolvlm2-500m"
        / "dev_select-abstain.predictions.jsonl"
    )
    saved = degrading / "hub_copy.jsonl"
    saved.write_bytes(local.read_bytes())
    local.unlink()
    fetched = []
    monkeypatch.setattr(
        run_stages, "download_predictions", lambda ptr: fetched.append(ptr["path"]) or saved
    )
    assert run_stages.cmd_resolution_choice(argparse.Namespace()) == 0
    assert fetched == ["baseline/smolvlm2-500m/dev_select-abstain.predictions.jsonl"]


def test_resolution_choice_rejects_a_mismatched_predictions_header(
    degrading: Path, monkeypatch
) -> None:
    probe()
    base = degrading / "predictions" / "baseline" / "smolvlm2-500m"
    wrong = base / "dev_select-abstain-le1536.predictions.jsonl"
    swapped = degrading / "swapped.jsonl"
    swapped.write_bytes(wrong.read_bytes())
    (base / "dev_select-abstain-le1024.predictions.jsonl").unlink()
    monkeypatch.setattr(run_stages, "download_predictions", lambda ptr: swapped)
    with pytest.raises(SystemExit, match="header doesn't match long edge 1024"):
        run_stages.cmd_resolution_choice(argparse.Namespace())


def test_chosen_default_resolution_runs_nothing(ready: Path, monkeypatch, capsys) -> None:
    choice = ready / "results" / "stage3_prep"
    choice.mkdir(parents=True)
    (choice / "resolution_choice.json").write_text(json.dumps({"chosen": {"long_edge": 2048}}))
    monkeypatch.setattr(run_stages, "make_backend", lambda *a: pytest.fail("loaded a model"))
    assert (
        run_stages.cmd_baseline(args(split="val", prompt="dev-selected", long_edge="chosen")) == 0
    )
    assert "nothing to run" in capsys.readouterr().out
