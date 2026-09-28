from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.data.annotations import parse_annotations
from lumen.eval.test_seal import FinalTestRun, TestSealed, open_final_test_run

CLEAN = {"commit": "a" * 40, "dirty": False}
TEST_REC = {
    "image": "VizWiz_test_00000000.jpg",
    "question": "What is this?",
    "answers": [{"answer": "dog", "answer_confidence": "yes"}] * 10,
    "answer_type": "other",
    "answerable": 1,
}


def test_requires_explicit_opt_in(tmp_path: Path) -> None:
    with pytest.raises(TestSealed, match="--final-test-evaluation"):
        open_final_test_run(opt_in=False, purpose="x", record_path=tmp_path / "r.json", git=CLEAN)
    assert not (tmp_path / "r.json").exists()


def test_refuses_dirty_tree_or_no_commit(tmp_path: Path) -> None:
    for git in ({"commit": "a" * 40, "dirty": True}, {"commit": None, "dirty": False}):
        with pytest.raises(TestSealed):
            open_final_test_run(opt_in=True, purpose="x", record_path=tmp_path / "r.json", git=git)
    assert not (tmp_path / "r.json").exists()


def test_single_run_is_recorded_then_sealed(tmp_path: Path) -> None:
    path = tmp_path / "test" / "FINAL_TEST_RUN.json"
    run = open_final_test_run(opt_in=True, purpose="final", record_path=path, git=CLEAN)
    started = json.loads(path.read_text())
    assert started["status"] == "started" and started["commit"] == "a" * 40
    run.complete({"accuracy": 61.2})
    done = json.loads(path.read_text())
    assert done["status"] == "completed" and done["results"] == {"accuracy": 61.2}
    assert done["started_utc"] and done["completed_utc"] and done["meta"]["hardware"]
    with pytest.raises(TestSealed, match="already ran"):
        open_final_test_run(opt_in=True, purpose="again", record_path=path, git=CLEAN)
    with pytest.raises(TestSealed):
        run.complete({"accuracy": 99.0})
    assert json.loads(path.read_text())["results"] == {"accuracy": 61.2}


def test_crashed_run_still_seals(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    open_final_test_run(
        opt_in=True, purpose="final", record_path=path, git=CLEAN
    )  # never completed
    with pytest.raises(TestSealed, match="'started'"):
        open_final_test_run(opt_in=True, purpose="retry", record_path=path, git=CLEAN)


def test_test_answers_load_only_inside_the_final_run(tmp_path: Path) -> None:
    with pytest.raises(TestSealed):
        parse_annotations([TEST_REC], "test")
    # Questions without answers are not sealed (needed to build prompts, never scores).
    assert parse_annotations(
        [{"image": TEST_REC["image"], "question": "Q?"}], "test", require_answers=False
    )
    run = open_final_test_run(
        opt_in=True, purpose="final", record_path=tmp_path / "r.json", git=CLEAN
    )
    assert isinstance(run, FinalTestRun)
    assert parse_annotations([TEST_REC], "test", final_test_run=run)[0]["answerable"] == 1


def test_committed_repo_has_no_record_yet() -> None:
    from lumen.eval.test_seal import TEST_RECORD

    assert not TEST_RECORD.exists(), "the final test run happened: update this test and devlog"
