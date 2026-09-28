"""The guard that keeps model prediction files out of git (staged, tracked, and history)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_no_predictions.py"
HEADER = json.dumps({"_header": {"prompt_key": "plain", "split": "val"}}) + "\n"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "README.md").write_text("hi\n")
    git(tmp_path, "add", "README.md")
    git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("val-plain.predictions.jsonl", '{"question_id": "x"}\n'),
        ("predictions/anything.txt", "x\n"),
        ("results/renamed.jsonl", HEADER),  # renamed file caught by its header
        ("results/sneaky.json", HEADER + '{"question_id": "a", "raw": "..."}\n'),
    ],
)
def test_staged_predictions_are_blocked(repo: Path, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    git(repo, "add", "-f", name)
    r = run(repo)
    assert r.returncode == 1 and "BLOCKED" in r.stderr and name in r.stderr


def test_normal_results_pass(repo: Path) -> None:
    (repo / "results").mkdir()
    (repo / "results" / "dev-plain.json").write_text(json.dumps({"metrics": {"n": 1}}))
    (repo / "results" / "log.jsonl").write_text('{"step": 1}\n')
    git(repo, "add", "results")
    assert run(repo).returncode == 0
    assert run(repo, "--tracked", "--history").returncode == 0


def test_history_is_checked(repo: Path) -> None:
    (repo / "a.predictions.jsonl").write_text("{}\n")
    git(repo, "add", "-f", "a.predictions.jsonl")
    git(repo, "commit", "-qm", "oops", "--no-verify")
    git(repo, "rm", "-q", "--cached", "a.predictions.jsonl")
    git(repo, "commit", "-qm", "remove", "--no-verify")
    assert run(repo, "--tracked").returncode == 0  # no longer tracked...
    r = run(repo, "--history")
    assert r.returncode == 1 and "in history: a.predictions.jsonl" in r.stderr  # ...but permanent


def test_pre_commit_hook_blocks_the_commit(repo: Path) -> None:
    shutil.copytree(REPO_ROOT / ".githooks", repo / ".githooks")
    (repo / "scripts").mkdir()
    shutil.copy(SCRIPT, repo / "scripts" / SCRIPT.name)
    git(repo, "config", "core.hooksPath", ".githooks")
    (repo / "x.predictions.jsonl").write_text("{}\n")
    git(repo, "add", "-f", "x.predictions.jsonl")
    r = subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "try"], capture_output=True, text=True
    )
    assert r.returncode != 0 and "BLOCKED" in r.stderr


def test_gitignore_excludes_predictions() -> None:
    for path in ("results/baseline/m/val-plain.predictions.jsonl", "predictions/x"):
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", path], capture_output=True
        )
        assert r.returncode == 0, f"{path} is not gitignored"


def test_this_repo_is_clean() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--tracked", "--history"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
