"""The notebook's clone cell must never expose the GitHub token.

Runs the real cell from kaggle/notebook.ipynb against a stub `git` on PATH: first a git that
fails and echoes everything it was given (worst case), then one that succeeds. The token must not
appear in stdout/stderr, the exception, the formatted traceback, git's argv, or any file on disk.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
import traceback
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN = "github_pat_TESTTOKEN_0123456789abcdefXYZ"
BASIC = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()

STUB_GIT = r"""#!{python}
import json, os, sys
log = os.environ["STUB_GIT_LOG"]
header = os.environ.get("GIT_CONFIG_VALUE_0", "")
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:],
                        "header_key": os.environ.get("GIT_CONFIG_KEY_0"),
                        "header_is_basic": header.startswith("Authorization: Basic ")}}) + "\n")
args = sys.argv[1:]
if args[:1] == ["rev-parse"] or args[2:3] == ["rev-parse"]:
    print("abc1234")
    sys.exit(0)
if os.environ["STUB_GIT_MODE"] == "fail":
    # Worst case: git echoes its whole environment-provided header and argv on failure.
    sys.stderr.write("fatal: could not read from remote; " + header + " argv=" + " ".join(args))
    sys.exit(128)
if args[0] == "clone":
    os.makedirs(os.path.join(args[-1], ".git"))
sys.exit(0)
"""


def clone_cell_source() -> str:
    nb = json.loads((REPO_ROOT / "kaggle" / "notebook.ipynb").read_text())
    cells = [c for c in nb["cells"] if "clone" in c.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, "exactly one notebook cell must be tagged 'clone'"
    return "".join(cells[0]["source"])


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(STUB_GIT.format(python=sys.executable))
    git.chmod(git.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STUB_GIT_LOG", str(tmp_path / "git_calls.jsonl"))
    monkeypatch.setenv("LUMEN_KAGGLE_WORKDIR", str(tmp_path / "working"))
    secrets = types.ModuleType("kaggle_secrets")

    class UserSecretsClient:
        def get_secret(self, label: str) -> str:
            assert label == "GITHUB_TOKEN"
            return TOKEN

    secrets.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)
    return tmp_path


def run_cell() -> BaseException | None:
    try:
        exec(compile(clone_cell_source(), "<clone cell>", "exec"), {"__name__": "__main__"})
    except Exception as exc:
        return exc
    return None


def assert_no_secret(text: str, where: str) -> None:
    assert TOKEN not in text, f"token leaked in {where}"
    assert BASIC not in text, f"base64 token header leaked in {where}"


def assert_nothing_on_disk(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            assert_no_secret(path.read_text(errors="replace"), str(path))


def test_failed_clone_never_exposes_token(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("STUB_GIT_MODE", "fail")
    exc = run_cell()
    assert exc is not None, "a failed clone must raise"
    out, err = capfd.readouterr()
    assert_no_secret(out + err, "notebook output")
    assert_no_secret(str(exc), "exception message")
    assert_no_secret("".join(traceback.format_exception(exc)), "traceback")
    assert_nothing_on_disk(sandbox)  # includes the stub's log of git's argv
    assert isinstance(exc, RuntimeError) and "git clone failed" in str(exc)
    assert "***" in str(exc)  # the stub echoed the header; the cell redacted it


def test_successful_clone_and_pull_deliver_token_only_via_header(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("STUB_GIT_MODE", "ok")
    assert run_cell() is None  # clone
    assert run_cell() is None  # second run takes the pull branch
    out, err = capfd.readouterr()
    assert "repo at abc1234" in out
    assert_no_secret(out + err, "notebook output")
    assert_nothing_on_disk(sandbox)
    calls = [json.loads(line) for line in (sandbox / "git_calls.jsonl").read_text().splitlines()]
    auth_calls = [
        c for c in calls if c["argv"][0] in ("clone", "-C") and "rev-parse" not in c["argv"]
    ]
    assert [c["argv"][0] for c in auth_calls] == ["clone", "-C"]
    assert auth_calls[1]["argv"][2:4] == ["pull", "--ff-only"]
    for c in auth_calls:
        assert c["header_key"] == "http.https://github.com/.extraheader"
        assert c["header_is_basic"]
        assert all("@" not in a for a in c["argv"]), "credentials must not be in a URL"


def test_cell_source_never_builds_a_token_url() -> None:
    src = clone_cell_source()
    assert "x-access-token:{token}@" not in src
    assert "check=True" not in src  # CalledProcessError would print the full command
