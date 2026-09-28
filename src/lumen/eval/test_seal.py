"""The sealed test split: exactly one final evaluation, ever.

VizWiz released its test answers publicly (VQA_test.json, 2026-04-28), so nothing external stops
us from peeking. This module does. Test answers can only be loaded through a `FinalTestRun`, and
opening one requires:

1. an explicit opt-in (`--final-test-evaluation` in the runner),
2. a clean git tree, so the record names exactly the code that produced the numbers, and
3. no existing record at `results/test/FINAL_TEST_RUN.json`.

Opening the run immediately *claims* the record (status "started", written with O_EXCL). A run
that crashes leaves the claim in place and further runs are refused: deciding whether a crashed
run counts is a human decision, recorded in docs/devlog.md, never an automatic retry. On success
the record is completed with the date, commit, and results. Commit it to the repo immediately.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from lumen.pins import REPO_ROOT
from lumen.runmeta import git_info, run_metadata

TEST_RECORD = REPO_ROOT / "results" / "test" / "FINAL_TEST_RUN.json"
OPT_IN_FLAG = "--final-test-evaluation"


class TestSealed(RuntimeError):
    __test__ = False  # not a pytest test class


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class FinalTestRun:
    """Proof that the single final test evaluation is in progress. Pass it to the test loader."""

    __test__ = False

    def __init__(self, record_path: Path, commit: str) -> None:
        self._path = record_path
        self.commit = commit
        self._completed = False

    @property
    def record_path(self) -> Path:
        return self._path

    def complete(self, results: dict[str, Any]) -> Path:
        if self._completed:
            raise TestSealed("this final test run is already recorded")
        record = json.loads(self._path.read_text(encoding="utf-8"))
        record.update(status="completed", completed_utc=_now(), results=results)
        self._path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        self._completed = True
        return self._path


def open_final_test_run(
    *,
    opt_in: bool,
    purpose: str,
    record_path: Path = TEST_RECORD,
    git: dict[str, Any] | None = None,
) -> FinalTestRun:
    """Claim the one final test evaluation, or raise TestSealed explaining why not."""
    if not opt_in:
        raise TestSealed(
            f"the test split is sealed; pass {OPT_IN_FLAG} to run the single final evaluation"
        )
    if record_path.exists():
        rec = json.loads(record_path.read_text(encoding="utf-8"))
        raise TestSealed(
            f"the final test evaluation already ran (status {rec.get('status')!r}, commit "
            f"{rec.get('commit')}, started {rec.get('started_utc')}); see {record_path}. "
            "Test is evaluated once. Report that result."
        )
    git = git if git is not None else git_info()
    if not git.get("commit"):
        raise TestSealed("not in a git checkout: the final test run must name its commit")
    if git.get("dirty"):
        raise TestSealed("working tree has uncommitted changes: commit first, then run test once")
    record = {
        "status": "started",
        "purpose": purpose,
        "commit": git["commit"],
        "started_utc": _now(),
        "meta": run_metadata(),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Mode "x" is O_EXCL: of two racing runs, only one can create the record.
        with open(record_path, "x", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2, default=str) + "\n")
    except FileExistsError:
        raise TestSealed(f"another run claimed the final test evaluation: {record_path}") from None
    return FinalTestRun(record_path, git["commit"])
