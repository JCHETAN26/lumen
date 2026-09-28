#!/usr/bin/env python3
"""Fail if a model predictions file is staged, tracked, or anywhere in git history.

Prediction files hold model outputs about photos blind users took and can repeat private
details. This repo will be public and git history is permanent, so they must never be
committed; they go to a private Hugging Face dataset repo instead (src/lumen/hub.py).

    python3 scripts/check_no_predictions.py              # staged files (pre-commit hook)
    python3 scripts/check_no_predictions.py --tracked --history   # CI

A file counts as predictions if its name ends in `.predictions.jsonl`, it lives under
`predictions/`, or (renamed) it is JSON Lines whose first line is a Lumen predictions header.
Standard library only, so the hook runs without the project environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def _names(out: bytes) -> list[str]:
    return [n for n in out.decode("utf-8", "replace").split("\0") if n]


def by_name(path: str) -> bool:
    return (
        path.endswith(".predictions.jsonl")
        or path.startswith("predictions/")
        or "/predictions/" in f"/{path}"
    )


def by_content(blob: bytes) -> bool:
    first = blob.split(b"\n", 1)[0].strip()
    if not first.startswith(b"{"):
        return False
    try:
        head = json.loads(first)
    except ValueError:
        return False
    return (
        isinstance(head, dict)
        and isinstance(head.get("_header"), dict)
        and "prompt_key" in head["_header"]
    )


def _index_blob(path: str) -> bytes:
    try:
        return _git("show", f":{path}")
    except subprocess.CalledProcessError:
        return b""


def check(paths: list[str], sniff: bool) -> list[str]:
    bad = []
    for path in paths:
        if by_name(path):
            bad.append(path)
        elif (
            sniff
            and path.endswith((".jsonl", ".json", ".txt", ".ndjson"))
            and by_content(_index_blob(path)[:65536])
        ):
            bad.append(f"{path} (content is a predictions file)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracked", action="store_true", help="check every tracked file")
    ap.add_argument("--history", action="store_true", help="check every path in git history")
    args = ap.parse_args()

    problems: list[str] = []
    if not (args.tracked or args.history):
        staged = _names(_git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"))
        problems += [f"staged: {p}" for p in check(staged, sniff=True)]
    if args.tracked:
        problems += [f"tracked: {p}" for p in check(_names(_git("ls-files", "-z")), sniff=True)]
    if args.history:
        log = _git("log", "--all", "--name-only", "--pretty=format:", "-z").replace(b"\n", b"\0")
        problems += [f"in history: {p}" for p in sorted(set(check(_names(log), sniff=False)))]
    if problems:
        print(
            "BLOCKED: model predictions must never enter git (they can repeat private details "
            "from users' photos; store them in the private Hub dataset, see src/lumen/hub.py):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "Unstage with: git rm --cached <file>. A file already in history needs the history "
            "rewritten before this repo is made public.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
