"""Private Hugging Face Hub storage for files that must never enter git: prediction files.

Prediction files hold model outputs about photos blind users took, and outputs can repeat
private details from those photos. This repo will be public and git history is permanent, so
predictions live in a *private* Hub dataset repo instead (configs/pins.yaml `predictions`), and
results in git record only a pointer: repo, path, commit revision, sha256, and row count.

The Hub token (HF_TOKEN; on Kaggle, the HF_TOKEN secret) needs write access to that dataset repo.
`preflight` checks this *before* any GPU time is spent; `upload` refuses a public repo.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lumen.pins import load_pins


class HubError(RuntimeError):
    pass


def predictions_repo() -> str:
    repo = (load_pins().raw.get("predictions") or {}).get("repo")
    if not repo:
        raise HubError("configs/pins.yaml has no predictions.repo")
    return repo


def adapter_repo() -> str:
    repo = load_pins().raw.get("adapter_repo")
    if not repo:
        raise HubError("configs/pins.yaml has no adapter_repo")
    return repo


def make_api(token: str | None = None) -> Any:
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise HubError("HF_TOKEN is not set: predictions can't be stored (see kaggle/README.md)")
    return HfApi(token=token)


def _can_write(whoami: dict, repo_id: str, repo_type: str = "dataset") -> bool:
    """Best-effort check of the token's write permission from `whoami()`."""
    tok = (whoami.get("auth") or {}).get("accessToken") or {}
    role = tok.get("role")
    if role == "write":
        return True
    if role != "fineGrained":
        return False
    owner = repo_id.split("/", 1)[0]
    for scope in (tok.get("fineGrained") or {}).get("scoped", []):
        entity = scope.get("entity") or {}
        target = entity.get("name")
        applies = (entity.get("type") in (repo_type, "repo") and target == repo_id) or (
            entity.get("type") in ("user", "org") and target == owner
        )
        if applies and "repo.write" in scope.get("permissions", []):
            return True
    return False


def preflight(api: Any, repo_id: str, repo_type: str = "dataset") -> dict[str, Any]:
    """Fail before a GPU run unless the repo exists, is private, and the token can write to it."""
    try:
        info = api.repo_info(repo_id, repo_type=repo_type)
    except Exception as e:  # missing repo or no access: say what to do
        raise HubError(
            f"can't access the private {repo_type} repo {repo_id}: {type(e).__name__}. Create it "
            f"on huggingface.co (private {repo_type}) and give HF_TOKEN write access to it."
        ) from None
    if not getattr(info, "private", False):
        raise HubError(f"{repo_id} is PUBLIC: Lumen only writes to private repos")
    if not _can_write(api.whoami(), repo_id, repo_type):
        raise HubError(f"HF_TOKEN can't write to {repo_id}: give it repo.write on that {repo_type}")
    return {"repo": repo_id, "repo_type": repo_type, "private": True}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload(api: Any, repo_id: str, local: Path, path_in_repo: str, message: str) -> dict[str, Any]:
    """Upload one file; return the pointer that results record (revision = the Hub commit)."""
    preflight(api, repo_id)
    commit = api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=message,
    )
    revision = getattr(commit, "oid", None)
    if not revision or len(revision) != 40:
        raise HubError(f"upload of {path_in_repo} returned no commit revision")
    rows = sum(1 for line in local.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "repo": repo_id,
        "repo_type": "dataset",
        "path": path_in_repo,
        "revision": revision,
        "sha256": _sha256(local),
        "lines": rows,
    }


def list_files(
    api: Any, repo_id: str, prefix: str, repo_type: str = "dataset"
) -> tuple[str, list[str]]:
    """(commit SHA, paths under `prefix`) of a private repo, read at one exact commit (never a
    branch name), so everything fetched afterwards comes from the same revision."""
    info = api.repo_info(repo_id, repo_type=repo_type)
    if not getattr(info, "private", False):
        raise HubError(f"{repo_id} is not private; refusing to use it")
    sha = getattr(info, "sha", None)
    if not sha or len(sha) != 40:
        raise HubError(f"{repo_id}: the Hub returned no commit SHA")
    files = api.list_repo_files(repo_id, repo_type=repo_type, revision=sha)
    return sha, sorted(f for f in files if f.startswith(prefix))


def fetch(
    repo_id: str,
    path: str,
    revision: str,
    token: str | None = None,
    repo_type: str = "dataset",
) -> Path:
    """Download one file of a private repo at an exact commit; returns the local path."""
    if len(revision) != 40:
        raise HubError(f"fetch needs a full commit SHA, not {revision!r}")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id,
            path,
            repo_type=repo_type,
            revision=revision,
            token=token or os.environ.get("HF_TOKEN"),
        )
    )


# ------------------------------------------------------------------- adapters (model repo)


def folder_digest(local_dir: Path) -> dict[str, str]:
    """sha256 of every file under `local_dir`, by relative path (what a result records)."""
    return {
        str(p.relative_to(local_dir)): _sha256(p)
        for p in sorted(local_dir.rglob("*"))
        if p.is_file()
    }


def upload_folder(
    api: Any, repo_id: str, local_dir: Path, path_in_repo: str, message: str
) -> dict[str, Any]:
    """Upload a folder (a LoRA adapter or a resumable checkpoint) to the private *model* repo;
    return the pointer results record: repo, path, Hub commit revision, and each file's sha256."""
    preflight(api, repo_id, "model")
    commit = api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=message,
    )
    revision = getattr(commit, "oid", None)
    if not revision or len(revision) != 40:
        raise HubError(f"upload of {path_in_repo} returned no commit revision")
    return {
        "repo": repo_id,
        "repo_type": "model",
        "path": path_in_repo,
        "revision": revision,
        "files": folder_digest(local_dir),
    }


def pointer(repo_id: str, path_in_repo: str, revision: str, local: Path) -> dict[str, Any]:
    """The pointer results record for a file already on the Hub at `revision`."""
    rows = sum(1 for line in local.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "repo": repo_id,
        "repo_type": "dataset",
        "path": path_in_repo,
        "revision": revision,
        "sha256": _sha256(local),
        "lines": rows,
    }


def download(pointer: dict[str, Any], token: str | None = None) -> Path:
    """Fetch a stored file at its pinned revision and verify its sha256 (e.g. for Stage 4)."""
    path = fetch(pointer["repo"], pointer["path"], pointer["revision"], token)
    if _sha256(path) != pointer["sha256"]:
        raise HubError(f"{pointer['path']}@{pointer['revision'][:8]}: sha256 mismatch")
    return path


def download_folder(
    pointer: dict[str, Any],
    dest: Path,
    fetch_file: Callable[[str, str, str], Path] | None = None,
) -> Path:
    """Fetch every file a folder pointer lists, at its pinned revision, into `dest`, verifying each
    sha256. `fetch_file(repo, path, revision)` defaults to `fetch` on the model repo."""
    import shutil

    def default(repo: str, path: str, revision: str) -> Path:
        return fetch(repo, path, revision, repo_type=pointer.get("repo_type", "model"))

    get = fetch_file or default
    tmp = dest.with_name(f".tmp-{dest.name}")
    shutil.rmtree(tmp, ignore_errors=True)
    for rel, sha in pointer["files"].items():
        src = get(pointer["repo"], f"{pointer['path']}/{rel}", pointer["revision"])
        if _sha256(src) != sha:
            raise HubError(f"{pointer['path']}/{rel}@{pointer['revision'][:8]}: sha256 mismatch")
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, tmp / rel)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest
