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


def download(pointer: dict[str, Any], token: str | None = None) -> Path:
    """Fetch a stored file at its pinned revision and verify its sha256 (e.g. for Stage 4)."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            pointer["repo"],
            pointer["path"],
            repo_type="dataset",
            revision=pointer["revision"],
            token=token or os.environ.get("HF_TOKEN"),
        )
    )
    if _sha256(path) != pointer["sha256"]:
        raise HubError(f"{pointer['path']}@{pointer['revision'][:8]}: sha256 mismatch")
    return path
