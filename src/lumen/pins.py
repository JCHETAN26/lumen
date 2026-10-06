"""Typed access to configs/pins.yaml, the single source of dataset and model revisions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PINS_PATH = REPO_ROOT / "configs" / "pins.yaml"
SPLITS_DIR = REPO_ROOT / "configs" / "splits"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class PinError(ValueError):
    pass


@dataclass(frozen=True)
class FilePin:
    name: str
    url: str
    size: int
    sha256: str | None
    etag: str | None = None

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ModelPin:
    key: str
    repo: str
    revision: str
    license: str
    role: str


@dataclass(frozen=True)
class Pins:
    files: dict[str, FilePin]
    models: dict[str, ModelPin]
    adapters: dict[str, ModelPin]
    eval_code_sha256: str
    seed: int
    dev_fraction: float
    mac_sample_size: int
    dev_select_size: int
    raw: dict[str, Any]


def _model(key: str, d: dict[str, Any]) -> ModelPin:
    rev = str(d.get("revision", ""))
    if not _GIT_SHA.match(rev):
        raise PinError(f"{key}: revision must be a full 40-hex commit SHA, got {rev!r}")
    return ModelPin(key, d["repo"], rev, d.get("license", "unknown"), d.get("role", ""))


def load_pins(path: Path = PINS_PATH) -> Pins:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    files = {}
    for name, d in raw["dataset"]["files"].items():
        sha = d.get("sha256")
        if sha is not None and not _SHA256.match(sha):
            raise PinError(f"{name}: sha256 must be 64 lowercase hex chars or null")
        if not str(d["url"]).startswith("https://"):
            raise PinError(f"{name}: url must be https")
        files[name] = FilePin(name, d["url"], int(d["size"]), sha, d.get("etag"))
    splits = raw["splits"]
    return Pins(
        files=files,
        models={k: _model(k, v) for k, v in raw["models"].items()},
        adapters={k: _model(k, v) for k, v in (raw.get("adapters") or {}).items()},
        eval_code_sha256=raw["dataset"]["eval_code"]["sha256"],
        seed=int(splits["seed"]),
        dev_fraction=float(splits["dev_fraction"]),
        mac_sample_size=int(splits["mac_sample_size"]),
        dev_select_size=int(splits["dev_select_size"]),
        raw=raw,
    )


ADAPTER_FIELDS = ("repo", "revision", "path", "base", "files", "image_longest_edge", "prompt_key")


def adapter_pointer(key: str, pins: Pins | None = None) -> dict[str, Any]:
    """A pinned LoRA adapter as a verified-download pointer (`lumen.hub.download_folder`), plus
    what it needs at inference: its base model key, image long edge, and prompt."""
    pins = pins or load_pins()
    entry = (pins.raw.get("adapters") or {}).get(key)
    if entry is None:
        raise PinError(f"no adapter {key!r} in configs/pins.yaml (pinned: {sorted(pins.adapters)})")
    missing = [f for f in ADAPTER_FIELDS if f not in entry]
    if missing:
        raise PinError(f"adapter {key}: missing {missing}")
    _model(key, entry)  # full commit SHA
    if entry["base"] not in pins.models:
        raise PinError(f"adapter {key}: base {entry['base']!r} is not a pinned model")
    files = entry["files"]
    if not files or not all(_SHA256.match(str(v)) for v in files.values()):
        raise PinError(f"adapter {key}: every file needs a sha256")
    return {**entry, "repo_type": "model", "key": key}
