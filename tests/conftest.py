"""Shared test helpers."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from lumen.live import usage_cap_reason
from lumen.pins import FilePin, Pins, load_pins

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_stages", REPO_ROOT / "kaggle" / "run_stages.py")
run_stages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_stages)


@contextlib.contextmanager
def skip_on_usage_cap(service: str) -> Iterator[None]:
    """Turn a usage cap or rate limit from `service` into a visible skip; re-raise anything else."""
    try:
        yield
    except Exception as exc:
        reason = usage_cap_reason(exc)
        if reason is None:
            raise
        pytest.skip(f"SKIPPED: {service} usage cap: {reason}")


# ----------------------------------------------- fake VizWiz on disk for runner tests


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, "JPEG")
    return buf.getvalue()


def fake_records(split: str, n: int) -> list[dict]:
    out = []
    for i in range(n):
        unans = i % 4 == 0
        out.append(
            {
                "image": f"VizWiz_{split}_{i:08d}.jpg",
                "question": "Is it blurry?" if unans else "What is this?",
                "answers": [
                    {"answer": "unanswerable" if unans else "dog", "answer_confidence": "yes"}
                ]
                * 10,
                "answer_type": "unanswerable" if unans else ["other", "yes/no", "number"][i % 3],
                "answerable": 0 if unans else 1,
            }
        )
    return out


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    ann = {"train": fake_records("train", 40), "val": fake_records("val", 24)}
    with zipfile.ZipFile(data / "Annotations.zip", "w") as z:
        for split, recs in ann.items():
            z.writestr(f"Annotations/{split}.json", json.dumps(recs))
    for split, recs in ann.items():
        with zipfile.ZipFile(data / f"{split}.zip", "w") as z:
            for r in recs:
                z.writestr(f"{split}/{r['image']}", jpeg())

    def pin(name: str, filename: str) -> FilePin:
        return FilePin(
            name, f"https://example.invalid/{filename}", (data / filename).stat().st_size, None
        )

    pins = Pins(
        files={
            "annotations": pin("annotations", "Annotations.zip"),
            "train_images": pin("train_images", "train.zip"),
            "val_images": pin("val_images", "val.zip"),
        },
        models=load_pins().models,
        adapters={},
        eval_code_sha256="0" * 64,
        seed=5,
        dev_fraction=0.25,
        mac_sample_size=8,
        raw={},
    )
    monkeypatch.setattr(run_stages, "load_pins", lambda: pins)
    monkeypatch.setattr(run_stages, "DATA_DIR", data)
    monkeypatch.setattr(run_stages, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_stages, "DOWNLOAD_DIR", tmp_path / "download")
    monkeypatch.setattr(run_stages, "SPLITS_DIR", tmp_path / "committed_splits")
    monkeypatch.setattr(run_stages, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_stages, "_BACKENDS", {})
    return tmp_path
