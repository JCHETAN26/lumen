"""End-to-end run of the CPU stages (data, mac_sample) on synthetic VizWiz-shaped zips."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from lumen.pins import FilePin, Pins

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_stages", REPO_ROOT / "kaggle" / "run_stages.py")
run_stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_stages)


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, "JPEG")
    return buf.getvalue()


def records(split: str, n: int) -> list[dict]:
    out = []
    for i in range(n):
        unans = i % 4 == 0
        out.append(
            {
                "image": f"VizWiz_{split}_{i:08d}.jpg",
                "question": "What is this?",
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
    ann = {"train": records("train", 40), "val": records("val", 24)}
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
        models={},
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
    return tmp_path


def test_data_then_mac_sample(env: Path) -> None:
    assert run_stages.cmd_data(Namespace()) == 0
    result = json.loads((env / "download" / "vizwiz_data.json").read_text())
    assert result["summary"]["train"]["n"] == 40
    assert result["splits"]["dev"] == {"n": 10, "status": "not committed yet"}
    assert all(not f["pinned"] for f in result["files"].values())
    assert (env / "download" / "splits" / "dev.txt").exists()
    assert result["meta"]["git"] is not None

    assert run_stages.cmd_mac_sample(Namespace()) == 0
    [tar_path] = (env / "download" / "mac_sample").glob("*.tar.gz")
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
        manifest = json.load(tar.extractfile(next(n for n in names if n.endswith("manifest.json"))))
    assert sum("/images/" in n for n in names) == 8
    assert len(manifest["images_sha256"]) == 8
    with pytest.raises(SystemExit, match="exists"):
        run_stages.cmd_mac_sample(Namespace())


def test_data_refuses_changed_committed_split(env: Path) -> None:
    (env / "committed_splits").mkdir()
    (env / "committed_splits" / "dev.txt").write_text("VizWiz_train_00000001\n")
    with pytest.raises(SystemExit, match="differs"):
        run_stages.cmd_data(Namespace())


def test_missing_annotated_image_stops(env: Path) -> None:
    with zipfile.ZipFile(env / "data" / "val.zip", "w") as z:  # rewrite without one image
        for r in records("val", 23):
            z.writestr(f"val/{r['image']}", jpeg())
    pins = run_stages.load_pins()
    size = (env / "data" / "val.zip").stat().st_size
    pins.files["val_images"] = FilePin("val_images", "https://example.invalid/val.zip", size, None)
    with pytest.raises(SystemExit, match="missing from zip"):
        run_stages.cmd_data(Namespace())
