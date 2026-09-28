"""End-to-end run of the CPU stages (data, mac_sample) on synthetic VizWiz-shaped zips."""

from __future__ import annotations

import json
import tarfile
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
from conftest import fake_records as records
from conftest import jpeg, run_stages

from lumen.pins import FilePin


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
