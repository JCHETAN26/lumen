from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumen.pins import PinError, load_pins
from lumen.runmeta import run_metadata, write_result


def test_committed_pins_are_valid() -> None:
    pins = load_pins()
    assert {"annotations", "test_answers", "train_images", "val_images"} <= set(pins.files)
    assert pins.models["smolvlm2-500m"].role == "finetune"
    assert all(len(m.revision) == 40 for m in pins.models.values())
    assert 0 < pins.dev_fraction < 1 and pins.mac_sample_size > 0


def test_branch_names_are_rejected_as_revisions(tmp_path: Path) -> None:
    text = (Path(__file__).parents[1] / "configs" / "pins.yaml").read_text()
    bad = tmp_path / "pins.yaml"
    bad.write_text(text.replace("7b375e1b73b11138ff12fe22c8f2822d8fe03467", "main"))
    with pytest.raises(PinError, match="commit SHA"):
        load_pins(bad)


def test_metadata_records_hardware_and_versions() -> None:
    meta = run_metadata()
    assert meta["git"]["commit"] is None or len(meta["git"]["commit"]) == 40
    assert meta["hardware"]["machine"]
    assert "numpy" in meta["libraries"] and meta["libraries"]["numpy"]


def test_write_result_never_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    first = write_result(p, {"value": 1})
    second = write_result(p, {"value": 2})
    assert first == p and second == tmp_path / "r.1.json"
    assert json.loads(p.read_text())["value"] == 1
    assert "meta" in json.loads(second.read_text())
