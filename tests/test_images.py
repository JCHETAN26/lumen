from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from lumen.data.images import ZipImages


def jpeg(mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (4, 3), 128 if mode == "L" else (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_reads_by_basename_and_converts_to_rgb(tmp_path: Path) -> None:
    z = make_zip(
        tmp_path / "train.zip",
        {
            "train/VizWiz_train_00000000.jpg": jpeg(),
            "train/VizWiz_train_00000001.jpg": jpeg("L"),
            "__MACOSX/train/._VizWiz_train_00000000.jpg": b"junk",
        },
    )
    images = ZipImages(z)
    assert len(images) == 2
    assert "VizWiz_train_00000001.jpg" in images
    im = images.open("VizWiz_train_00000001.jpg")
    assert im.mode == "RGB" and im.size == (4, 3)
    with pytest.raises(KeyError):
        images.read_bytes("VizWiz_train_99999999.jpg")


def test_duplicate_across_zips_rejected(tmp_path: Path) -> None:
    a = make_zip(tmp_path / "a.zip", {"x/VizWiz_train_00000000.jpg": jpeg()})
    b = make_zip(tmp_path / "b.zip", {"y/VizWiz_train_00000000.jpg": jpeg()})
    with pytest.raises(ValueError, match="twice"):
        ZipImages(a, b)


def test_pickles_without_open_handles(tmp_path: Path) -> None:
    z = make_zip(tmp_path / "val.zip", {"val/VizWiz_val_00000000.jpg": jpeg()})
    images = ZipImages(z)
    images.read_bytes("VizWiz_val_00000000.jpg")  # opens a handle
    clone = pickle.loads(pickle.dumps(images))
    assert clone.open("VizWiz_val_00000000.jpg").size == (4, 3)
