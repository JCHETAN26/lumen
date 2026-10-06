"""The pinned 300-question Mac sample, read from its extracted archive (`data/mac_sample/`).

Every image is checked against the archive manifest's sha256, and the question IDs must equal the
committed `configs/splits/mac_sample_val.txt`, so every precision and every machine answers exactly
the same questions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from lumen.data.annotations import Record
from lumen.data.splits import read_ids


class SampleError(RuntimeError):
    pass


class DirImages:
    """The `ZipImages` interface (`open`, `__contains__`) over a folder of verified images."""

    def __init__(self, folder: Path, sha256: dict[str, str]) -> None:
        self.folder = folder
        self.sha256 = sha256

    def __contains__(self, image: str) -> bool:
        return image in self.sha256

    def path(self, image: str) -> Path:
        if image not in self.sha256:
            raise KeyError(f"{image} is not in the Mac sample")
        return self.folder / image

    def open(self, image: str) -> Image.Image:
        """Decode to RGB, like ZipImages (VizWiz has some grayscale and CMYK JPEGs)."""
        with Image.open(self.path(image)) as im:
            return im.convert("RGB")


def load_mac_sample(sample_dir: Path, ids_file: Path) -> tuple[list[Record], DirImages]:
    """Records (sorted by question ID) and their images; every file checked against the manifest."""
    roots = [p for p in sample_dir.iterdir() if p.is_dir()] if sample_dir.is_dir() else []
    if len(roots) != 1:
        raise SampleError(
            f"expected one extracted Mac sample under {sample_dir}, found {len(roots)}: extract "
            "the mac_sample archive from Kaggle there (see kaggle/README.md)"
        )
    root = roots[0]
    manifest = json.loads((root / "manifest.json").read_text())["images_sha256"]
    records = json.loads((root / "annotations.json").read_text())
    ids = read_ids(ids_file)
    by_id = {r["question_id"]: r for r in records}
    if sorted(by_id) != sorted(ids):
        raise SampleError(f"{root.name}: its questions differ from {ids_file.name}")
    images = DirImages(root / "images", manifest)
    for r in records:
        digest = hashlib.sha256(images.path(r["image"]).read_bytes()).hexdigest()
        if digest != manifest[r["image"]]:
            raise SampleError(f"{r['image']}: sha256 differs from the sample manifest")
    return sorted(records, key=lambda r: r["question_id"]), images
