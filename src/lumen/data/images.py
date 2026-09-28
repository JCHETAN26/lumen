"""Read VizWiz images straight out of the official zips.

Extracting train.zip (11.3 GB) next to the zip would need ~23 GB, more than Kaggle's 20 GB
persistent /kaggle/working. Zip members are individually addressable, so we never extract.

`ZipImages` is safe to hand to DataLoader worker processes: the zip handle is opened lazily and
reopened after a fork, never shared between processes.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from PIL import Image


class ZipImages:
    def __init__(self, *zip_paths: Path) -> None:
        if not zip_paths:
            raise ValueError("need at least one zip")
        self.zip_paths = [Path(p) for p in zip_paths]
        self._index: dict[str, tuple[int, str]] = {}
        for zi, path in enumerate(self.zip_paths):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base.endswith(".jpg") and not base.startswith("._"):
                        if base in self._index:
                            raise ValueError(f"image {base} appears twice across {self.zip_paths}")
                        self._index[base] = (zi, name)
        self._handles: dict[int, zipfile.ZipFile] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, image: str) -> bool:
        return image in self._index

    def names(self) -> list[str]:
        return sorted(self._index)

    def _handle(self, zi: int) -> zipfile.ZipFile:
        if os.getpid() != self._pid:  # forked into a worker: never reuse the parent's handles
            self._handles, self._pid = {}, os.getpid()
        if zi not in self._handles:
            self._handles[zi] = zipfile.ZipFile(self.zip_paths[zi])
        return self._handles[zi]

    def read_bytes(self, image: str) -> bytes:
        try:
            zi, member = self._index[image]
        except KeyError:
            raise KeyError(f"{image} is not in {[str(p) for p in self.zip_paths]}") from None
        return self._handle(zi).read(member)

    def open(self, image: str) -> Image.Image:
        """Decode to RGB (VizWiz includes some grayscale and CMYK JPEGs)."""
        with Image.open(io.BytesIO(self.read_bytes(image))) as im:
            return im.convert("RGB")

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles = {}

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_handles": {}}
