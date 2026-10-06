"""The precisions Stage 4 compares. mlx-vlm quantizes the language model and connector; it leaves
the vision encoder in fp16 (`mlx_vlm.utils.skip_multimodal_module`), which results record."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Variant:
    name: str
    bits: int | None  # None = unquantized
    group_size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


VARIANTS = {
    "fp16": Variant("fp16", None),
    "8bit": Variant("8bit", 8, 64),
    "4bit": Variant("4bit", 4, 64),
}


def size_on_disk(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
