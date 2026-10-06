#!/usr/bin/env python3
"""Stage 4, step 1 (Mac): merge a pinned LoRA adapter into its base, convert to MLX at fp16,
8-bit, and 4-bit.

    uv run --group gpu --group mlx python scripts/mac_prepare.py --adapter lr2e-4-e1
    uv run --group gpu --group mlx python scripts/mac_prepare.py --base   # zero-shot reference

The adapter comes only from its pin in configs/pins.yaml (exact Hub commit, every file's sha256
verified; HF_TOKEN needs read access to the private adapter repo). The merge runs on the CPU in
fp32 with transformers + peft and is saved in fp16; mlx-vlm then converts it (quantizing the
language model and connector; the vision encoder stays fp16). Models go to the gitignored
`models/`; the result records sizes and the merged weights' sha256.

Run one heavy job at a time on the 8 GB Air.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumen import hub  # noqa: E402
from lumen.ondevice.variants import VARIANTS, size_on_disk  # noqa: E402
from lumen.pins import adapter_pointer, load_pins  # noqa: E402
from lumen.runmeta import write_result  # noqa: E402

MODELS = REPO_ROOT / "models"


def merged_checkpoint(key: str, out: Path) -> dict:
    """Base + adapter merged (or the base alone for key 'base'), saved as fp16 safetensors."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    pins = load_pins()
    if key == "base":
        base_key, adapter = "smolvlm2-500m", None
    else:
        adapter = adapter_pointer(key, pins)
        base_key = adapter["base"]
    pin = pins.models[base_key]
    model = AutoModelForImageTextToText.from_pretrained(
        pin.repo, revision=pin.revision, dtype=torch.float32
    )
    info: dict = {"base": {"key": pin.key, "repo": pin.repo, "revision": pin.revision}}
    if adapter is not None:
        from peft import PeftModel

        local = hub.download_folder(adapter, MODELS / "adapters" / key)
        model = PeftModel.from_pretrained(model, str(local)).merge_and_unload()
        info["adapter"] = {k: adapter[k] for k in ("key", "repo", "revision", "path", "files")}
    shutil.rmtree(out, ignore_errors=True)
    model.to(torch.float16).save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(pin.repo, revision=pin.revision).save_pretrained(out)
    info["merged_weights_sha256"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(out.glob("*.safetensors"))
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--adapter", help="an adapter key under `adapters:` in configs/pins.yaml")
    who.add_argument("--base", action="store_true", help="the base model, no adapter")
    ap.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    args = ap.parse_args()
    from mlx_vlm import convert

    key = "base" if args.base else args.adapter
    merged = MODELS / "merged" / key
    t0 = time.perf_counter()
    info = merged_checkpoint(key, merged)
    merge_s = time.perf_counter() - t0
    variants = {}
    for name in args.variants:
        v = VARIANTS[name]
        out = MODELS / "mlx" / key / name
        shutil.rmtree(out, ignore_errors=True)
        t = time.perf_counter()
        convert(
            hf_path=str(merged),
            mlx_path=str(out),
            quantize=v.bits is not None,
            q_bits=v.bits or 4,
            q_group_size=v.group_size or 64,
            dtype="float16",
        )
        variants[name] = {
            **v.as_dict(),
            "path": str(out.relative_to(REPO_ROOT)),
            "size_on_disk_bytes": size_on_disk(out),
            "convert_seconds": round(time.perf_counter() - t, 1),
        }
        print(f"{key} {name}: {variants[name]['size_on_disk_bytes'] / 2**20:.0f} MiB")
    path = write_result(
        REPO_ROOT / "results" / "mac" / f"prepare_{key}.json",
        {
            "stage": "mac_prepare",
            "model": key,
            **info,
            "merge_seconds": round(merge_s, 1),
            "variants": variants,
            "quantization_scope": "language model and connector; vision encoder kept in fp16 "
            "(mlx_vlm.utils.skip_multimodal_module)",
        },
    )
    print("wrote", path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
