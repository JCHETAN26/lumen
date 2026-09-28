#!/usr/bin/env python3
"""Early check of the Mac side of Stage 4: run a pinned model through mlx-vlm on a few images.

Loads the model at its pinned revision (unquantized, as published), answers a handful of
questions from the pinned Mac sample (image sha256 checked against the sample's manifest), and
records load time, time to first token, decode speed, and peak memory. Deliberately small: one
warm-up plus N images, capped output length. This is a smoke check, not the Stage 4 benchmark.

Answers are scored with the official metric, but their text is not stored: the photos can
contain private details, and model outputs can repeat them.

    uv run --group mlx python scripts/mlx_smoke.py --model smolvlm2-500m --n 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumen.data.splits import read_ids  # noqa: E402
from lumen.eval.vizwiz_metric import vqa_accuracy  # noqa: E402
from lumen.pins import SPLITS_DIR, load_pins  # noqa: E402
from lumen.runmeta import hardware_info, write_result  # noqa: E402
from lumen.zeroshot.extract import extract_answer  # noqa: E402
from lumen.zeroshot.prompts import prompt_text  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "data" / "mac_sample"


def sample_items(n: int) -> list[tuple[dict, Path]]:
    [root] = [p for p in SAMPLE_DIR.iterdir() if p.is_dir()]
    manifest = json.loads((root / "manifest.json").read_text())["images_sha256"]
    anns = {r["question_id"]: r for r in json.loads((root / "annotations.json").read_text())}
    items = []
    for qid in read_ids(SPLITS_DIR / "mac_sample_val.txt")[:n]:
        rec = anns[qid]
        path = root / "images" / rec["image"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest[rec["image"]]:
            raise SystemExit(f"{rec['image']}: sha256 mismatch with the sample manifest")
        items.append((rec, path))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvlm2-500m")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt", default="abstain")
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    pin = load_pins().models[args.model]
    items = sample_items(args.n)
    thermal_before = {k: hardware_info()[k] for k in ("power_source", "cpu_speed_limit_pct")}

    t0 = time.perf_counter()
    model, processor = load(pin.repo, revision=pin.revision)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0
    mem_after_load = mx.get_active_memory() / 2**30

    def answer(rec: dict, path: Path) -> dict:
        prompt = apply_chat_template(
            processor, model.config, prompt_text(rec["question"], args.prompt), num_images=1
        )
        t = time.perf_counter()
        out = generate(
            model,
            processor,
            prompt,
            image=[str(path)],
            max_tokens=args.max_tokens,
            temperature=0.0,
            verbose=False,
        )
        wall = time.perf_counter() - t
        ex = extract_answer(out.text)
        return {
            "question_id": rec["question_id"],
            "answerable": rec["answerable"],
            "wall_s": round(wall, 3),
            "prompt_tokens": out.prompt_tokens,
            "generation_tokens": out.generation_tokens,
            "prompt_tps": round(out.prompt_tps, 1),
            "generation_tps": round(out.generation_tps, 1),
            "ttft_s_approx": round(out.prompt_tokens / out.prompt_tps, 3)
            if out.prompt_tps
            else None,
            "peak_memory_gib": round(out.peak_memory, 3),
            "vizwiz_accuracy": vqa_accuracy(ex.answer, [a["answer"] for a in rec["answers"]]),
            "said_unanswerable": ex.answer == "unanswerable",
        }

    warmup = answer(*items[0])  # first call compiles Metal kernels
    runs = [answer(rec, path) for rec, path in items]
    peak = mx.get_peak_memory() / 2**30
    payload = {
        "stage": "mlx_smoke",
        "model": {
            "key": pin.key,
            "repo": pin.repo,
            "revision": pin.revision,
            "weights": "as published (unquantized)",
            "param_dtypes": sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())}),
            "parameters": sum(v.size for _, v in tree_flatten(model.parameters())),
        },
        "prompt_key": args.prompt,
        "max_tokens": args.max_tokens,
        "load_seconds": round(load_s, 2),
        "active_memory_after_load_gib": round(mem_after_load, 3),
        "mlx_peak_memory_gib": round(peak, 3),
        "process_max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3),
        "warmup": warmup,
        "runs": runs,
        "thermal_before": thermal_before,
        "thermal_after": {k: hardware_info()[k] for k in ("power_source", "cpu_speed_limit_pct")},
        "note": "smoke check on 3 Mac-sample images; not a benchmark (no repeats, no variance)",
    }
    path = write_result(REPO_ROOT / "results" / "mac" / f"mlx_smoke_{pin.key}.json", payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "load_seconds",
                    "active_memory_after_load_gib",
                    "mlx_peak_memory_gib",
                    "process_max_rss_gib",
                )
            }
        )
    )
    for r in [warmup, *runs]:
        print(
            {
                k: r[k]
                for k in (
                    "wall_s",
                    "prompt_tokens",
                    "prompt_tps",
                    "generation_tokens",
                    "generation_tps",
                    "ttft_s_approx",
                    "vizwiz_accuracy",
                )
            }
        )
    print("wrote", path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
