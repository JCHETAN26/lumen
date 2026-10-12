#!/usr/bin/env python3
"""Stage 4, step 2 (Mac): accuracy and latency of each MLX variant, then the summary table.

    uv run --group gpu --group mlx python scripts/mac_bench.py self-test --model lr2e-4-e1
    uv run --group mlx python scripts/mac_bench.py bench --model lr2e-4-e1 --variant fp16
    uv run --group mlx python scripts/mac_bench.py bench --model lr2e-4-e1 --variant 4bit
    uv run --group mlx python scripts/mac_bench.py summary --model lr2e-4-e1 \\
        --kaggle-result results/finetune/lr2e-4/val-e1.json

`self-test` must pass first: on two sample questions it checks the MLX P(unanswerable) (KV cache)
against uncached forward passes, and the prompt's token IDs against the transformers processor
the Kaggle runs used. `bench` refuses to start while the Mac is throttled (it waits for it to
cool first). Run one variant at a time, plugged in, with nothing else heavy running. The summary
compares every variant with MLX fp16 on the same 300 questions; `--kaggle-result` adds Kaggle
fp16 on those questions as a reference (needs HF_TOKEN to fetch its predictions).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumen.ondevice.bench import (  # noqa: E402
    Throttled,
    accuracy_run,
    latency_run,
    summarize,
    wait_until_cool,
)
from lumen.ondevice.models import load_backend  # noqa: E402
from lumen.ondevice.sample import load_mac_sample  # noqa: E402
from lumen.ondevice.variants import VARIANTS, size_on_disk  # noqa: E402
from lumen.pins import SPLITS_DIR, load_pins  # noqa: E402
from lumen.runmeta import hardware_info, write_result  # noqa: E402
from lumen.zeroshot.prompts import chat_messages  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "data" / "mac_sample"
PRED_DIR = REPO_ROOT / "predictions" / "mac"
RESULTS = REPO_ROOT / "results" / "mac"
SELF_TEST_TOLERANCE = 2e-3  # absolute, on a probability (fp16 cache vs uncached recompute)


def newest(stem: str) -> dict | None:
    """The newest result named stem.json, stem.1.json, ... (write_result never overwrites)."""
    found = []
    for f in RESULTS.glob(f"{stem}*.json"):
        rest = f.name[len(stem) :].removesuffix(".json")
        if rest == "":
            found.append((0, f))
        elif rest.startswith(".") and rest[1:].isdigit():
            found.append((int(rest[1:]), f))
    return json.loads(max(found)[1].read_text()) if found else None


def thermal() -> dict:
    h = hardware_info()
    return {k: h.get(k) for k in ("power_source", "cpu_speed_limit_pct")}


def describe_mismatch(tokenizer, hf_ids: list[int], mlx_ids: list[int], size) -> dict:
    """Where two prompts' token IDs part ways. Only special tokens (image and template markers)
    are shown by name; question text is masked, so nothing from VizWiz is printed or stored."""
    special = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)

    def show(ids: list[int]) -> list[str]:
        return [tokenizer.convert_ids_to_tokens(i) if i in special else "·" for i in ids]

    def counts(ids: list[int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in ids:
            if i in special:
                t = tokenizer.convert_ids_to_tokens(i)
                out[t] = out.get(t, 0) + 1
        return out

    n = min(len(hf_ids), len(mlx_ids))
    at = next((k for k in range(n) if hf_ids[k] != mlx_ids[k]), n)
    lo = max(0, at - 4)
    return {
        "image_size": list(size),
        "tokens_transformers": len(hf_ids),
        "tokens_mlx": len(mlx_ids),
        "first_difference_at": at,
        "transformers_from_there": show(hf_ids[lo : at + 8]),
        "mlx_from_there": show(mlx_ids[lo : at + 8]),
        "special_counts_transformers": counts(hf_ids),
        "special_counts_mlx": counts(mlx_ids),
    }


def cmd_self_test(args: argparse.Namespace) -> int:
    from transformers import SmolVLMProcessor

    from lumen.zeroshot.backends import long_edge_kwargs

    records, images = load_mac_sample(SAMPLE_DIR, SPLITS_DIR / "mac_sample_val.txt")
    backend, s, _, _ = load_backend(args.model, args.variant)
    pin = load_pins().models[s["describe"]["model_key"]]
    # Not AutoProcessor: importing mlx-vlm patches AutoProcessor.from_pretrained to return
    # mlx-vlm's own SmolVLM processor, which would make this check compare mlx-vlm with itself
    # (and, loaded from a Hub ID, with default settings). The Kaggle runs used this class.
    hf = SmolVLMProcessor.from_pretrained(pin.repo, revision=pin.revision)
    if not type(hf).__module__.startswith("transformers."):
        raise SystemExit(f"reference processor is {type(hf).__module__}, not transformers'")
    checks, ok = [], True
    for r in records[:2]:
        msgs = chat_messages(r["question"], s["prompt_key"])
        image = images.open(r["image"])
        enc = backend.encode(image, msgs)
        cached = backend.p_unanswerable(enc)
        uncached = backend.p_unanswerable_uncached(enc)
        text = hf.apply_chat_template(msgs, add_generation_prompt=True)
        hf_ids = hf(
            text=[text], images=[[image]], **long_edge_kwargs(hf, s["long_edge"])
        ).input_ids[0]
        mlx_ids = enc["input_ids"][0].tolist()
        hf_ids = [int(i) for i in hf_ids]
        c = {
            "p_cached_vs_uncached_abs_diff": abs(cached - uncached),
            "prompt_ids_match_transformers": hf_ids == mlx_ids,
            "prompt_tokens": len(mlx_ids),
        }
        if hf_ids != mlx_ids:
            c["mismatch"] = describe_mismatch(hf.tokenizer, hf_ids, mlx_ids, image.size)
        ok &= c["p_cached_vs_uncached_abs_diff"] <= SELF_TEST_TOLERANCE
        ok &= c["prompt_ids_match_transformers"]
        checks.append(c)
    path = write_result(
        RESULTS / f"self_test_{args.model}_{args.variant}.json",
        {
            "stage": "mac_self_test",
            "model": args.model,
            "variant": args.variant,
            "passed": ok,
            "tolerance": SELF_TEST_TOLERANCE,
            "checks": checks,
            "thermal": thermal(),
        },
    )
    print(json.dumps(checks, indent=1))
    print(("PASSED" if ok else "FAILED"), "->", path.relative_to(REPO_ROOT))
    return 0 if ok else 1


def cmd_bench(args: argparse.Namespace) -> int:
    import mlx.core as mx

    test = newest(f"self_test_{args.model}_fp16")
    if not (test and test["passed"]) and not args.skip_self_test_check:
        raise SystemExit("run `self-test` first (and check it passes) for this model")
    try:
        cool = wait_until_cool(
            lambda: hardware_info().get("cpu_speed_limit_pct"),
            max_wait_s=0 if args.allow_throttled else args.cool_wait_minutes * 60,
        )
    except Throttled as e:
        if not args.allow_throttled:
            print(f"NOT STARTED: {e}")
            return 3
        cool = {"cpu_speed_limit_pct": hardware_info().get("cpu_speed_limit_pct"), "waited_s": 0}
    before = thermal()
    records, images = load_mac_sample(SAMPLE_DIR, SPLITS_DIR / "mac_sample_val.txt")
    backend, s, load_s, path = load_backend(args.model, args.variant)
    pred_path = PRED_DIR / args.model / f"{args.variant}.predictions.jsonl"
    accuracy = accuracy_run(
        backend, records, images, prompt_key=s["prompt_key"], pred_path=pred_path
    )
    mid = thermal()
    latency = latency_run(
        backend,
        records,
        images,
        prompt_key=s["prompt_key"],
        n=args.latency_questions,
        repeats=args.repeats,
    )
    payload = {
        "stage": "mac_bench",
        "model": args.model,
        "variant": VARIANTS[args.variant].as_dict(),
        "backend": backend.describe(),
        "image_longest_edge": s["long_edge"],
        "image_tokens": backend.image_tokens.as_dict(),
        "size_on_disk_bytes": size_on_disk(path),
        "load_seconds": round(load_s, 2),
        "accuracy": accuracy,
        "latency": latency,
        "mlx_peak_memory_gib": round(mx.get_peak_memory() / 2**30, 3),
        "process_max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3),
        "thermal": {"cool_down": cool, "before": before, "after_accuracy": mid, "after": thermal()},
    }
    out = write_result(RESULTS / f"bench_{args.model}_{args.variant}.json", payload)
    m = accuracy["metrics"]
    print(
        f"{args.model} {args.variant}: accuracy {m['accuracy']['overall']:.2f}, AP "
        f"{m['answerability']['average_precision']:.2f}; TTFT median "
        f"{latency['ttft_s']['median']:.2f} s (p90 {latency['ttft_s']['p90']:.2f}), total median "
        f"{latency['total_s']['median']:.2f} s -> {out.relative_to(REPO_ROOT)}"
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    records, _ = load_mac_sample(SAMPLE_DIR, SPLITS_DIR / "mac_sample_val.txt")
    benches, files = {}, {}
    for name in VARIANTS:
        found = newest(f"bench_{args.model}_{name}")
        if found:
            benches[name] = found
            files[name] = PRED_DIR / args.model / f"{name}.predictions.jsonl"
    kaggle = None
    if args.kaggle_result:
        from lumen import hub
        from lumen.zeroshot.harness import read_predictions

        doc = json.loads(Path(args.kaggle_result).read_text())
        kaggle = read_predictions(hub.download(doc["predictions"]))[1]
    table = summarize(records, benches, files, kaggle_preds=kaggle)
    table["kaggle_result"] = args.kaggle_result
    out = write_result(RESULTS / f"summary_{args.model}.json", {"stage": "mac_summary", **table})
    print(
        f"{'variant':<8}{'MiB':>7}{'acc':>8}{'dAcc [95% CI]':>24}{'AP':>8}{'agree':>7}"
        f"{'TTFT p50/p90':>15}{'total p50/p90':>15}{'peak GB':>9}"
    )
    for r in table["variants"]:
        v = r.get("vs_fp16")
        d = (
            f"{v['accuracy_diff']:+.2f} [{v['accuracy_diff_ci95']['low']:+.1f}, "
            f"{v['accuracy_diff_ci95']['high']:+.1f}]"
            if v
            else "(reference)"
        )
        agree = r.get("agreement_with_fp16", {}).get("exact_answer_agreement")
        ttft, tot = r["ttft_s"] or {}, r["total_s"] or {}
        print(
            f"{r['variant']:<8}{(r['size_on_disk_bytes'] or 0) / 2**20:>7.0f}"
            f"{r['accuracy']:>8.2f}{d:>24}{r['average_precision']:>8.2f}"
            f"{'' if agree is None else f'{100 * agree:.0f}%':>7}"
            f"{ttft.get('median', float('nan')):>8.2f}/{ttft.get('p90', float('nan')):<6.2f}"
            f"{tot.get('median', float('nan')):>8.2f}/{tot.get('p90', float('nan')):<6.2f}"
            f"{r['peak_memory_gb_max'] or float('nan'):>9.2f}"
        )
    print("wrote", out.relative_to(REPO_ROOT))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("self-test")
    st.add_argument("--model", required=True, help="an adapter key in pins, or 'base'")
    st.add_argument("--variant", choices=sorted(VARIANTS), default="fp16")
    st.set_defaults(fn=cmd_self_test)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    b.add_argument("--latency-questions", type=int, default=20)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--cool-wait-minutes", type=float, default=10.0)
    b.add_argument("--allow-throttled", action="store_true", help="record, don't wait")
    b.add_argument("--skip-self-test-check", action="store_true")
    b.set_defaults(fn=cmd_bench)
    sm = sub.add_parser("summary")
    sm.add_argument("--model", required=True)
    sm.add_argument("--kaggle-result", default=None, help="a finetune val result JSON")
    sm.set_defaults(fn=cmd_summary)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
