#!/usr/bin/env python3
"""Kaggle runner for Lumen. All dataset downloads and all GPU work happen here, never on the Mac.

Stages (each writes result JSON under results/ and copies it to the download folder):

    env         print and save the machine, GPU, and library versions
    data        download + verify the pinned VizWiz files, validate annotations against the
                image zips, build the dev split and the Mac sample ID lists      (CPU only)
    mac_sample  package the ~300 pinned val images + their annotations for the Mac   (CPU only)
    storage_check  confirm HF_TOKEN can write to the private predictions + adapter repos
    baseline_summary  one line per baseline result: accuracy, CI, AP, s/question, GPU hours
    rebuild_baseline  re-create lost baseline result files from the predictions in the private
                Hugging Face dataset (CPU only; re-scores, never re-answers)
    resolution_choice  pick SmolVLM2-500M's image resolution from the dev probe runs
                (CPU only; reads the probe's predictions)
    train_speed  a short LoRA run on train_fit: s/step, memory, tokens/s, projected h/epoch (GPU)
    train       LoRA fine-tuning on train_fit (Stage 3; GPU); resumable across sessions through
                checkpoints in the private adapter repo; publishes each epoch's adapter
    finetune_eval  one run's epoch adapter on dev (all sweep epochs; picks its F1 threshold) or
                val (only the dev-chosen config and its replicates)                     (GPU)
    finetune_choice  pick the config and epoch on dev (lumen.train.selection)      (CPU only)
    finetune_compare  val table: fine-tuned seeds vs the matched zero-shot baseline, paired,
                and the no-unanswerable ablation                                   (CPU only)
    pin_adapter print the configs/pins.yaml entry for an epoch adapter (for the Mac) (CPU only)
    final_test  the single sealed test evaluation: the dev-chosen fine-tune and the matched
                zero-shot baseline, once, ever (--final-test-evaluation)                (GPU)

    baseline    zero-shot baselines (Stage 2) for the pinned models on dev or val        (GPU)
                --dry-run prints the GPU-hour plan without loading anything
                --long-edge N caps the image's long edge (SmolVLM2; the resolution probe)

Every GPU stage goes through the quota guard in
lumen.budget. The final test evaluation is sealed (lumen.eval.test_seal) and needs
--final-test-evaluation.

Only files in the download folder leave Kaggle: result JSONs, split ID lists, and the Mac sample.
Image zips stay on Kaggle.

    uv run python kaggle/run_stages.py env
    uv run python kaggle/run_stages.py data
    uv run python kaggle/run_stages.py mac_sample
    uv run python kaggle/run_stages.py baseline --model all --split dev --prompt both --dry-run
    uv run python kaggle/run_stages.py baseline --model smolvlm2-500m --split dev_select \
        --prompt abstain --long-edge 1024 --gpu-hours-left 8
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumen.data.annotations import Record, parse_annotations, summarize  # noqa: E402
from lumen.data.download import ensure_free_space, fetch_pinned  # noqa: E402
from lumen.data.images import ZipImages  # noqa: E402
from lumen.data.splits import (  # noqa: E402
    make_dev_select,
    make_dev_split,
    make_mac_sample,
    read_ids,
    write_ids,
)
from lumen.pins import SPLITS_DIR, load_pins  # noqa: E402
from lumen.runmeta import hardware_info, run_metadata, write_result  # noqa: E402
from lumen.zeroshot.estimates import DEFAULT_LONG_EDGE, PORTRAIT_TILES  # noqa: E402
from lumen.zeroshot.prompts import PROMPTS  # noqa: E402

ON_KAGGLE = Path("/kaggle/working").is_dir()
DOWNLOAD_DIR = Path(
    os.environ.get(
        "LUMEN_OUTPUT_DIR",
        "/kaggle/working/lumen-results" if ON_KAGGLE else REPO_ROOT / "results" / "_download",
    )
)
DATA_DIR = Path(
    os.environ.get(
        "LUMEN_DATA_DIR", "/kaggle/working/lumen-data" if ON_KAGGLE else REPO_ROOT / "data"
    )
)
# Prediction files never enter git or the download folder (lumen.hub): locally they live here
# (outside results/, gitignored off-Kaggle), and completed runs upload them to a private Hub repo.
PREDICTIONS_DIR = Path(
    os.environ.get(
        "LUMEN_PREDICTIONS_DIR",
        "/kaggle/working/lumen-predictions" if ON_KAGGLE else REPO_ROOT / "predictions",
    )
)
RESULTS = REPO_ROOT / "results"
# LoRA checkpoints: local only (gitignored off-Kaggle); adapters are published to the private Hub.
CHECKPOINT_DIR = Path(
    os.environ.get(
        "LUMEN_CHECKPOINT_DIR",
        "/kaggle/working/lumen-checkpoints" if ON_KAGGLE else REPO_ROOT / "checkpoints",
    )
)
SPLIT_FILES = {
    "train_fit": "train_fit.txt",
    "dev": "dev.txt",
    "mac_sample": "mac_sample_val.txt",
    "dev_select": "dev_select.txt",
}


def collect(path: Path, subdir: str = "") -> None:
    """Copy one output into the flat download folder, never overwriting what is there."""
    dest_dir = DOWNLOAD_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        print(f"  (download folder already has {dest.name}; left as is)")
        return
    shutil.copy2(path, dest)
    print(f"  -> {dest}")


def warn_if_gpu_attached(stage: str) -> None:
    gpus = hardware_info()["gpus"]
    if gpus:
        print(
            f"NOTE: {stage} is CPU-only but a GPU ({gpus[0]['name']}) is attached; it uses "
            "weekly GPU quota while this session runs. Prefer Accelerator: None for this stage.",
            file=sys.stderr,
        )


# ------------------------------------------------------------------------------------ env


def cmd_env(args: argparse.Namespace) -> int:
    meta = run_metadata()
    print(json.dumps(meta, indent=2))
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    collect(write_result(RESULTS / "kaggle" / "environment.json", {"stage": "env"}))
    return 0


# ----------------------------------------------------------------------------------- data


def _annotation_json(zip_path: Path, split: str) -> list:
    with zipfile.ZipFile(zip_path) as z:
        names = [
            n
            for n in z.namelist()
            if n.rsplit("/", 1)[-1] == f"{split}.json" and "__MACOSX" not in n
        ]
        if len(names) != 1:
            raise SystemExit(f"expected one {split}.json in {zip_path.name}, found {names}")
        return json.loads(z.read(names[0]))


def _check_images(records: list[Record], images: ZipImages, split: str) -> dict:
    annotated = {r["image"] for r in records}
    in_zip = set(images.names())
    missing = sorted(annotated - in_zip)
    if missing:
        raise SystemExit(
            f"{split}: {len(missing)} annotated images missing from zip: {missing[:3]}"
        )
    # Decode a fixed handful as a smoke test of the zip-backed reader.
    for name in sorted(annotated)[:5]:
        images.open(name)
    return {
        "annotated": len(annotated),
        "in_zip": len(in_zip),
        "unannotated_in_zip": len(in_zip - annotated),
    }


def cmd_data(args: argparse.Namespace) -> int:
    warn_if_gpu_attached("data")
    pins = load_pins()
    wanted = ["annotations", "train_images", "val_images"]  # test is fetched only by the final run
    ensure_free_space(
        DATA_DIR,
        sum(pins.files[k].size for k in wanted if not (DATA_DIR / pins.files[k].filename).exists()),
    )
    verified = {k: fetch_pinned(pins.files[k], DATA_DIR) for k in wanted}

    observed = {
        k: {"sha256": v.sha256, "size": v.size, "pinned": v.pinned} for k, v in verified.items()
    }
    unpinned = [k for k, v in verified.items() if not v.pinned]
    if unpinned:
        print(
            f"\nTRUST ON FIRST USE: {unpinned} had no sha256 pin. Commit these hashes to "
            "configs/pins.yaml (from observed_hashes.json):"
        )
        for k in unpinned:
            print(f"  {k}: {verified[k].sha256}")

    ann_zip = verified["annotations"].path
    train = parse_annotations(_annotation_json(ann_zip, "train"), "train")
    val = parse_annotations(_annotation_json(ann_zip, "val"), "val")

    image_checks = {
        "train": _check_images(train, ZipImages(verified["train_images"].path), "train"),
        "val": _check_images(val, ZipImages(verified["val_images"].path), "val"),
    }

    fit, dev = make_dev_split(train, pins.dev_fraction, pins.seed)
    mac = make_mac_sample(val, pins.mac_sample_size, pins.seed)
    dev_ids = set(dev)
    dev_select = make_dev_select(
        [r for r in train if r["question_id"] in dev_ids], pins.dev_select_size, pins.seed
    )
    generated = {"train_fit": fit, "dev": dev, "mac_sample": mac, "dev_select": dev_select}
    split_check = {}
    for key, ids in generated.items():
        out = RESULTS / "splits" / SPLIT_FILES[key]
        write_ids(out, ids)
        collect(out, "splits")
        committed = SPLITS_DIR / SPLIT_FILES[key]
        split_check[key] = (
            "matches committed"
            if committed.exists() and read_ids(committed) == ids
            else "DIFFERS FROM COMMITTED"
            if committed.exists()
            else "not committed yet"
        )
        if split_check[key] == "DIFFERS FROM COMMITTED":
            raise SystemExit(f"{key}: regenerated split differs from {committed}; pins changed?")

    path = write_result(
        RESULTS / "data" / "vizwiz_data.json",
        {
            "stage": "data",
            "files": observed,
            "summary": {"train": summarize(train), "val": summarize(val)},
            "image_checks": image_checks,
            "splits": {k: {"n": len(v), "status": split_check[k]} for k, v in generated.items()},
            "seed": pins.seed,
            "dev_fraction": pins.dev_fraction,
        },
    )
    collect(path)
    hashes = write_result(RESULTS / "data" / "observed_hashes.json", {"files": observed})
    collect(hashes)
    print(
        json.dumps(
            {"summary": {"train": summarize(train), "val": summarize(val)}, "splits": split_check},
            indent=2,
        )
    )
    return 0


# ----------------------------------------------------------------------------- mac_sample


def cmd_mac_sample(args: argparse.Namespace) -> int:
    warn_if_gpu_attached("mac_sample")
    pins = load_pins()
    committed = SPLITS_DIR / SPLIT_FILES["mac_sample"]
    generated = RESULTS / "splits" / SPLIT_FILES["mac_sample"]
    source = committed if committed.exists() else generated
    if not source.exists():
        raise SystemExit("no Mac sample ID list: run the data stage first")
    ids = read_ids(source)
    if committed.exists() and generated.exists() and read_ids(generated) != ids:
        raise SystemExit("generated Mac sample differs from the committed one")

    val_zip = DATA_DIR / pins.files["val_images"].filename
    ann_zip = DATA_DIR / pins.files["annotations"].filename
    for p in (val_zip, ann_zip):
        if not p.exists():
            raise SystemExit(f"{p} missing: run the data stage first")
    by_id = {
        r["question_id"]: r for r in parse_annotations(_annotation_json(ann_zip, "val"), "val")
    }
    records = [by_id[q] for q in ids]
    images = ZipImages(val_zip)

    name = f"mac_sample_val_n{len(ids)}_seed{pins.seed}"
    tar_path = RESULTS / "mac_sample" / f"{name}.tar.gz"
    if tar_path.exists():
        raise SystemExit(f"{tar_path} exists; delete it deliberately to rebuild")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tarfile.open(tar_path, "w:gz") as tar:

        def add(arcname: str, data: bytes) -> None:
            info = tarfile.TarInfo(f"{name}/{arcname}")
            info.size, info.mtime = len(data), 0  # fixed mtime: same content -> same bytes
            tar.addfile(info, io.BytesIO(data))

        for r in records:
            data = images.read_bytes(r["image"])
            manifest[r["image"]] = hashlib.sha256(data).hexdigest()
            add(f"images/{r['image']}", data)
        add("annotations.json", json.dumps(records, indent=1).encode())
        add(
            "manifest.json",
            json.dumps(
                {
                    "ids_source": str(source.relative_to(REPO_ROOT)),
                    "seed": pins.seed,
                    "images_sha256": manifest,
                },
                indent=1,
            ).encode(),
        )

    tar_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    result = write_result(
        RESULTS / "mac_sample" / f"{name}.json",
        {
            "stage": "mac_sample",
            "n": len(ids),
            "archive": tar_path.name,
            "archive_sha256": tar_sha,
            "archive_bytes": tar_path.stat().st_size,
            "ids_source": str(source.relative_to(REPO_ROOT)),
            "summary": summarize(records),
        },
    )
    collect(tar_path, "mac_sample")
    collect(result)
    print(
        f"Mac sample: {len(ids)} images, {tar_path.stat().st_size / 2**20:.1f} MiB, "
        f"sha256 {tar_sha}"
    )
    return 0


# ------------------------------------------------------------------------------- baseline

BASELINE_DIR = "baseline"
PREFETCH_ON = 2  # workers used when prefetching is on (`auto` compares this against 0)
AB_BATCHES = 4  # batches per arm of the automatic prefetch on/off comparison
EXIT_BUDGET, EXIT_NON_FINITE, EXIT_STORAGE, EXIT_REBUILD = 3, 4, 5, 6
SESSION_START = time.monotonic()


def _dev_select_ids(dev_records: list[Record], pins) -> list[str]:
    """The fixed prompt-choice subset of dev. Regenerated deterministically from the committed dev
    IDs + pinned seed every time; if a committed copy exists it must match exactly. The generated
    list is written to the download folder so it can be committed like the other splits."""
    ids = make_dev_select(dev_records, pins.dev_select_size, pins.seed)
    out = RESULTS / "splits" / SPLIT_FILES["dev_select"]
    write_ids(out, ids)
    collect(out, "splits")
    committed = SPLITS_DIR / SPLIT_FILES["dev_select"]
    if committed.exists() and read_ids(committed) != ids:
        raise SystemExit(f"dev_select: regenerated subset differs from {committed}; pins changed?")
    return ids


def _load_split(
    split: str, pins, *, limit: int | None, images: bool = True
) -> tuple[list[Record], Path | None]:
    """Records and image zip for dev / dev_select (committed dev IDs, from official train) or
    val (all). With images=False only the annotations (1.7 MB) are fetched, not the image zip."""
    ann = fetch_pinned(pins.files["annotations"], DATA_DIR).path
    if split in ("dev", "dev_select"):
        ids = set(read_ids(SPLITS_DIR / SPLIT_FILES["dev"]))
        records = [
            r
            for r in parse_annotations(_annotation_json(ann, "train"), "train")
            if r["question_id"] in ids
        ]
        if len(records) != len(ids):
            raise SystemExit(f"dev: {len(ids) - len(records)} committed IDs not in train.json")
        if split == "dev_select":
            keep = set(_dev_select_ids(records, pins))
            records = [r for r in records if r["question_id"] in keep]
        zip_pin = pins.files["train_images"]
    elif split == "val":
        records = parse_annotations(_annotation_json(ann, "val"), "val")
        zip_pin = pins.files["val_images"]
    else:
        raise SystemExit(
            f"baseline runs on dev, dev_select, or val, not {split!r} (test is sealed)"
        )
    records.sort(key=lambda r: r["question_id"])
    if limit is not None:
        records = records[:limit]
    return records, fetch_pinned(zip_pin, DATA_DIR).path if images else None


def _run_kind(args: argparse.Namespace) -> str:
    """ "" for real runs; smoke runs are kept apart, and stand-in smoke runs apart from real-model
    smoke runs (same model key, different weights: their prediction files must never meet)."""
    if args.stand_in:
        return "stand-in"
    return "smoke" if args.limit is not None else ""


def _run_dir(model_key: str, kind: str) -> Path:
    return RESULTS / BASELINE_DIR / kind / model_key


def _run_name(split: str, prompt_key: str, long_edge: int | None = None) -> str:
    """dev_select-abstain, or dev_select-abstain-le1024 with a capped image resolution."""
    return f"{split}-{prompt_key}" + ("" if long_edge is None else f"-le{long_edge}")


def _pred_path(
    model_key: str, split: str, prompt_key: str, kind: str, long_edge: int | None = None
) -> Path:
    return (
        PREDICTIONS_DIR
        / BASELINE_DIR
        / kind
        / model_key
        / f"{_run_name(split, prompt_key, long_edge)}.predictions.jsonl"
    )


def make_hub_api():
    """Tests replace this with a fake Hub client."""
    from lumen import hub

    return hub.make_api()


def _result_files(run_dir: Path, name: str) -> list[Path]:
    """name.json, name.1.json, name.2.json, ...: oldest first (write_result never overwrites)."""
    numbered = [f for f in run_dir.glob(f"{name}.*.json") if f.stem.split(".")[1].isdigit()]
    plain = [run_dir / f"{name}.json"] if (run_dir / f"{name}.json").exists() else []
    return plain + sorted(numbered, key=lambda f: int(f.stem.split(".")[1]))


def _dev_selected_prompt(model_key: str) -> tuple[str, dict]:
    """The prompt with the best accuracy on dev_select, the fixed 512-question dev subset (or on
    full dev if that's what was run); ties: higher AP, then name. Never val."""
    run_dir = _run_dir(model_key, "")
    sel_split = next(
        (
            sp
            for sp in ("dev_select", "dev")
            if all(_result_files(run_dir, _run_name(sp, k)) for k in PROMPTS)
        ),
        None,
    )
    if sel_split is None:
        raise SystemExit(
            f"{model_key}: no dev result for both prompts; "
            "run --split dev_select --prompt both first"
        )
    found = {}
    for key in PROMPTS:
        files = _result_files(run_dir, _run_name(sel_split, key))
        doc = json.loads(files[-1].read_text())
        found[key] = {
            "file": str(files[-1].relative_to(REPO_ROOT)),
            "accuracy": doc["metrics"]["accuracy"]["overall"],
            "average_precision": doc["metrics"]["answerability"]["average_precision"],
        }
    best = max(sorted(found), key=lambda k: (found[k]["accuracy"], found[k]["average_precision"]))
    return best, {
        "rule": f"max {sel_split} accuracy; ties by AP, then name",
        "split": sel_split,
        "dev": found,
        "chosen": best,
    }


def _make_backend(model_key: str, args: argparse.Namespace):
    import torch

    from lumen.zeroshot.backends import GenerationConfig, HFBackend, tiny_stand_in

    pin = load_pins().models[model_key]
    device = args.device or ("cuda" if torch.cuda.is_available() else None)
    if device is None:
        raise SystemExit("no CUDA GPU: attach a T4, or pass --device cpu (only for --stand-in)")
    gen = GenerationConfig()
    long_edge = args.long_edge
    if args.stand_in:
        return tiny_stand_in(pin, device=device, gen=gen, image_longest_edge=long_edge)
    dtype = "float16" if device == "cuda" else "float32"  # T4: no bf16
    return HFBackend.from_pin(
        pin, device=device, dtype=dtype, gen=gen, image_longest_edge=long_edge
    )


_BACKENDS: dict[str, object] = {}


def _cached_backend(model_key: str, args: argparse.Namespace):
    """Load each model once per process, even when it runs several prompts."""
    key = f"{model_key}@le{args.long_edge}"
    if key not in _BACKENDS:
        _BACKENDS[key] = _make_backend(model_key, args)
    return _BACKENDS[key]


# Tests replace this to run the whole stage with a fake or tiny backend.
make_backend = _cached_backend


def _baseline_one(
    model_key: str,
    split: str,
    prompt_key: str,
    args: argparse.Namespace,
    prompt_selection: dict | None,
) -> int:
    from lumen.budget import BudgetExceeded, calibrated_estimate, check_budget
    from lumen.zeroshot.backends import NonFiniteOutput
    from lumen.zeroshot.estimates import MODEL_COSTS, estimate_run
    from lumen.zeroshot.harness import (
        predictions_header,
        read_predictions,
        run_predictions,
        score_predictions,
    )

    pins = load_pins()
    kind = _run_kind(args)
    smoke = kind != ""
    records, zip_path = _load_split(split, pins, limit=args.limit)
    run_dir = _run_dir(model_key, kind)
    long_edge = args.long_edge
    name = _run_name(split, prompt_key, long_edge)
    pred_path = _pred_path(model_key, split, prompt_key, kind, long_edge)
    _, done = read_predictions(pred_path)
    remaining = len([r for r in records if r["question_id"] not in done])
    prior = estimate_run(model_key, split, prompt_key, remaining, long_edge)
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    res = "" if long_edge is None else f" | long edge {long_edge} px"
    print(
        f"\n== {model_key} | {split} | prompt={prompt_key}{res} | {remaining}/{len(records)} to go"
        f" | prior estimate {prior.hours:.2f} GPU h"
    )
    if not args.stand_in:
        check_budget(prior, left)

    backend = make_backend(model_key, args)
    if hasattr(backend, "image_tokens"):  # count this run only (backends are cached per model)
        backend.image_tokens = type(backend.image_tokens)()
    batch_size = args.batch_size or MODEL_COSTS[model_key].batch_size
    header = predictions_header(backend, prompt_key, split)
    calibration: dict = {}

    def on_batch(n_done: int, seconds: float) -> None:
        if calibration or n_done < args.calibration_items or n_done >= remaining:
            return
        cal = calibrated_estimate(prior, n_done, seconds)
        rest = calibrated_estimate(
            estimate_run(model_key, split, prompt_key, remaining - n_done, long_edge),
            n_done,
            seconds,
        )
        calibration.update(
            estimate=cal.as_dict(), measured_items=n_done, measured_seconds=round(seconds, 2)
        )
        print(f"   calibrated: {seconds / n_done:.3f} s/question -> {cal.hours:.2f} GPU h total")
        if not args.stand_in and left is not None:
            check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)

    images = ZipImages(zip_path)
    common = dict(prompt_key=prompt_key, split=split, out_path=pred_path, batch_size=batch_size)
    workers = getattr(args, "prefetch_workers", 0)
    prefetch_ab: dict | None = None
    seg_new, seg_seconds = 0, 0.0

    t0 = time.monotonic()
    status, error = "completed", None
    try:
        if workers == "auto":
            arm = batch_size * AB_BATCHES
            if remaining >= batch_size + 2 * arm + batch_size:
                # Warm-up batch (CUDA init, first-call costs) is answered but not timed. Then the
                # same number of real questions with prefetching on and off; keep the faster.
                _, warm = run_predictions(
                    backend, records, images, limit_new=batch_size, prefetch_workers=0, **common
                )
                _, on = run_predictions(
                    backend, records, images, limit_new=arm, prefetch_workers=PREFETCH_ON, **common
                )
                _, off = run_predictions(
                    backend, records, images, limit_new=arm, prefetch_workers=0, **common
                )
                r_on, r_off = on.seconds / on.new, off.seconds / off.new
                workers = PREFETCH_ON if r_on < r_off else 0
                seg_new = warm.new + on.new + off.new
                seg_seconds = warm.seconds + on.seconds + off.seconds
                prefetch_ab = {
                    "questions_per_arm": arm,
                    "on_workers": PREFETCH_ON,
                    "on_sec_per_question": round(r_on, 4),
                    "off_sec_per_question": round(r_off, 4),
                    "chosen": workers,
                }
                print(
                    f"   prefetch A/B: on {r_on:.3f} vs off {r_off:.3f} s/question "
                    f"-> prefetch_workers={workers}"
                )
                chosen = on if workers else off
                cal = calibrated_estimate(prior, chosen.new, chosen.seconds)
                rest = calibrated_estimate(
                    estimate_run(model_key, split, prompt_key, remaining - seg_new, long_edge),
                    chosen.new,
                    chosen.seconds,
                )
                calibration.update(
                    estimate=cal.as_dict(),
                    measured_items=chosen.new,
                    measured_seconds=round(chosen.seconds, 2),
                    source="prefetch A/B, chosen arm",
                )
                print(
                    f"   calibrated: {min(r_on, r_off):.3f} s/question -> "
                    f"{cal.hours:.2f} GPU h total"
                )
                if not args.stand_in and left is not None:
                    check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)
            else:
                workers = PREFETCH_ON  # too few questions to measure; use the default
        preds, stats = run_predictions(
            backend, records, images, on_batch=on_batch, prefetch_workers=workers, **common
        )
    except BudgetExceeded as e:
        print(f"STOPPED (budget): {e}\n   answered so far is saved in {pred_path}; rerun resumes.")
        return EXIT_BUDGET
    except NonFiniteOutput as e:
        status, error = "failed_non_finite", str(e)
        _, preds = read_predictions(pred_path)
        stats = None
    hours = (time.monotonic() - t0) / 3600
    image_tokens = getattr(backend, "image_tokens", None)

    payload = {
        "stage": "baseline",
        "status": status,
        "error": error,
        "smoke": smoke,
        "model": backend.describe(),
        "split": split,
        "prompt": header,
        "prompt_selection": prompt_selection,
        "image_longest_edge": long_edge,  # None = the processor's default (2,048 px for SmolVLM2)
        "image_tokens": image_tokens.as_dict() if image_tokens is not None else None,
        "batch_size": batch_size,
        "prefetch_workers": workers,
        "prefetch_ab": prefetch_ab,
        "gpu_hours": {
            "prior_estimate": prior.as_dict(),
            "calibration": calibration or None,
            "actual_this_run": round(hours, 4),
        },
        "run": None
        if stats is None
        else {
            "new": stats.new + seg_new,
            "resumed": stats.resumed - seg_new,
            "seconds": round(stats.seconds + seg_seconds, 1),
            # Steady-state GPU rate with the chosen prefetch setting (excludes model load and
            # the A/B segments): replaces the prior.
            "sec_per_question": round(stats.seconds / stats.new, 4) if stats.new else None,
        },
    }
    if status == "completed":
        payload["metrics"] = score_predictions(records, preds)
    else:
        payload["answered"] = len(preds)
    payload["predictions"], storage_error = _store_predictions(
        pred_path,
        f"{BASELINE_DIR}/{model_key}/{pred_path.name}",
        f"{model_key} {split} {prompt_key} predictions",
        smoke=smoke or status != "completed",
    )
    result = write_result(run_dir / f"{name}.json", payload)
    collect(result, str(Path(BASELINE_DIR) / kind / model_key))
    if status != "completed":
        print(f"FAILED: {error}\n   recorded in {result}")
        return EXIT_NON_FINITE
    if storage_error:
        print(
            f"STORAGE FAILED: {storage_error}\n   predictions kept at {pred_path}; the rerun "
            "resumes (nothing to answer), re-scores, and uploads again."
        )
        return EXIT_STORAGE
    m = payload["metrics"]
    print(
        f"   accuracy {m['accuracy']['overall']:.2f} "
        f"[{m['accuracy']['ci95']['low']:.2f}, {m['accuracy']['ci95']['high']:.2f}]  "
        f"AP {m['answerability']['average_precision']:.2f}  "
        f"said-unanswerable {100 * m['said_unanswerable_rate']:.1f}%  ({hours:.2f} h)"
    )
    if stats is not None and stats.new:
        print(
            f"   measured {stats.seconds / stats.new:.3f} s/question over {stats.new} questions "
            f"(batch {batch_size}; prior was {prior.sec_per_item:.2f})"
        )
    return 0


def _store_predictions(
    pred_path: Path, hub_path: str, message: str, *, smoke: bool
) -> tuple[dict, str | None]:
    """Upload a completed run's predictions to the private Hub repo; return (pointer, error).

    Smoke and failed runs are not uploaded: their files stay on local disk only. Either way the
    result in git gets metadata (sha256, pointer), never the predictions themselves.
    """
    from lumen import hub

    local = {
        "sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "local_file": pred_path.name,
    }
    if smoke:
        return {**local, "stored": "local only (smoke or failed run; not uploaded)"}, None
    try:
        pointer = hub.upload(
            make_hub_api(),
            hub.predictions_repo(),
            pred_path,
            hub_path,
            message,
        )
    except Exception as e:  # recorded, never swallowed: the stage exits nonzero
        return {**local, "stored": "upload failed", "error": f"{type(e).__name__}: {e}"}, str(e)
    return {**pointer, "stored": "private Hugging Face dataset"}, None


def _baseline_plan(models: list[str], split: str, prompts: list[str], args) -> list[tuple]:
    from lumen.zeroshot.estimates import estimate_run
    from lumen.zeroshot.harness import read_predictions

    pins = load_pins()
    n_split = {
        "dev": len(read_ids(SPLITS_DIR / SPLIT_FILES["dev"])),
        "dev_select": pins.dev_select_size,
    }
    if split == "val":
        n_split["val"] = json.loads((RESULTS / "data" / "vizwiz_data.json").read_text())["summary"][
            "val"
        ]["n"]
    plan = []
    for m in models:
        for p in prompts:
            n = min(n_split[split], args.limit) if args.limit else n_split[split]
            pred = _pred_path(m, split, p, _run_kind(args), args.long_edge)
            done = len(read_predictions(pred)[1])
            plan.append(
                (m, p, n - done, estimate_run(m, split, p, max(n - done, 0), args.long_edge))
            )
    del pins
    return plan


def cmd_baseline(args: argparse.Namespace) -> int:
    from lumen.budget import BudgetExceeded

    pins = load_pins()
    models = sorted(pins.models) if args.model == ["all"] else args.model
    unknown = set(models) - set(pins.models)
    if unknown:
        raise SystemExit(f"unknown model(s) {sorted(unknown)}; pinned: {sorted(pins.models)}")
    if args.split == "val" and args.prompt != "dev-selected":
        raise SystemExit("val uses the prompt chosen on dev: pass --prompt dev-selected")
    if args.split in ("dev", "dev_select") and args.prompt == "dev-selected":
        raise SystemExit("--prompt dev-selected is for val")
    requested = args.long_edge
    args.long_edge = _resolve_long_edge(requested, models)
    if requested == "chosen" and args.long_edge is None:
        print(
            f"the chosen resolution is the default ({DEFAULT_LONG_EDGE} px): the default-"
            "resolution runs already are the matched baseline; nothing to run"
        )
        return 0

    selections: dict[str, dict] = {}
    if args.split == "val":
        if args.dry_run:
            prompts_by_model = {m: ["<dev-selected>"] for m in models}
        else:
            prompts_by_model = {}
            for m in models:
                chosen_prompt, selections[m] = _dev_selected_prompt(m)
                prompts_by_model[m] = [chosen_prompt]
    else:
        chosen = list(PROMPTS) if args.prompt == "both" else [args.prompt]
        prompts_by_model = {m: chosen for m in models}

    if args.dry_run:
        rows = []
        for m in models:
            prompts = [p if p != "<dev-selected>" else "plain" for p in prompts_by_model[m]]
            for model, p, n, est in _baseline_plan([m], args.split, prompts, args):
                label = "dev-selected" if args.split == "val" else p
                rows.append((model, label, n, est.hours))
        total = sum(r[3] for r in rows)
        print(f"{'model':<16} {'split':<5} {'prompt':<13} {'questions':>9} {'GPU h (prior)':>14}")
        for model, p, n, h in rows:
            print(f"{model:<16} {args.split:<5} {p:<13} {n:>9} {h:>14.2f}")
        print(f"{'total':<46} {total:>14.2f}")
        if args.gpu_hours_left is not None:
            verdict = "fits" if total + 1.0 <= args.gpu_hours_left else "DOES NOT FIT"
            print(f"with a 1.0 h reserve and {args.gpu_hours_left:.1f} h left: {verdict}")
        return 0

    if not (args.limit is not None or args.stand_in):
        # Check the private predictions repo before spending any GPU time.
        from lumen import hub

        try:
            hub.preflight(make_hub_api(), hub.predictions_repo())
        except hub.HubError as e:
            print(f"NOT STARTED (predictions storage): {e}")
            return EXIT_STORAGE

    for m in models:
        for p in prompts_by_model[m]:
            try:
                code = _baseline_one(
                    m, args.split, p, args, selections.get(m) if args.split == "val" else None
                )
            except BudgetExceeded as e:
                print(f"NOT STARTED (budget): {e}")
                return EXIT_BUDGET
            if code:
                return code
    return 0


# ----------------------------------------------------------------------- baseline_summary


def _latest_results(root: Path) -> list[Path]:
    """The newest result per (kind, model, split, prompt): name.json, name.1.json, ... -> last."""
    latest: dict[tuple, tuple[int, Path]] = {}
    for f in root.rglob("*.json"):
        parts = f.stem.split(".")
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        key = (f.parent, parts[0])
        if key not in latest or n > latest[key][0]:
            latest[key] = (n, f)
    return sorted(p for _, p in latest.values())


def _num(x: float | None) -> float:
    return float("nan") if x is None else x


def cmd_baseline_summary(args: argparse.Namespace) -> int:
    """One line per baseline result: what ran, how it scored, how fast, and what it cost."""
    root = RESULTS / BASELINE_DIR
    files = _latest_results(root) if root.exists() else []
    if not files:
        print(f"no baseline results under {root}")
        return 0
    head = (
        f"{'kind':<9}{'model':<16}{'split':<11}{'prompt':<9}{'status':<10}{'n':>5}"
        f"{'acc':>7}{'95% CI':>15}{'AP':>7}{'unans%':>8}{'s/q':>7}{'GPU h':>7}{'edge':>6}  stored"
    )
    print(head)
    for f in files:
        d = json.loads(f.read_text())
        rel = f.relative_to(root).parts
        kind = rel[0] if len(rel) > 2 else "real"
        m = d.get("metrics") or {}
        acc = m.get("accuracy", {})
        ci = acc.get("ci95", {})
        ans = m.get("answerability", {})
        run = d.get("run") or {}
        spq = run.get("sec_per_question")
        stored = (d.get("predictions") or {}).get("stored", "")
        ci_txt = f"[{ci['low']:.1f}, {ci['high']:.1f}]" if ci else ""
        print(
            f"{kind:<9}{f.parent.name:<16}"
            f"{d['split']:<11}{d['prompt']['prompt_key']:<9}{d['status']:<10}"
            f"{m.get('n', d.get('answered', 0)):>5}"
            f"{acc.get('overall', float('nan')):>7.2f}"
            f"{ci_txt:>15}"
            f"{ans.get('average_precision', float('nan')):>7.2f}"
            f"{100 * m.get('said_unanswerable_rate', float('nan')):>8.1f}"
            f"{spq if spq is not None else float('nan'):>7.3f}"
            f"{_num(d['gpu_hours']['actual_this_run']):>7.2f}"
            f"{d.get('image_longest_edge') or DEFAULT_LONG_EDGE:>6}  {stored}"
        )
    return 0


# ----------------------------------------------------------------------- rebuild_baseline


def fetch_from_hub(repo: str, path: str, revision: str) -> Path:
    """Tests replace this; on Kaggle it downloads from the private Hub dataset (HF_TOKEN)."""
    from lumen import hub

    return hub.fetch(repo, path, revision)


_SPLIT_ORDER = {"dev_select": 0, "dev": 1, "val": 2}  # val last: it records the dev-chosen prompt


def cmd_rebuild_baseline(args: argparse.Namespace) -> int:
    """Re-create baseline result files lost with a Kaggle session, from the private Hub predictions.

    Lists `baseline/<model>/*.predictions.jsonl` at one exact dataset commit and downloads each
    file at that commit. A file is refused unless its header matches the pinned model revision,
    the current prompt templates, and the current extractor, and it answers exactly its split's
    questions. Accepted files are scored with the official metric, as the original runs were.
    GPU hours, timing, and batch settings can't be recovered and are recorded as lost. Existing
    result files are never touched. Loading dev_select regenerates dev_select.txt (to the
    download folder), checked against any committed copy.
    """
    from lumen import hub
    from lumen.zeroshot.extract import EXTRACTOR_VERSION
    from lumen.zeroshot.harness import read_predictions, score_predictions
    from lumen.zeroshot.prompts import PROMPT_VERSION

    warn_if_gpu_attached("rebuild_baseline")
    pins = load_pins()
    model = args.model
    if model not in pins.models:
        raise SystemExit(f"unknown model {model!r}; pinned: {sorted(pins.models)}")
    pin = pins.models[model]
    api, repo = make_hub_api(), hub.predictions_repo()
    try:
        revision, paths = hub.list_files(api, repo, f"{BASELINE_DIR}/{model}/")
    except hub.HubError as e:
        print(f"NOT STARTED (predictions storage): {e}")
        return EXIT_STORAGE
    files = [p for p in paths if p.endswith(".predictions.jsonl")]
    print(f"{repo} @ {revision[:12]}: {len(files)} prediction file(s) for {model}")
    if not files:
        print("nothing to rebuild from: these runs must be rerun")
        return EXIT_REBUILD

    def order(path: str) -> tuple[int, str]:
        return (_SPLIT_ORDER.get(Path(path).name.split("-")[0], 9), path)

    run_dir = _run_dir(model, "")
    failed = 0
    for path in sorted(files, key=order):
        name = Path(path).name.removesuffix(".predictions.jsonl")
        if _result_files(run_dir, name):
            print(f"  {name}: a result already exists; left as is")
            continue
        local = fetch_from_hub(repo, path, revision)
        header, preds = read_predictions(local)
        problems = []
        if header is None:
            problems.append("no header line")
        else:
            m = header["model"]
            split, prompt_key = header["split"], header["prompt_key"]
            long_edge = m.get("image_longest_edge")
            if _run_name(split, prompt_key, long_edge) != name:
                problems.append(f"header says {_run_name(split, prompt_key, long_edge)!r}")
            if m.get("model_key") != model or m.get("revision") != pin.revision:
                problems.append(f"model {m.get('model_key')}@{m.get('revision')} is not the pin")
            if m.get("stand_in"):
                problems.append("a stand-in run")
            if (
                header.get("prompt_template") != PROMPTS.get(prompt_key)
                or header.get("prompt_version") != PROMPT_VERSION
                or header.get("extractor_version") != EXTRACTOR_VERSION
            ):
                problems.append("prompt or extractor differs from the current code")
        if not problems:
            records, _ = _load_split(split, pins, limit=None, images=False)
            want = {r["question_id"] for r in records}
            missing, extra = want - set(preds), set(preds) - want
            if missing or extra:
                problems.append(f"incomplete: {len(missing)} questions missing, {len(extra)} extra")
        selection = None
        if not problems and split == "val":
            try:
                selection = _dev_selected_prompt(model)[1]
            except SystemExit as e:  # e.g. a dev_select file was refused above
                problems.append(f"no dev prompt choice to check it against ({e})")
            else:
                if selection["chosen"] != prompt_key:
                    problems.append(
                        f"val used {prompt_key!r} but dev chose {selection['chosen']!r}"
                    )
        if problems:
            failed += 1
            print(f"  {name}: REFUSED ({'; '.join(problems)})")
            continue
        payload = {
            "stage": "baseline",
            "status": "completed",
            "error": None,
            "smoke": False,
            "rebuilt": {
                "from": "predictions in the private Hugging Face dataset",
                "reason": "the original result files were lost with the Kaggle session's disk",
                "lost": "GPU hours, timing, batch size, prefetch, image-token counts, model "
                "description beyond the predictions header",
            },
            "model": {**header["model"], "generation": header["generation"]},
            "split": split,
            "prompt": header,
            "prompt_selection": selection,
            "image_longest_edge": long_edge,
            "image_tokens": None,
            "batch_size": None,
            "prefetch_workers": None,
            "prefetch_ab": None,
            "gpu_hours": {"prior_estimate": None, "calibration": None, "actual_this_run": None},
            "run": None,
            "metrics": score_predictions(records, preds),
            "predictions": {
                **hub.pointer(repo, path, revision, local),
                "stored": "private Hugging Face dataset",
            },
        }
        result = write_result(run_dir / f"{name}.json", payload)
        collect(result, str(Path(BASELINE_DIR) / model))
        acc = payload["metrics"]["accuracy"]
        print(
            f"  {name}: rebuilt, n={payload['metrics']['n']}, accuracy {acc['overall']:.2f} "
            f"[{acc['ci95']['low']:.2f}, {acc['ci95']['high']:.2f}], "
            f"AP {payload['metrics']['answerability']['average_precision']:.2f}"
        )
    return EXIT_REBUILD if failed else 0


# ---------------------------------------------------------------------- resolution_choice

PROBE_MODEL = "smolvlm2-500m"  # the model Lumen fine-tunes
LONG_EDGE_MODELS = ("smolvlm2-500m", "smolvlm2-256m")  # same processor: 512 px tiles, 2,048 cap
PREP_DIR = "stage3_prep"
# The probe runs on all of dev (2,052 questions), not dev_select: a paired CI with a 2-point margin
# needs the sample size (approved 2026-10-03). The 2,048 px arm is the default-resolution dev run.
PROBE_SPLIT = "dev"


def _long_edge_arg(value: str) -> int | str:
    if value == "chosen":
        return value
    n = int(value)
    if n not in PORTRAIT_TILES:
        raise argparse.ArgumentTypeError(f"must be one of {sorted(PORTRAIT_TILES)} or 'chosen'")
    return n


RESOLUTION_OVERRIDE = "resolution_override.yaml"  # under configs/


def _chosen_long_edge() -> int:
    """The rule's choice from the newest resolution_choice result, unless a committed override
    (configs/resolution_override.yaml, a documented deviation) replaces it. The override must name
    the rule's choice it replaces, so it can't silently outlive a rerun that changes the rule's
    answer."""
    import yaml

    files = _result_files(RESULTS / PREP_DIR, "resolution_choice")
    if not files:
        raise SystemExit(
            "no resolution choice yet: run the dev probe at every long edge, then the "
            "resolution_choice stage (and commit its result)"
        )
    rule = int(json.loads(files[-1].read_text())["chosen"]["long_edge"])
    override_path = REPO_ROOT / "configs" / RESOLUTION_OVERRIDE
    if not override_path.exists():
        return rule
    o = yaml.safe_load(override_path.read_text())
    if int(o["rule_choice"]) != rule:
        raise SystemExit(
            f"{override_path.name} overrides a rule choice of {o['rule_choice']} px, but the "
            f"newest resolution_choice result chose {rule} px: review and update the override"
        )
    long_edge = int(o["long_edge"])
    if long_edge not in PORTRAIT_TILES or not str(o.get("reason", "")).strip():
        raise SystemExit(f"{override_path.name}: needs a valid long_edge and a reason")
    print(f"resolution: {long_edge} px, overriding the rule's {rule} px ({override_path.name})")
    return long_edge


def _resolve_long_edge(value: int | str | None, models: list[str]) -> int | None:
    """The cap to run with; None = the processor's default (2,048), so default-resolution runs
    keep their existing file names and predictions headers."""
    if value is None:
        return None
    if value == "chosen":
        value = _chosen_long_edge()
    other = sorted(set(models) - set(LONG_EDGE_MODELS))
    if other:
        raise SystemExit(f"--long-edge applies to {list(LONG_EDGE_MODELS)} only, not {other}")
    return None if value == DEFAULT_LONG_EDGE else int(value)


def download_predictions(pointer: dict) -> Path:
    """Tests replace this; on Kaggle it fetches from the private Hub repo, checking the sha256."""
    from lumen import hub

    return hub.download(pointer)


def _probe_predictions(doc: dict, prompt: str, long_edge: int | None) -> dict[str, dict]:
    """A probe run's predictions: the local file if its sha256 matches the result, else the Hub."""
    pointer = doc["predictions"]
    local = _pred_path(PROBE_MODEL, PROBE_SPLIT, prompt, "", long_edge)
    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == pointer["sha256"]:
        path = local
    elif pointer.get("stored") == "private Hugging Face dataset":
        path = download_predictions(pointer)
    else:
        raise SystemExit(f"{local.name}: not on disk and not stored on the Hub ({pointer})")
    from lumen.zeroshot.harness import read_predictions

    header, preds = read_predictions(path)
    if header is None or header["model"].get("image_longest_edge") != long_edge:
        raise SystemExit(f"{path.name}: predictions header doesn't match long edge {long_edge}")
    return preds


def cmd_resolution_choice(args: argparse.Namespace) -> int:
    """Pick the image resolution for fine-tuning from the dev probe (dev data only).

    Each long edge's dev run (all 2,052 dev questions, with the prompt chosen on dev_select) is
    re-scored from its predictions so
    every arm is compared on the same questions. The rule (lumen.zeroshot.resolution.RULE, fixed in
    advance) uses paired bootstrap CIs on the accuracy difference from the best arm; differences
    from the default 2,048 px and all AP differences are reported, not decisive.
    """
    import numpy as np

    from lumen.eval.answerability import evaluate_answerability
    from lumen.eval.bootstrap import bootstrap_ci
    from lumen.eval.vizwiz_metric import evaluate_accuracy
    from lumen.zeroshot.extract import UNANSWERABLE
    from lumen.zeroshot.harness import question_ids_sha256
    from lumen.zeroshot.resolution import RULE, abstention_breakdown, choose_long_edge

    warn_if_gpu_attached("resolution_choice")
    prompt, selection = _dev_selected_prompt(PROBE_MODEL)
    run_dir = _run_dir(PROBE_MODEL, "")
    edges = sorted(PORTRAIT_TILES, reverse=True)
    docs: dict[int, tuple[Path, dict]] = {}
    for le in edges:
        files = _result_files(run_dir, _run_name(PROBE_SPLIT, prompt, _resolve(le)))
        if files:
            docs[le] = (files[-1], json.loads(files[-1].read_text()))
    missing = [le for le in edges if le not in docs]
    if missing:
        raise SystemExit(
            f"no {PROBE_SPLIT} probe run for long edge {missing}: run `baseline --model "
            f"{PROBE_MODEL} --split {PROBE_SPLIT} --prompt {prompt} --long-edge N` for each "
            f"(2048 = the default-resolution run, without --long-edge)"
        )
    for f, d in docs.values():
        if d["status"] != "completed":
            raise SystemExit(f"{f.name}: status {d['status']}; rerun it to completion first")

    records, _ = _load_split(PROBE_SPLIT, load_pins(), limit=None, images=False)
    ids = [r["question_id"] for r in records]
    for f, d in docs.values():
        if d["metrics"]["question_ids_sha256"] != question_ids_sha256(ids):
            raise SystemExit(f"{f.name}: answered a different dev than the committed one")
    y = np.array([r["answerable"] for r in records])

    def ap(labels: np.ndarray, scores: np.ndarray) -> float:
        return evaluate_answerability(labels, scores).average_precision

    scored, breakdowns = {}, {}
    for le in edges:
        preds = _probe_predictions(docs[le][1], prompt, _resolve(le))
        acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
        a = np.array([100 * acc.per_question[q] for q in ids])
        s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
        said = np.array([preds[q]["answer"].strip().lower() == UNANSWERABLE for q in ids])
        scored[le] = (acc, a, s)
        breakdowns[le] = abstention_breakdown(y, said, a)

    choice = choose_long_edge({le: a for le, (_, a, _) in scored.items()})
    best = choice["best"]
    _, a0, s0 = scored[DEFAULT_LONG_EDGE]
    _, _, s_best = scored[best]
    arms = []
    for le in edges:
        f, d = docs[le]
        acc, a, s = scored[le]
        arm = {
            "long_edge": le,
            "result": str(f.relative_to(REPO_ROOT)),
            "accuracy": acc.overall,
            "accuracy_official": acc.overall_official,
            "average_precision": ap(y, s),
            "portrait_tiles": PORTRAIT_TILES[le],
            "image_tokens": d.get("image_tokens"),
            "sec_per_question": (d.get("run") or {}).get("sec_per_question"),
            "answerable_breakdown": breakdowns[le],  # reported, not decisive (see the rule)
        }
        if le != best:
            arm["vs_best"] = {
                **choice["vs_best"][le],
                "ap_diff": ap(y, s) - ap(y, s_best),
                "ap_diff_ci95": bootstrap_ci(
                    len(ids), lambda i, s=s: ap(y[i], s[i]) - ap(y[i], s_best[i])
                ),
            }
        if le != DEFAULT_LONG_EDGE:
            arm["vs_default"] = {
                "accuracy_diff": float(a.mean() - a0.mean()),
                "accuracy_diff_ci95": bootstrap_ci(
                    len(ids), lambda i, a=a: float(a[i].mean() - a0[i].mean())
                ),
                "ap_diff": ap(y, s) - ap(y, s0),
                "ap_diff_ci95": bootstrap_ci(
                    len(ids), lambda i, s=s: ap(y[i], s[i]) - ap(y[i], s0[i])
                ),
            }
        arms.append(arm)

    chosen = next(r for r in arms if r["long_edge"] == choice["chosen"])
    payload = {
        "stage": "resolution_choice",
        "model": PROBE_MODEL,
        "split": PROBE_SPLIT,
        "n": len(ids),
        "question_ids_sha256": question_ids_sha256(ids),
        "prompt": prompt,
        "prompt_selection": selection,
        "rule": RULE,
        "margin_points": choice["margin_points"],
        "best": best,
        "arms": arms,
        "chosen": {"long_edge": chosen["long_edge"], "accuracy": chosen["accuracy_official"]},
    }
    result = write_result(RESULTS / PREP_DIR / "resolution_choice.json", payload)
    collect(result, PREP_DIR)
    print(
        f"{'long edge':>9}{'tiles':>7}{'acc':>8}{'vs best [95% CI]':>24}{'ok':>4}"
        f"{'AP':>8}{'AP vs best':>12}{'s/q':>8}"
    )
    for r in arms:
        v = r.get("vs_best")
        if v:
            ci = v["accuracy_diff_ci95"]
            diff = f"{v['accuracy_diff']:+.2f} [{ci['low']:+.1f}, {ci['high']:+.1f}]"
            ok, ap_diff = ("yes" if v["within_margin"] else "no"), f"{v['ap_diff']:+.2f}"
        else:
            diff, ok, ap_diff = "(best)", "yes", ""
        spq = r["sec_per_question"]
        print(
            f"{r['long_edge']:>9}{r['portrait_tiles']:>7}{r['accuracy_official']:>8.2f}"
            f"{diff:>24}{ok:>4}{r['average_precision']:>8.2f}{ap_diff:>12}"
            f"{spq if spq is not None else float('nan'):>8.3f}"
        )
    print("answerable questions only: says-unanswerable %, credit from 'unanswerable' / from real")
    print("answers, accuracy when it answers (reported, not decisive):")
    for r in arms:
        b = r["answerable_breakdown"]
        print(
            f"{r['long_edge']:>9}  {b['says_unanswerable']:5.1f}%  "
            f"{b['credit_from_unanswerable']:6.2f}  {b['credit_from_answers']:6.2f}  "
            f"{b['accuracy_when_answering']:6.2f}  (n={b['n']})"
        )
    print(
        f"best: {best} px; chosen: {chosen['long_edge']} px (smallest long edge whose CI lower "
        f"bound vs best is >= -{choice['margin_points']:g} point)"
    )
    return 0


def _resolve(long_edge: int) -> int | None:
    return None if long_edge == DEFAULT_LONG_EDGE else long_edge


# ---------------------------------------------------------------------------- train_speed


def _train_fit_records(pins, *, limit: int | None) -> tuple[list[Record], Path]:
    """train_fit (official train minus the committed dev IDs), sorted, and the train image zip."""
    ann = fetch_pinned(pins.files["annotations"], DATA_DIR).path
    ids = set(read_ids(SPLITS_DIR / SPLIT_FILES["train_fit"]))
    records = [
        r
        for r in parse_annotations(_annotation_json(ann, "train"), "train")
        if r["question_id"] in ids
    ]
    if len(records) != len(ids):
        raise SystemExit(f"train_fit: {len(ids) - len(records)} committed IDs not in train.json")
    dev = set(read_ids(SPLITS_DIR / SPLIT_FILES["dev"]))
    if dev & ids:
        raise SystemExit("train_fit and dev overlap: the committed splits are broken")
    records.sort(key=lambda r: r["question_id"])
    if limit is not None:
        records = records[:limit]
    return records, fetch_pinned(pins.files["train_images"], DATA_DIR).path


def _train_model(args: argparse.Namespace, long_edge: int | None):
    """(model, processor, device) for training: the pinned SmolVLM2-500M in fp16 on the T4, or the
    tiny random-weight stand-in (fp32) for pipeline checks."""
    import torch

    from lumen.zeroshot.backends import tiny_stand_in

    pin = load_pins().models[PROBE_MODEL]
    device = args.device or ("cuda" if torch.cuda.is_available() else None)
    if device is None:
        raise SystemExit("no CUDA GPU: attach a T4, or pass --device cpu (only for --stand-in)")
    if args.stand_in:
        b = tiny_stand_in(pin, device=device, image_longest_edge=long_edge)
        return b.model, b.processor, device
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(pin.repo, revision=pin.revision)
    dtype = torch.float16 if device == "cuda" else torch.float32  # T4: no bf16
    model = AutoModelForImageTextToText.from_pretrained(
        pin.repo, revision=pin.revision, dtype=dtype
    )
    return model.to(device), processor, device


def cmd_train_speed(args: argparse.Namespace) -> int:
    """A short LoRA run on train_fit to measure training cost before Stage 3 is budgeted.

    Same model, data, prompt (the one chosen on dev_select), targets, and image resolution as the
    real training runs; LoRA settings are the TrainConfig defaults (chosen on dev later). Records
    s/step (median and p90 after warm-up), peak GPU memory, tokens/s, and projected hours per
    epoch. Starts fresh each time; the adapter is not published.
    """
    from lumen.budget import BudgetExceeded, calibrated_estimate, check_budget
    from lumen.train.data import TrainCollator, TrainDataset, examples_from_targets
    from lumen.train.estimates import estimate_train
    from lumen.train.loop import StopTraining, TrainConfig, attach_lora, parameter_counts, train
    from lumen.train.targets import TARGET_VERSION, build_targets

    long_edge = _resolve_long_edge(args.long_edge, [PROBE_MODEL])
    prompt, selection = _dev_selected_prompt(PROBE_MODEL)

    cfg = TrainConfig(
        prompt_key=prompt,
        image_longest_edge=long_edge,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        schedule_steps=max(args.steps, 1000),
        checkpoint_every=args.steps,  # only the final state; this run is a measurement
        fp16=not args.stand_in,
        dataloader_workers=0 if args.stand_in else 2,
    )
    name = "train_speed" + ("" if long_edge is None else f"-le{long_edge}")
    prior = estimate_train(f"{name}:{PROBE_MODEL}", args.steps, cfg.batch_size, long_edge)
    res = f"{long_edge or DEFAULT_LONG_EDGE} px"
    print(
        f"\n== train_speed | {PROBE_MODEL} | {res} | prompt={prompt} | {args.steps} steps x "
        f"{cfg.batch_size} | prior {prior.hours:.2f} GPU h ({prior.sec_per_item:.2f} s/step)"
    )
    if args.dry_run:
        return 0
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    if not args.stand_in:
        try:
            check_budget(prior, left)
        except BudgetExceeded as e:
            print(f"NOT STARTED (budget): {e}")
            return EXIT_BUDGET

    records, zip_path = _train_fit_records(load_pins(), limit=args.limit)
    pairs, target_stats = build_targets(records, include_unanswerable=cfg.include_unanswerable)
    model, processor, device = make_train_model(args, cfg.image_longest_edge)
    model = attach_lora(model, cfg)
    dataset = TrainDataset(examples_from_targets(pairs), ZipImages(zip_path))
    collate = TrainCollator(processor, cfg.prompt_key, image_longest_edge=cfg.image_longest_edge)
    ckpt_dir = CHECKPOINT_DIR / "train_speed" / name
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # a measurement: always from step 0
    calibration: dict = {}

    def on_step(step: int, report) -> None:
        if calibration or step < args.calibration_steps or step >= args.steps:
            return
        seconds = sum(report.step_seconds)
        cal = calibrated_estimate(prior, step, seconds)
        rest = calibrated_estimate(
            estimate_train(prior.stage, args.steps - step, cfg.batch_size, cfg.image_longest_edge),
            step,
            seconds,
        )
        calibration.update(estimate=cal.as_dict(), measured_steps=step, seconds=round(seconds, 2))
        print(f"   calibrated: {seconds / step:.2f} s/step -> {cal.hours:.2f} GPU h total")
        if not args.stand_in and left is not None:
            try:
                check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)
            except BudgetExceeded as e:
                print(f"STOPPING (budget): {e}")
                raise StopTraining("budget") from e

    t0 = time.monotonic()
    report = train(
        model,
        dataset,
        collate,
        cfg,
        max_steps=args.steps,
        checkpoint_dir=ckpt_dir,
        device=device,
        on_step=on_step,
    )
    hours = (time.monotonic() - t0) / 3600
    summary = report.summary()
    spq = summary["sec_per_step_median"]
    steps_per_epoch = math.ceil(len(dataset) / cfg.batch_size)
    payload = {
        "stage": "train_speed",
        "status": report.status,
        "smoke": bool(args.stand_in or args.limit is not None),
        "model": {
            "model_key": PROBE_MODEL,
            "stand_in": bool(args.stand_in),
            "parameters": parameter_counts(model),
            "device": device,
        },
        "config": cfg.as_dict(),
        "prompt_selection": selection,
        "targets": {"version": TARGET_VERSION, **target_stats.as_dict()},
        "train_fit_examples": len(dataset),
        "train": summary,
        "projection": {
            "steps_per_epoch": steps_per_epoch,
            "hours_per_epoch": None if spq is None else round(steps_per_epoch * spq / 3600, 3),
            "basis": "median s/step after warm-up x steps per epoch; excludes model load and eval",
        },
        "gpu_hours": {
            "prior_estimate": prior.as_dict(),
            "calibration": calibration or None,
            "actual_this_run": round(hours, 4),
        },
        "adapter": "not published (speed test); checkpoints stay on local disk",
    }
    sub = "smoke" if payload["smoke"] else ""
    result = write_result(RESULTS / PREP_DIR / sub / f"{name}.json", payload)
    collect(result, str(Path(PREP_DIR) / sub))
    print(
        f"   {report.status}: {report.final_step} steps, median {spq or float('nan'):.2f} s/step, "
        f"peak {report.peak_memory_gib or float('nan'):.1f} GiB, "
        f"~{payload['projection']['hours_per_epoch']} h/epoch, loss "
        f"{summary['loss_first']} -> {summary['loss_last']}"
    )
    if report.status == "stopped_non_finite":
        return EXIT_NON_FINITE
    if report.status.startswith("stopped:"):
        return EXIT_BUDGET
    return 0


# Tests replace this to run the stage with a tiny model.
make_train_model = _train_model


# ---------------------------------------------------------------------------------- train

FINETUNE_DIR = "finetune"
CHECKPOINT_META = "lumen_checkpoint.json"  # written into a checkpoint before it is uploaded
ADAPTER_META = "lumen_adapter.json"  # written into each published epoch adapter
_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# Headroom over the speed test's median s/step when it replaces the prior (p90 was ~10% above).
MEASURED_STEP_HEADROOM = 1.1


def _run_name_arg(value: str) -> str:
    if not _RUN_NAME.match(value) or value in ("smoke", "choice", "comparison"):
        raise argparse.ArgumentTypeError(
            "a run name: lowercase letters, digits, '.', '_', '-' (e.g. lr2e-4-s1)"
        )
    return value


def _finetune_kind(args: argparse.Namespace) -> str:
    """Stand-in and limited runs are smoke runs: kept apart, never uploaded, never chosen."""
    return "smoke" if args.stand_in or args.limit is not None else ""


def _finetune_root(kind: str) -> Path:
    return RESULTS / FINETUNE_DIR / kind


def _measured_sec_per_step(long_edge: int | None, batch_size: int) -> tuple[float, str] | None:
    """The speed test's median s/step at this resolution and batch, if it has run."""
    name = "train_speed" + ("" if long_edge is None else f"-le{long_edge}")
    for f in reversed(_result_files(RESULTS / PREP_DIR, name)):
        d = json.loads(f.read_text())
        spq = (d.get("train") or {}).get("sec_per_step_median")
        if (
            d["status"] == "completed"
            and not d["smoke"]
            and d["config"]["effective_batch_size"] == batch_size
            and spq
        ):
            return float(spq), str(f.relative_to(REPO_ROOT))
    return None


def _train_estimate(run: str, steps: int, cfg) -> tuple[object, str]:
    from lumen.budget import prior_estimate
    from lumen.train.estimates import TRAIN_OVERHEAD_S, estimate_train

    measured = _measured_sec_per_step(cfg.image_longest_edge, cfg.batch_size)
    if measured is None:
        est = estimate_train(f"train:{run}", steps, cfg.batch_size, cfg.image_longest_edge)
        return est, "prior (lumen.train.estimates; no speed test at this resolution and batch)"
    sps, source = measured
    est = prior_estimate(
        f"train:{run}", steps, sps * MEASURED_STEP_HEADROOM, overhead_s=TRAIN_OVERHEAD_S
    )
    return est, f"speed test {source} (median x {MEASURED_STEP_HEADROOM})"


def _train_spec(cfg, epochs: int, steps_per_epoch: int, records: list[Record]) -> dict:
    """Everything that determines what a run learns. A run name is bound to one spec: resuming or
    re-running a name with a different spec is refused."""
    from lumen.train.selection import NOT_IN_SPEC
    from lumen.train.targets import TARGET_VERSION
    from lumen.zeroshot.harness import question_ids_sha256
    from lumen.zeroshot.prompts import PROMPT_VERSION

    pin = load_pins().models[PROBE_MODEL]
    return {
        "model": {"model_key": PROBE_MODEL, "repo": pin.repo, "revision": pin.revision},
        "config": {k: v for k, v in cfg.as_dict().items() if k not in NOT_IN_SPEC},
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": epochs * steps_per_epoch,
        "train_fit_ids_sha256": question_ids_sha256([r["question_id"] for r in records]),
        "target_version": TARGET_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


def fetch_model_file(repo: str, path: str, revision: str) -> Path:
    """Tests replace this; on Kaggle it downloads one file of the private adapter repo."""
    from lumen import hub

    return hub.fetch(repo, path, revision, repo_type="model")


def _step_of(checkpoint: Path) -> int:
    return int(checkpoint.name.split("-")[1])


def _hub_run_files(api, run: str) -> tuple[str, list[str]]:
    from lumen import hub

    return hub.list_files(api, hub.adapter_repo(), f"runs/{run}/", "model")


def _restore_checkpoint_from_hub(api, run: str, spec_sha: str, ckpt_root: Path) -> Path | None:
    """A new Kaggle session starts with an empty disk: fetch the run's last uploaded checkpoint."""
    from lumen import hub

    revision, paths = _hub_run_files(api, run)
    meta_path = f"runs/{run}/checkpoint/{CHECKPOINT_META}"
    if meta_path not in paths:
        return None
    meta = json.loads(fetch_model_file(hub.adapter_repo(), meta_path, revision).read_text())
    if meta["spec_sha256"] != spec_sha:
        raise SystemExit(
            f"run {run!r} on the Hub was trained with a different spec: pick a new run name"
        )
    dest = ckpt_root / f"step-{meta['step']:07d}"
    pointer = {
        "repo": hub.adapter_repo(),
        "repo_type": "model",
        "path": f"runs/{run}/checkpoint",
        "revision": revision,
        "files": meta["files"],
    }
    hub.download_folder(pointer, dest, fetch_file=fetch_model_file)
    (dest / CHECKPOINT_META).write_text(json.dumps(meta, indent=1))
    print(f"   resumed from the Hub: {run} step {meta['step']} (@{revision[:12]})")
    return dest


def _with_meta(folder: Path, meta_name: str, meta: dict) -> dict:
    """Write `meta` (plus the sha256 of every other file) into `folder`; return it."""
    from lumen import hub

    (folder / meta_name).unlink(missing_ok=True)
    meta = {**meta, "files": hub.folder_digest(folder)}
    (folder / meta_name).write_text(json.dumps(meta, indent=1))
    return meta


def _publish(api, local: Path, path_in_repo: str, message: str, smoke: bool) -> dict:
    from lumen import hub

    if smoke:
        return {
            "stored": "local only (smoke run; not uploaded)",
            "local": str(local),
            "files": hub.folder_digest(local),
        }
    pointer = hub.upload_folder(api, hub.adapter_repo(), local, path_in_repo, message)
    return {**pointer, "stored": "private Hugging Face model repo"}


def cmd_train(args: argparse.Namespace) -> int:
    """LoRA fine-tuning of SmolVLM2-500M on train_fit (Stage 3), resumable across Kaggle sessions.

    One run = one spec (config, epochs, data, model pin) under one name. The LR schedule spans all
    `--epochs`; `--stop-after-epoch` trains only the first k of them (seed and ablation replicates
    of a config chosen at an earlier epoch). At each epoch's end the adapter is published to the
    private adapter repo (`runs/<run>/epoch-<k>/`), and every `--checkpoint-every` steps the full
    resumable state goes to `runs/<run>/checkpoint/`, so a new session continues where the last
    stopped. Each session writes its own result (`results/finetune/<run>/train.json`, .1, ...).
    """
    from lumen import hub
    from lumen.budget import BudgetExceeded, calibrated_estimate, check_budget, prior_estimate
    from lumen.train.data import TrainCollator, TrainDataset, examples_from_targets
    from lumen.train.loop import (
        StopTraining,
        TrainConfig,
        attach_lora,
        latest_checkpoint,
        parameter_counts,
        train,
    )
    from lumen.train.selection import spec_sha256
    from lumen.train.targets import TARGET_VERSION, build_targets

    run, kind = args.run, _finetune_kind(args)
    smoke = kind != ""
    long_edge = _resolve_long_edge(args.long_edge, [PROBE_MODEL])
    prompt, selection = _dev_selected_prompt(PROBE_MODEL)
    records, zip_path = _train_fit_records(load_pins(), limit=args.limit)
    include_unans = not args.no_unanswerable
    pairs, target_stats = build_targets(records, include_unanswerable=include_unans)
    defaults = TrainConfig()
    micro, accum = args.micro_batch, args.grad_accum
    steps_per_epoch = math.ceil(len(pairs) / (micro * accum))
    cfg = TrainConfig(
        prompt_key=prompt,
        image_longest_edge=long_edge,
        include_unanswerable=include_unans,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        schedule_steps=args.epochs * steps_per_epoch,
        micro_batch=micro,
        grad_accum=accum,
        fp16=not args.stand_in,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed if args.seed is not None else defaults.seed,
        dataloader_workers=0 if args.stand_in else 2,
    )
    stop_epoch = args.stop_after_epoch or args.epochs
    if not 1 <= stop_epoch <= args.epochs:
        raise SystemExit(f"--stop-after-epoch must be between 1 and --epochs ({args.epochs})")
    spec = _train_spec(cfg, args.epochs, steps_per_epoch, records)
    spec_sha = spec_sha256(spec)
    target_step = stop_epoch * steps_per_epoch

    res_dir = _finetune_root(kind) / run
    for f in _result_files(res_dir, "train"):
        if json.loads(f.read_text())["spec_sha256"] != spec_sha:
            raise SystemExit(
                f"run {run!r} already has results with a different spec ({f.name}): a run name "
                "is bound to one config; pick a new name"
            )
    ckpt_root = CHECKPOINT_DIR / "train" / kind / run
    spec_file = ckpt_root / "run.json"
    if spec_file.exists() and json.loads(spec_file.read_text())["spec_sha256"] != spec_sha:
        raise SystemExit(f"{ckpt_root} holds run {run!r} with a different spec; pick a new name")

    api = None
    if not smoke and not args.dry_run:
        try:
            api = make_hub_api()
            hub.preflight(api, hub.adapter_repo(), "model")
        except hub.HubError as e:
            print(f"NOT STARTED (adapter storage): {e}")
            return EXIT_STORAGE
    local = latest_checkpoint(ckpt_root)
    if local is None and api is not None:
        local = _restore_checkpoint_from_hub(api, run, spec_sha, ckpt_root)
    start = _step_of(local) if local is not None else 0
    remaining = max(0, target_step - start)
    prior, prior_source = _train_estimate(run, remaining, cfg)
    res = f"{long_edge or DEFAULT_LONG_EDGE} px"
    print(
        f"\n== train | {run} | {res} | prompt={prompt} | lr {cfg.lr:g} | r {cfg.lora_r} | seed "
        f"{cfg.seed} | unanswerable {'in' if include_unans else 'OUT (ablation)'} | "
        f"{len(pairs)} examples | {steps_per_epoch} steps/epoch x {args.epochs} epochs"
        + (f" (stopping after epoch {stop_epoch})" if stop_epoch < args.epochs else "")
        + f"\n   step {start} -> {target_step}: {prior.hours:.2f} GPU h ({prior_source})"
    )
    if args.dry_run:
        return 0
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    if not args.stand_in and remaining:
        try:
            check_budget(prior, left)
        except BudgetExceeded as e:
            print(f"NOT STARTED (budget): {e}")
            return EXIT_BUDGET

    ckpt_root.mkdir(parents=True, exist_ok=True)
    spec_file.write_text(json.dumps({"run": run, "spec_sha256": spec_sha, "spec": spec}, indent=1))
    published_epochs: set[int] = set()
    if api is not None:
        _, paths = _hub_run_files(api, run)
        published_epochs = {
            int(p.split("/")[2].removeprefix("epoch-"))
            for p in paths
            if p.endswith(f"/{ADAPTER_META}") and p.split("/")[2].startswith("epoch-")
        }
    else:
        published_epochs = {
            int(d.name.removeprefix("epoch-"))
            for d in ckpt_root.glob("epoch-*")
            if (d / ADAPTER_META).exists()
        }
    behind = [e for e in range(1, stop_epoch + 1) if e * steps_per_epoch < start]
    lost = [e for e in behind if e not in published_epochs]
    if lost:
        raise SystemExit(
            f"run {run!r}: epoch adapter(s) {lost} were never published and the run is past them; "
            "they can't be recovered from this checkpoint. Start a new run name."
        )
    if not remaining and all(e in published_epochs for e in range(1, stop_epoch + 1)):
        print(f"   {run}: trained to step {target_step}, every epoch published; nothing to do")
        return 0

    model, processor, device = make_train_model(args, cfg.image_longest_edge)
    model = attach_lora(model, cfg)
    dataset = TrainDataset(examples_from_targets(pairs), ZipImages(zip_path))
    collate = TrainCollator(processor, cfg.prompt_key, image_longest_edge=cfg.image_longest_edge)

    calibration: dict = {}
    synced: dict = {"path": str(local) if local is not None else None, "pointer": None}
    upload_errors: list[str] = []
    t0 = time.monotonic()
    session_limit_s = args.session_hours * 3600

    def sync_checkpoint(path_str: str | None) -> None:
        if smoke or path_str is None or path_str == synced["path"]:
            return
        path = Path(path_str)
        _with_meta(
            path,
            CHECKPOINT_META,
            {"run": run, "step": _step_of(path), "spec_sha256": spec_sha},
        )
        try:
            step = _step_of(path)
            synced["pointer"] = _publish(
                api, path, f"runs/{run}/checkpoint", f"{run}: checkpoint step {step}", False
            )
            synced["path"] = path_str
            print(f"   checkpoint step {step} -> Hub @{synced['pointer']['revision'][:12]}")
        except Exception as e:  # keep training; the next checkpoint retries; recorded
            upload_errors.append(f"step {_step_of(path)}: {type(e).__name__}: {e}")
            print(f"   WARNING: checkpoint upload failed ({e}); retrying at the next checkpoint")

    def on_step(step: int, report) -> None:
        sync_checkpoint(report.last_good_checkpoint)
        if not calibration and report.steps_done >= args.calibration_steps and step < target_step:
            seconds = sum(report.step_seconds)
            cal = calibrated_estimate(prior, report.steps_done, seconds)
            rest = calibrated_estimate(
                prior_estimate(prior.stage, target_step - step, prior.sec_per_item, 0.0),
                report.steps_done,
                seconds,
            )
            calibration.update(
                estimate=cal.as_dict(), measured_steps=report.steps_done, seconds=round(seconds, 2)
            )
            rate = seconds / report.steps_done
            print(f"   calibrated: {rate:.2f} s/step -> {cal.hours:.2f} GPU h for this session")
            if not args.stand_in and left is not None:
                try:
                    check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)
                except BudgetExceeded as e:
                    print(f"STOPPING (budget): {e}")
                    raise StopTraining("budget") from e
        if time.monotonic() - SESSION_START > session_limit_s and step < target_step:
            print(f"STOPPING (session limit {args.session_hours:g} h): the next session resumes")
            raise StopTraining("session limit")

    status, segments, epochs_out = "completed", [], []
    for epoch in range(1, stop_epoch + 1):
        end = epoch * steps_per_epoch
        if end <= start and epoch in published_epochs:
            continue
        report = train(
            model,
            dataset,
            collate,
            cfg,
            max_steps=end,
            checkpoint_dir=ckpt_root,
            device=device,
            on_step=on_step,
        )
        if report.steps_done:
            segments.append({"epoch": epoch, **report.summary()})
        if report.status != "completed":
            status = report.status
            sync_checkpoint(report.last_good_checkpoint)
            break
        last = latest_checkpoint(ckpt_root)
        if last is None or _step_of(last) != end:
            raise RuntimeError(f"epoch {epoch}: no checkpoint at step {end}")
        sync_checkpoint(str(last))
        epoch_dir = ckpt_root / f"epoch-{epoch}"
        shutil.rmtree(epoch_dir, ignore_errors=True)
        shutil.copytree(last / "adapter", epoch_dir)
        meta = _with_meta(
            epoch_dir,
            ADAPTER_META,
            {"run": run, "epoch": epoch, "step": end, "spec_sha256": spec_sha, "spec": spec},
        )
        try:
            pointer = _publish(
                api, epoch_dir, f"runs/{run}/epoch-{epoch}", f"{run}: epoch {epoch} adapter", smoke
            )
        except Exception as e:
            status = "stopped:adapter upload failed"
            upload_errors.append(f"epoch {epoch}: {type(e).__name__}: {e}")
            print(f"STORAGE FAILED: epoch {epoch} adapter: {e}; the rerun publishes it")
            break
        published_epochs.add(epoch)
        epochs_out.append(
            {"epoch": epoch, "step": end, "adapter": pointer, "adapter_files": meta["files"]}
        )
        where = pointer.get("revision", "local")[:12]
        print(f"   epoch {epoch} (step {end}) adapter published @{where}")

    hours = (time.monotonic() - t0) / 3600
    final = latest_checkpoint(ckpt_root)
    final_step = _step_of(final) if final is not None else 0
    steps_s = [s for seg in segments for s in [seg["sec_per_step_median"]] if s]
    payload = {
        "stage": "train",
        "status": status,
        "smoke": smoke,
        "run": run,
        "spec": spec,
        "spec_sha256": spec_sha,
        "config": cfg.as_dict(),
        "stop_after_epoch": stop_epoch,
        "model": {
            "model_key": PROBE_MODEL,
            "stand_in": bool(args.stand_in),
            "parameters": parameter_counts(model),
            "device": device,
        },
        "prompt_selection": selection,
        "targets": {"version": TARGET_VERSION, **target_stats.as_dict()},
        "session": {"start_step": start, "final_step": final_step, "target_step": target_step},
        "segments": segments,
        "epochs_published": epochs_out,
        "checkpoint": synced["pointer"],
        "upload_errors": upload_errors,
        "gpu_hours": {
            "prior_estimate": prior.as_dict(),
            "prior_source": prior_source,
            "calibration": calibration or None,
            "actual_this_run": round(hours, 4),
        },
        "sec_per_step_median_by_epoch": steps_s,
    }
    result = write_result(res_dir / "train.json", payload)
    collect(result, str(Path(FINETUNE_DIR) / kind / run))
    losses = [seg["loss_last"] for seg in segments]
    print(f"   {status}: step {start} -> {final_step} of {target_step}; last losses {losses}")
    if status == "stopped_non_finite":
        return EXIT_NON_FINITE
    if status == "stopped:adapter upload failed":
        return EXIT_STORAGE
    if status.startswith("stopped:"):
        return EXIT_BUDGET
    return 0


# ------------------------------------------------------------------------- finetune_eval


def _epoch_adapter(run: str, epoch: int, kind: str) -> tuple[dict, dict]:
    """(spec, adapter pointer) of a run's published epoch adapter: from the run's train results
    (committed, or written earlier this session), else from the adapter repo's listing."""
    for f in reversed(_result_files(_finetune_root(kind) / run, "train")):
        d = json.loads(f.read_text())
        for e in d["epochs_published"]:
            if e["epoch"] == epoch:
                return d["spec"], {**e["adapter"], "step": e["step"]}
    if kind:
        raise SystemExit(f"smoke run {run!r} has no epoch-{epoch} adapter: train it first")
    from lumen import hub

    revision, paths = _hub_run_files(make_hub_api(), run)
    meta_path = f"runs/{run}/epoch-{epoch}/{ADAPTER_META}"
    if meta_path not in paths:
        raise SystemExit(f"run {run!r} has no published epoch-{epoch} adapter (train it first)")
    raw = fetch_model_file(hub.adapter_repo(), meta_path, revision).read_bytes()
    meta = json.loads(raw)
    files = {**meta["files"], ADAPTER_META: hashlib.sha256(raw).hexdigest()}
    pointer = {
        "repo": hub.adapter_repo(),
        "repo_type": "model",
        "path": f"runs/{run}/epoch-{epoch}",
        "revision": revision,
        "files": files,
        "stored": "private Hugging Face model repo",
        "step": meta["step"],
    }
    return meta["spec"], pointer


def _adapter_dir(run: str, epoch: int, kind: str, pointer: dict) -> Path:
    """A local copy of the adapter whose files match the pointer's sha256s exactly."""
    from lumen import hub

    local = CHECKPOINT_DIR / "train" / kind / run / f"epoch-{epoch}"
    if local.is_dir() and hub.folder_digest(local) == pointer["files"]:
        return local
    if pointer.get("revision") is None:
        raise SystemExit(f"{local}: missing or changed, and the smoke adapter was never uploaded")
    dest = CHECKPOINT_DIR / "adapters" / run / f"epoch-{epoch}"
    if dest.is_dir() and hub.folder_digest(dest) == pointer["files"]:
        return dest
    return hub.download_folder(pointer, dest, fetch_file=fetch_model_file)


def _finetuned_backend(args: argparse.Namespace, spec: dict, adapter_dir: Path, adapter: dict):
    """The pinned base with the LoRA adapter merged in (fp16 on the T4), behind the same HFBackend
    the zero-shot runs use: same prompt handling, decoding, extractor, and P(unanswerable)."""
    import torch
    from peft import PeftModel

    from lumen.zeroshot.backends import GenerationConfig, HFBackend, tiny_stand_in

    pin = load_pins().models[PROBE_MODEL]
    if spec["model"]["revision"] != pin.revision:
        raise SystemExit("the adapter was trained on a different base revision than the pin")
    device = args.device or ("cuda" if torch.cuda.is_available() else None)
    if device is None:
        raise SystemExit("no CUDA GPU: attach a T4, or pass --device cpu (only for --stand-in)")
    long_edge = spec["config"]["image_longest_edge"]
    if args.stand_in:
        b = tiny_stand_in(pin, device=device, image_longest_edge=long_edge)
        base, processor = b.model, b.processor
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(pin.repo, revision=pin.revision)
        base = AutoModelForImageTextToText.from_pretrained(
            pin.repo, revision=pin.revision, dtype=torch.float16 if device == "cuda" else None
        ).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    return HFBackend(
        PROBE_MODEL,
        model.to(device),
        processor,
        device=device,
        gen=GenerationConfig(),
        pin=pin,
        stand_in=bool(args.stand_in),
        image_longest_edge=long_edge,
        adapter=adapter,
    )


# Tests replace this to run the stage with a fake backend.
make_finetuned_backend = _finetuned_backend


def _newest(run_dir: Path, name: str) -> dict | None:
    files = _result_files(run_dir, name)
    return json.loads(files[-1].read_text()) if files else None


def _choice(kind: str) -> dict:
    files = _result_files(_finetune_root(kind), "choice")
    if not files:
        raise SystemExit(
            "no fine-tuning choice yet: evaluate the sweep runs on dev, then run finetune_choice "
            "(val is only for the dev-chosen config)"
        )
    return json.loads(files[-1].read_text())


def cmd_finetune_eval(args: argparse.Namespace) -> int:
    """A fine-tuned adapter (one run, one epoch) on dev or val, scored exactly like the baselines.

    dev: every sweep epoch is evaluated here, and the F1 threshold for answerability is chosen on
    these predictions. val: only for the config and epoch finetune_choice picked on dev (its
    seed and ablation replicates included), with the threshold from the same run's dev result.
    """
    import numpy as np

    from lumen.budget import BudgetExceeded, calibrated_estimate, check_budget
    from lumen.eval.answerability import best_f1_threshold, f1_at_threshold
    from lumen.eval.vizwiz_metric import evaluate_accuracy
    from lumen.train.selection import family
    from lumen.zeroshot.backends import NonFiniteOutput
    from lumen.zeroshot.estimates import MODEL_COSTS, estimate_run
    from lumen.zeroshot.extract import UNANSWERABLE
    from lumen.zeroshot.harness import read_predictions, run_predictions, score_predictions
    from lumen.zeroshot.resolution import abstention_breakdown

    run, epoch, split, kind = args.run, args.epoch, args.split, _finetune_kind(args)
    smoke = kind != ""
    spec, pointer = _epoch_adapter(run, epoch, kind)
    prompt = spec["config"]["prompt_key"]
    long_edge = spec["config"]["image_longest_edge"]
    run_dir = _finetune_root(kind) / run
    name = f"{split}-e{epoch}"
    threshold, choice_ref = None, None
    if split == "val":
        choice = _choice(kind)
        chosen = choice["chosen"]
        if family(spec) != family(chosen["spec"]) or epoch != chosen["epoch"]:
            raise SystemExit(
                f"val is only for the config and epoch chosen on dev ({chosen['run']} epoch "
                f"{chosen['epoch']}, and its seed/ablation replicates): {run} epoch {epoch} isn't"
            )
        dev = _newest(run_dir, f"dev-e{epoch}")
        if dev is None or dev["status"] != "completed":
            raise SystemExit(f"{run} epoch {epoch}: evaluate it on dev first (its F1 threshold)")
        threshold = dev["threshold_choice"]["threshold"]
        choice_ref = {"run": chosen["run"], "epoch": chosen["epoch"], "rule": choice["rule"]}

    pins = load_pins()
    records, zip_path = _load_split(split, pins, limit=args.limit)
    pred_path = PREDICTIONS_DIR / FINETUNE_DIR / kind / run / f"{name}.predictions.jsonl"
    _, done = read_predictions(pred_path)
    remaining = len([r for r in records if r["question_id"] not in done])
    prior = estimate_run(PROBE_MODEL, split, prompt, remaining, long_edge)
    res = f"{long_edge or DEFAULT_LONG_EDGE} px"
    print(
        f"\n== finetune_eval | {run} epoch {epoch} | {split} | prompt={prompt} | {res} | "
        f"{remaining}/{len(records)} to go | prior {prior.hours:.2f} GPU h"
    )
    if args.dry_run:
        return 0
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    if not args.stand_in:
        try:
            check_budget(prior, left)
        except BudgetExceeded as e:
            print(f"NOT STARTED (budget): {e}")
            return EXIT_BUDGET
    if not smoke:
        from lumen import hub

        try:
            hub.preflight(make_hub_api(), hub.predictions_repo())
        except hub.HubError as e:
            print(f"NOT STARTED (predictions storage): {e}")
            return EXIT_STORAGE

    adapter = {
        "run": run,
        "epoch": epoch,
        "step": pointer["step"],
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "repo": pointer.get("repo"),
        "path": pointer.get("path"),
        "revision": pointer.get("revision"),
        "adapter_model_sha256": pointer["files"].get("adapter_model.safetensors"),
    }
    backend = make_finetuned_backend(args, spec, _adapter_dir(run, epoch, kind, pointer), adapter)
    batch_size = args.batch_size or MODEL_COSTS[PROBE_MODEL].batch_size
    calibration: dict = {}

    def on_batch(n_done: int, seconds: float) -> None:
        if calibration or n_done < args.calibration_items or n_done >= remaining:
            return
        cal = calibrated_estimate(prior, n_done, seconds)
        calibration.update(estimate=cal.as_dict(), measured_items=n_done, seconds=round(seconds, 2))
        print(f"   calibrated: {seconds / n_done:.3f} s/question -> {cal.hours:.2f} GPU h total")
        if not args.stand_in and left is not None:
            rest = calibrated_estimate(
                estimate_run(PROBE_MODEL, split, prompt, remaining - n_done, long_edge),
                n_done,
                seconds,
            )
            check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)

    t0 = time.monotonic()
    status, error = "completed", None
    try:
        preds, stats = run_predictions(
            backend,
            records,
            ZipImages(zip_path),
            prompt_key=prompt,
            split=split,
            out_path=pred_path,
            batch_size=batch_size,
            on_batch=on_batch,
            prefetch_workers=args.prefetch_workers,
        )
    except BudgetExceeded as e:
        print(f"STOPPED (budget): {e}\n   answered so far is saved in {pred_path}; rerun resumes.")
        return EXIT_BUDGET
    except NonFiniteOutput as e:
        status, error, stats = "failed_non_finite", str(e), None
        _, preds = read_predictions(pred_path)
    hours = (time.monotonic() - t0) / 3600
    payload: dict = {
        "stage": "finetune_eval",
        "status": status,
        "error": error,
        "smoke": smoke,
        "run": run,
        "epoch": epoch,
        "split": split,
        "model": backend.describe(),
        "spec": spec,
        "adapter": adapter,
        "prompt_key": prompt,
        "image_longest_edge": long_edge,
        "batch_size": batch_size,
        "gpu_hours": {
            "prior_estimate": prior.as_dict(),
            "calibration": calibration or None,
            "actual_this_run": round(hours, 4),
        },
        "run_stats": None
        if stats is None
        else {
            "new": stats.new,
            "resumed": stats.resumed,
            "seconds": round(stats.seconds, 1),
            "sec_per_question": round(stats.seconds / stats.new, 4) if stats.new else None,
        },
    }
    if status == "completed":
        m = payload["metrics"] = score_predictions(records, preds)
        ids = [r["question_id"] for r in records]
        y = np.array([r["answerable"] for r in records])
        s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
        acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
        per_q = np.array([100 * acc.per_question[q] for q in ids])
        said = np.array([preds[q]["answer"].strip().lower() == UNANSWERABLE for q in ids])
        payload["answerable_breakdown"] = abstention_breakdown(y, said, per_q)
        if split == "dev":
            t, f1 = best_f1_threshold(y, s)
            payload["threshold_choice"] = {
                "threshold": t,
                "f1_dev": f1,
                "rule": "maximize F1 (unanswerable positive) on this run's dev predictions; "
                "applied unchanged to val",
            }
        else:
            payload["choice"] = choice_ref
            payload["f1_at_dev_threshold"] = {
                "threshold": threshold,
                "f1": f1_at_threshold(y, s, threshold),
                "label": "F1 at the threshold chosen on dev (not the official 0.5)",
            }
        print(
            f"   accuracy {m['accuracy']['overall']:.2f} [{m['accuracy']['ci95']['low']:.2f}, "
            f"{m['accuracy']['ci95']['high']:.2f}]  AP "
            f"{m['answerability']['average_precision']:.2f}  said-unanswerable "
            f"{100 * m['said_unanswerable_rate']:.1f}%  ({hours:.2f} h)"
        )
    else:
        payload["answered"] = len(preds)
    payload["predictions"], storage_error = _store_predictions(
        pred_path,
        f"{FINETUNE_DIR}/{run}/{pred_path.name}",
        f"{run} epoch {epoch} {split} predictions",
        smoke=smoke or status != "completed",
    )
    result = write_result(run_dir / f"{name}.json", payload)
    collect(result, str(Path(FINETUNE_DIR) / kind / run))
    if status != "completed":
        print(f"FAILED: {error}\n   recorded in {result}")
        return EXIT_NON_FINITE
    if storage_error:
        print(f"STORAGE FAILED: {storage_error}\n   the rerun re-scores and uploads again")
        return EXIT_STORAGE
    return 0


# ----------------------------------------------------------------------- finetune_choice


def _run_dirs(kind: str) -> list[Path]:
    root = _finetune_root(kind)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name != "smoke")


def _epoch_results(run_dir: Path, split: str) -> list[dict]:
    """The newest result per epoch for one run and split, completed runs only."""
    out = []
    names = {f.name.split(".")[0] for f in run_dir.glob(f"{split}-e*.json")}
    for name in sorted(names):
        d = _newest(run_dir, name)
        if d is not None and d["status"] == "completed":
            out.append(d)
    return out


def cmd_finetune_choice(args: argparse.Namespace) -> int:
    """Pick the fine-tuning config and epoch on dev (lumen.train.selection.RULE). CPU, seconds."""
    from lumen.train.loop import TrainConfig
    from lumen.train.selection import choose_finetune, family, is_sweep_run, spec_sha256
    from lumen.zeroshot.harness import question_ids_sha256

    kind = "smoke" if args.smoke else ""
    sweep_seed = TrainConfig().seed
    candidates, docs = [], {}
    for run_dir in _run_dirs(kind):
        for d in _epoch_results(run_dir, "dev"):
            if not is_sweep_run(d["spec"], sweep_seed):
                continue
            key = (d["run"], d["epoch"])
            docs[key] = d
            candidates.append(
                {
                    "run": d["run"],
                    "epoch": d["epoch"],
                    "step": d["adapter"]["step"],
                    "accuracy": d["metrics"]["accuracy"]["overall"],
                    "average_precision": d["metrics"]["answerability"]["average_precision"],
                    "said_unanswerable_rate": d["metrics"]["said_unanswerable_rate"],
                    "answerable_breakdown": d["answerable_breakdown"],
                    "lr": d["spec"]["config"]["lr"],
                    "lora_r": d["spec"]["config"]["lora_r"],
                    "epochs_in_schedule": d["spec"]["epochs"],
                }
            )
    question_sets = {docs[k]["metrics"]["question_ids_sha256"] for k in docs}
    if len(question_sets) > 1:
        raise SystemExit("the dev results answered different question sets; can't compare them")
    if not kind and candidates:
        full_dev = question_ids_sha256(read_ids(SPLITS_DIR / SPLIT_FILES["dev"]))
        if question_sets != {full_dev}:
            raise SystemExit("the dev results don't cover all of the committed dev split")
    try:
        choice = choose_finetune(candidates)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    c = choice["chosen"]
    d = docs[(c["run"], c["epoch"])]
    payload = {
        "stage": "finetune_choice",
        "smoke": bool(kind),
        "rule": choice["rule"],
        "sweep_seed": sweep_seed,
        "chosen": {
            **c,
            "spec": d["spec"],
            "family_sha256": spec_sha256(family(d["spec"])),
            "threshold": d["threshold_choice"]["threshold"],
        },
        "ranking": choice["ranking"],
        "data_used": "dev only; no val or test data",
    }
    result = write_result(_finetune_root(kind) / "choice.json", payload)
    collect(result, str(Path(FINETUNE_DIR) / kind))
    print(f"{'run':<24}{'epoch':>6}{'step':>7}{'lr':>9}{'acc':>8}{'AP':>8}{'unans%':>8}")
    for r in choice["ranking"]:
        print(
            f"{r['run']:<24}{r['epoch']:>6}{r['step']:>7}{r['lr']:>9.0e}{r['accuracy']:>8.2f}"
            f"{r['average_precision']:>8.2f}{100 * r['said_unanswerable_rate']:>8.1f}"
        )
    print(f"chosen: {c['run']} epoch {c['epoch']} (dev accuracy {c['accuracy']:.2f})")
    return 0


# ---------------------------------------------------------------------- finetune_compare


def _predictions_of(doc: dict, local: Path) -> dict[str, dict]:
    """A result's predictions: the local file if its sha256 matches the result, else the Hub."""
    from lumen.zeroshot.harness import read_predictions

    pointer = doc["predictions"]
    if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == pointer["sha256"]:
        return read_predictions(local)[1]
    if pointer.get("stored") == "private Hugging Face dataset":
        return read_predictions(download_predictions(pointer))[1]
    raise SystemExit(f"{local.name}: not on disk and not stored on the Hub")


def cmd_finetune_compare(args: argparse.Namespace) -> int:
    """The Stage 3 results table on val: fine-tuned (each seed) vs the matched zero-shot baseline,
    paired on the same questions, and the no-unanswerable ablation vs the full run. CPU."""
    import numpy as np

    from lumen.eval.paired import paired_difference, seed_summary
    from lumen.eval.vizwiz_metric import evaluate_accuracy
    from lumen.train.selection import family
    from lumen.zeroshot.harness import question_ids_sha256

    kind = "smoke" if args.smoke else ""
    choice = _choice(kind)
    chosen = choice["chosen"]
    epoch = chosen["epoch"]
    prompt = chosen["spec"]["config"]["prompt_key"]
    long_edge = chosen["spec"]["config"]["image_longest_edge"]

    systems: dict[str, tuple[dict, Path]] = {}
    main_runs, ablation_runs = [], []
    for run_dir in _run_dirs(kind):
        d = _newest(run_dir, f"val-e{epoch}")
        if d is None or d["status"] != "completed":
            continue
        if family(d["spec"]) != family(chosen["spec"]):
            continue
        local = PREDICTIONS_DIR / FINETUNE_DIR / kind / d["run"] / f"val-e{epoch}.predictions.jsonl"
        systems[d["run"]] = (d, local)
        cfg = d["spec"]["config"]
        (main_runs if cfg["include_unanswerable"] else ablation_runs).append(d["run"])
    if not main_runs:
        raise SystemExit(f"no val result for the chosen config at epoch {epoch}: run finetune_eval")
    base_kind = "smoke" if kind else ""
    base_name = _run_name("val", prompt, long_edge)
    base_doc = _newest(_run_dir(PROBE_MODEL, base_kind), base_name)
    if base_doc is None or base_doc["status"] != "completed":
        raise SystemExit(
            f"no matched zero-shot baseline ({PROBE_MODEL} {base_name}): run `baseline --model "
            f"{PROBE_MODEL} --split val --prompt dev-selected --long-edge chosen` first"
        )
    base_local = _pred_path(PROBE_MODEL, "val", prompt, base_kind, long_edge)
    base_preds = _predictions_of(base_doc, base_local)

    ids = sorted(base_preds)
    by_id = {
        r["question_id"]: r for r in _load_split("val", load_pins(), limit=None, images=False)[0]
    }
    records = [by_id[q] for q in ids]
    y = np.array([r["answerable"] for r in records])

    def per_question(preds: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
        if sorted(preds) != ids:
            raise SystemExit("systems answered different val questions; refusing to compare")
        acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
        a = np.array([100 * acc.per_question[q] for q in ids])
        s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
        return a, s

    a0, s0 = per_question(base_preds)
    scored = {run: per_question(_predictions_of(d, local)) for run, (d, local) in systems.items()}

    def row(run: str) -> dict:
        d = systems[run][0]
        m = d["metrics"]
        a, s = scored[run]
        return {
            "run": run,
            "seed": d["spec"]["config"]["seed"],
            "include_unanswerable": d["spec"]["config"]["include_unanswerable"],
            "accuracy": m["accuracy"]["overall"],
            "accuracy_ci95": m["accuracy"]["ci95"],
            "per_answer_type": m["accuracy"]["per_answer_type"],
            "average_precision": m["answerability"]["average_precision"],
            "f1_at_0_5_official": m["answerability"]["f1_at_0_5_official"],
            "f1_at_dev_threshold": d["f1_at_dev_threshold"],
            "said_unanswerable_rate": m["said_unanswerable_rate"],
            "answerable_breakdown": d["answerable_breakdown"],
            "vs_zero_shot": paired_difference(y, a, a0, s, s0),
            "result": d["run"],
        }

    mains = [row(r) for r in sorted(main_runs)]
    ablations = []
    for r in sorted(ablation_runs):
        ab = row(r)
        seed = ab["seed"]
        partner = next((m for m in mains if m["seed"] == seed), mains[0])
        a, s = scored[r]
        pa, ps = scored[partner["run"]]
        ab["vs_full_training"] = {"partner": partner["run"], **paired_difference(y, a, pa, s, ps)}
        ablations.append(ab)
    bm = base_doc["metrics"]
    payload = {
        "stage": "finetune_compare",
        "smoke": bool(kind),
        "split": "val",
        "n": len(ids),
        "question_ids_sha256": question_ids_sha256(ids),
        "choice": {"run": chosen["run"], "epoch": epoch, "rule": choice["rule"]},
        "zero_shot": {
            "model": PROBE_MODEL,
            "result": base_name,
            "accuracy": bm["accuracy"]["overall"],
            "accuracy_ci95": bm["accuracy"]["ci95"],
            "average_precision": bm["answerability"]["average_precision"],
            "said_unanswerable_rate": bm["said_unanswerable_rate"],
        },
        "fine_tuned": mains,
        "across_seeds": {
            "accuracy": seed_summary([m["accuracy"] for m in mains]),
            "average_precision": seed_summary([m["average_precision"] for m in mains]),
            "f1_at_dev_threshold": seed_summary([m["f1_at_dev_threshold"]["f1"] for m in mains]),
            "accuracy_gain_vs_zero_shot": seed_summary(
                [m["vs_zero_shot"]["accuracy_diff"] for m in mains]
            ),
            "ap_gain_vs_zero_shot": seed_summary([m["vs_zero_shot"]["ap_diff"] for m in mains]),
        },
        "ablation_no_unanswerable": ablations,
    }
    result = write_result(_finetune_root(kind) / "comparison.json", payload)
    collect(result, str(Path(FINETUNE_DIR) / kind))
    print(f"val, n={len(ids)}; fine-tuned = {chosen['run']} config at epoch {epoch}")
    print(f"{'system':<28}{'acc':>8}{'vs 0-shot [95% CI]':>26}{'AP':>8}{'dAP':>8}{'unans%':>8}")
    print(
        f"{'zero-shot ' + base_name:<28}{bm['accuracy']['overall']:>8.2f}{'':>26}"
        f"{bm['answerability']['average_precision']:>8.2f}{'':>8}"
        f"{100 * bm['said_unanswerable_rate']:>8.1f}"
    )
    for m in mains + ablations:
        v = m["vs_zero_shot"]
        ci = v["accuracy_diff_ci95"]
        diff = f"{v['accuracy_diff']:+.2f} [{ci['low']:+.2f}, {ci['high']:+.2f}]"
        print(
            f"{m['run']:<28}{m['accuracy']:>8.2f}{diff:>26}{m['average_precision']:>8.2f}"
            f"{v['ap_diff']:>+8.2f}{100 * m['said_unanswerable_rate']:>8.1f}"
        )
    acc = payload["across_seeds"]["accuracy"]
    std = f" +/- {acc['std']:.2f}" if acc["std"] is not None else " (one seed)"
    print(f"across {acc['n_seeds']} seed(s): accuracy {acc['mean']:.2f}{std}")
    return 0


# --------------------------------------------------------------------------- pin_adapter


def cmd_pin_adapter(args: argparse.Namespace) -> int:
    """Print the configs/pins.yaml `adapters:` entry for one run's epoch adapter (no GPU, no
    upload). The Mac (Stage 4) downloads an adapter only through such a pin: exact Hub commit and
    every file's sha256."""
    import yaml

    spec, pointer = _epoch_adapter(args.run, args.epoch, "")
    if pointer.get("revision") is None:
        raise SystemExit("this adapter was never uploaded (a smoke run?); nothing to pin")
    cfg = spec["config"]
    entry = {
        f"{args.run}-e{args.epoch}": {
            "repo": pointer["repo"],
            "revision": pointer["revision"],
            "path": pointer["path"],
            "base": spec["model"]["model_key"],
            "run": args.run,
            "epoch": args.epoch,
            "step": pointer["step"],
            "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "image_longest_edge": cfg["image_longest_edge"],
            "prompt_key": cfg["prompt_key"],
            "license": "apache-2.0",
            "role": "fine-tuned LoRA adapter (Stage 3)",
            "files": dict(sorted(pointer["files"].items())),
        }
    }
    print("# add under `adapters:` in configs/pins.yaml")
    print(yaml.safe_dump(entry, sort_keys=False).rstrip())
    return 0


# ---------------------------------------------------------------------------- final_test

FINAL_TEST_HUB_RECORD = "test/FINAL_TEST_RUN.json"


def final_test_record_path() -> Path:
    return RESULTS / "test" / "FINAL_TEST_RUN.json"


def final_test_git() -> dict:
    """Tests replace this; the record names the exact commit that produced the numbers."""
    from lumen.runmeta import git_info

    return git_info()


def cmd_final_test(args: argparse.Namespace) -> int:
    """The single, sealed evaluation on VizWiz test (lumen.eval.test_seal). Never rerun.

    Two systems answer the same 8,000 test questions in this one run: the fine-tuned adapter
    chosen on dev (the default-seed run at the chosen epoch, F1 at its dev threshold) and the
    matched zero-shot SmolVLM2-500M (same resolution and prompt). Everything that can fail cheaply
    is checked *before* the seal is claimed (opt-in, clean tree, no existing record here or on the
    Hub, quota, storage, disk). The claim is uploaded to the private predictions dataset before
    any test answer is loaded, so a wiped Kaggle disk can't hide a crashed run. There is no
    resume: a crashed run stays claimed, and what to report is a human decision in the devlog.
    """
    import numpy as np

    from lumen import hub
    from lumen.budget import BudgetExceeded, check_budget, prior_estimate
    from lumen.eval.answerability import f1_at_threshold
    from lumen.eval.paired import paired_difference
    from lumen.eval.test_seal import OPT_IN_FLAG, TestSealed, open_final_test_run
    from lumen.eval.vizwiz_metric import evaluate_accuracy
    from lumen.zeroshot.estimates import MODEL_COSTS, estimate_run
    from lumen.zeroshot.harness import run_predictions, score_predictions

    pins = load_pins()
    choice = _choice("")
    chosen = choice["chosen"]
    run, epoch = chosen["run"], chosen["epoch"]
    spec, pointer = _epoch_adapter(run, epoch, "")
    prompt = spec["config"]["prompt_key"]
    long_edge = spec["config"]["image_longest_edge"]
    dev = _newest(_finetune_root("") / run, f"dev-e{epoch}")
    if dev is None:
        raise SystemExit(f"{run} epoch {epoch} has no dev result (its F1 threshold)")
    threshold = dev["threshold_choice"]["threshold"]
    n_test = 8000
    one = estimate_run(PROBE_MODEL, "test", prompt, n_test, long_edge)
    est = prior_estimate("final_test", 2 * n_test, one.sec_per_item, overhead_s=2 * one.overhead_s)
    print(
        f"\n== final_test | {run} epoch {epoch} (dev-chosen) + zero-shot {PROBE_MODEL} | "
        f"{long_edge or DEFAULT_LONG_EDGE} px | prompt={prompt} | 2 x {n_test} questions | "
        f"{est.hours:.2f} GPU h (prior)"
    )
    if args.dry_run:
        print("dry run: nothing claimed, nothing downloaded")
        return 0
    if not args.final_test_evaluation:
        raise TestSealed(f"the test split is sealed; pass {OPT_IN_FLAG} to run it, once")
    git = final_test_git()
    if not git.get("commit") or git.get("dirty"):
        raise TestSealed("the final test run needs a clean checkout of a commit")
    record_path = final_test_record_path()
    if record_path.exists():
        raise TestSealed(f"{record_path} exists: the final test evaluation already ran")
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    try:
        check_budget(est, left)
    except BudgetExceeded as e:
        print(f"NOT STARTED (budget): {e}")
        return EXIT_BUDGET
    api, repo = make_hub_api(), hub.predictions_repo()
    try:
        hub.preflight(api, repo)
        _, existing = hub.list_files(api, repo, "test/")
    except hub.HubError as e:
        print(f"NOT STARTED (predictions storage): {e}")
        return EXIT_STORAGE
    if FINAL_TEST_HUB_RECORD in existing:
        raise TestSealed(
            f"{repo} already holds {FINAL_TEST_HUB_RECORD}: the final test evaluation was claimed "
            "in an earlier session. Download that record and report it; never run test again."
        )
    need = sum(
        pins.files[k].size
        for k in ("test_answers", "test_images")
        if not (DATA_DIR / pins.files[k].filename).exists()
    )
    ensure_free_space(DATA_DIR, need)

    # ---- the seal is claimed from here on: no retries, ever ----
    final = open_final_test_run(
        opt_in=True,
        purpose=f"final VizWiz test: {run} epoch {epoch} vs zero-shot {PROBE_MODEL}",
        record_path=record_path,
        git=git,
    )
    collect(record_path, "test")
    claim = hub.upload(api, repo, record_path, FINAL_TEST_HUB_RECORD, "final test run: claimed")
    print(f"   claimed: {record_path} (also on the Hub @{claim['revision'][:12]})")

    answers_file = fetch_pinned(pins.files["test_answers"], DATA_DIR)
    images_file = fetch_pinned(pins.files["test_images"], DATA_DIR)
    records = parse_annotations(
        json.loads(answers_file.path.read_text(encoding="utf-8")), "test", final_test_run=final
    )
    questions = parse_annotations(
        _annotation_json(fetch_pinned(pins.files["annotations"], DATA_DIR).path, "test"),
        "test",
        require_answers=False,
    )
    if sorted(r["question_id"] for r in records) != sorted(q["question_id"] for q in questions):
        raise SystemExit("VQA_test.json and the official test.json list different questions")
    records.sort(key=lambda r: r["question_id"])
    images = ZipImages(images_file.path)

    adapter = {
        "run": run,
        "epoch": epoch,
        "step": pointer["step"],
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "repo": pointer.get("repo"),
        "path": pointer.get("path"),
        "revision": pointer.get("revision"),
        "adapter_model_sha256": pointer["files"].get("adapter_model.safetensors"),
    }
    systems = {
        "fine_tuned": lambda: make_finetuned_backend(
            args, spec, _adapter_dir(run, epoch, "", pointer), adapter
        ),
        "zero_shot": lambda: make_backend(
            PROBE_MODEL, argparse.Namespace(**{**vars(args), "long_edge": long_edge})
        ),
    }
    ids = [r["question_id"] for r in records]
    y = np.array([r["answerable"] for r in records])
    out: dict = {}
    scored: dict = {}
    t0 = time.monotonic()
    for name, make in systems.items():
        backend = make()
        pred_path = PREDICTIONS_DIR / "test" / f"{name}.predictions.jsonl"
        preds, _ = run_predictions(
            backend,
            records,
            images,
            prompt_key=prompt,
            split="test",
            out_path=pred_path,
            batch_size=args.batch_size or MODEL_COSTS[PROBE_MODEL].batch_size,
            prefetch_workers=PREFETCH_ON,
        )
        metrics = score_predictions(records, preds)
        acc = evaluate_accuracy({q: p["answer"] for q, p in preds.items()}, records)
        a = np.array([100 * acc.per_question[q] for q in ids])
        s = np.array([1.0 - preds[q]["p_unanswerable"] for q in ids])
        scored[name] = (a, s)
        out[name] = {"model": backend.describe(), "metrics": metrics}
        if name == "fine_tuned":
            out[name]["f1_at_dev_threshold"] = {
                "threshold": threshold,
                "f1": f1_at_threshold(y, s, threshold),
                "label": "F1 at the threshold chosen on dev (not the official 0.5)",
            }
        pointer_, err = _store_predictions(
            pred_path, f"test/{pred_path.name}", f"final test: {name}", smoke=False
        )
        out[name]["predictions"] = pointer_
        if err:
            out[name]["storage_error"] = err
        print(
            f"   {name}: accuracy {metrics['accuracy']['overall']:.2f} "
            f"[{metrics['accuracy']['ci95']['low']:.2f}, {metrics['accuracy']['ci95']['high']:.2f}]"
            f"  AP {metrics['answerability']['average_precision']:.2f}"
        )
        del backend
        _BACKENDS.clear()  # the zero-shot backend is cached by make_backend; free it too
        _free_gpu()
    (fa, fs), (za, zs) = scored["fine_tuned"], scored["zero_shot"]
    results = {
        "n": len(records),
        "choice": {"run": run, "epoch": epoch, "rule": choice["rule"]},
        "image_longest_edge": long_edge,
        "prompt_key": prompt,
        "systems": out,
        "fine_tuned_vs_zero_shot": paired_difference(y, fa, za, fs, zs),
        "test_files_sha256": {
            "test_answers": answers_file.sha256,
            "test_images": images_file.sha256,
        },
        "gpu_hours": {"prior_estimate": est.as_dict(), "actual": (time.monotonic() - t0) / 3600},
    }
    final.complete(results)
    try:
        hub.upload(api, repo, record_path, FINAL_TEST_HUB_RECORD, "final test run: completed")
    except Exception as e:  # the completed record is on disk and in the download folder
        print(f"WARNING: couldn't upload the completed record ({e}); download it from Kaggle")
    dl = DOWNLOAD_DIR / "test" / record_path.name
    dl.unlink(missing_ok=True)  # replace the "started" copy with the completed record
    collect(record_path, "test")
    d = results["fine_tuned_vs_zero_shot"]
    ci = d["accuracy_diff_ci95"]
    print(
        f"   fine-tuned - zero-shot: {d['accuracy_diff']:+.2f} [{ci['low']:+.2f}, "
        f"{ci['high']:+.2f}] accuracy, {d['ap_diff']:+.2f} AP. Commit {record_path} now."
    )
    return 0


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# -------------------------------------------------------------------------- storage_check


def cmd_storage_check(args: argparse.Namespace) -> int:
    """Confirm HF_TOKEN can write to both private Hub repos. No GPU, no data, nothing uploaded."""
    from lumen import hub

    try:
        api = make_hub_api()
        checks = [
            hub.preflight(api, hub.predictions_repo(), "dataset"),
            hub.preflight(api, hub.adapter_repo(), "model"),
        ]
    except hub.HubError as e:
        print(f"STORAGE CHECK FAILED: {e}")
        return EXIT_STORAGE
    for c in checks:
        print(f"ok: {c['repo']} ({c['repo_type']}) is private and writable with HF_TOKEN")
    return 0


def _workers_arg(value: str) -> int | str:
    if value == "auto":
        return value
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0 or 'auto'")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("mac_sample").set_defaults(fn=cmd_mac_sample)
    sub.add_parser("baseline_summary", help="table of all baseline results").set_defaults(
        fn=cmd_baseline_summary
    )
    sub.add_parser(
        "storage_check", help="check HF_TOKEN against the private Hub repos"
    ).set_defaults(fn=cmd_storage_check)
    sub.add_parser(
        "resolution_choice", help="pick the image resolution from the dev probe (CPU)"
    ).set_defaults(fn=cmd_resolution_choice)
    rb = sub.add_parser(
        "rebuild_baseline", help="re-create lost baseline results from Hub predictions (CPU)"
    )
    rb.add_argument("--model", default="smolvlm2-500m")
    rb.set_defaults(fn=cmd_rebuild_baseline)
    b = sub.add_parser("baseline", help="zero-shot baselines (Stage 2)")
    b.add_argument("--model", nargs="+", default=["all"], help="pinned model keys, or 'all'")
    b.add_argument("--split", choices=["dev", "dev_select", "val"], required=True)
    b.add_argument("--prompt", choices=[*PROMPTS, "both", "dev-selected"], required=True)
    b.add_argument(
        "--gpu-hours-left",
        type=float,
        default=None,
        help="remaining weekly Kaggle GPU quota (kaggle.com -> Settings); required",
    )
    b.add_argument("--batch-size", type=int, default=None)
    b.add_argument("--calibration-items", type=int, default=32)
    b.add_argument(
        "--prefetch-workers",
        type=_workers_arg,
        default=PREFETCH_ON,  # measured faster on the T4 (0.666 vs 0.798 s/question, 2026-09-30)
        help="threads preparing the next batches while the GPU runs; 0 = off; 'auto' times "
        "prefetching on vs off on the first real batches and keeps the faster",
    )
    b.add_argument("--limit", type=int, default=None, help="smoke run on the first N questions")
    b.add_argument(
        "--stand-in",
        action="store_true",
        help="tiny random-weight model with the real processor (pipeline check)",
    )
    b.add_argument(
        "--long-edge",
        type=_long_edge_arg,
        default=None,
        help="cap the image's long edge in px (SmolVLM2-500M/256M: 512, 1024, 1536, 2048 = "
        "default), or 'chosen' for the resolution picked by resolution_choice",
    )
    b.add_argument("--device", choices=["cuda", "cpu"], default=None)
    b.add_argument("--dry-run", action="store_true", help="print the GPU-hour plan and exit")
    b.set_defaults(fn=cmd_baseline)
    t = sub.add_parser("train_speed", help="short LoRA run: training cost per step and epoch")
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--micro-batch", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--calibration-steps", type=int, default=10)
    t.add_argument(
        "--long-edge",
        type=_long_edge_arg,
        required=True,
        help="image long edge in px (512/1024/1536/2048), or 'chosen' (resolution_choice)",
    )
    t.add_argument("--gpu-hours-left", type=float, default=None)
    t.add_argument("--limit", type=int, default=None, help="smoke run on the first N train_fit")
    t.add_argument("--stand-in", action="store_true", help="tiny random-weight model (pipeline)")
    t.add_argument("--device", choices=["cuda", "cpu"], default=None)
    t.add_argument("--dry-run", action="store_true", help="print the GPU-hour prior and exit")
    t.set_defaults(fn=cmd_train_speed)
    tr = sub.add_parser("train", help="LoRA fine-tuning on train_fit (Stage 3)")
    tr.add_argument("--run", type=_run_name_arg, required=True, help="name bound to this config")
    tr.add_argument("--epochs", type=int, default=1, help="length of the LR schedule, in epochs")
    tr.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="train only the first k epochs of the schedule (replicates of a chosen epoch)",
    )
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=None, help="default: 2 x rank")
    tr.add_argument("--warmup-steps", type=int, default=50)
    tr.add_argument("--seed", type=int, default=None, help="default: the pinned split seed")
    tr.add_argument(
        "--no-unanswerable", action="store_true", help="ablation: drop unanswerable examples"
    )
    tr.add_argument("--micro-batch", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=2)
    tr.add_argument("--checkpoint-every", type=int, default=250)
    tr.add_argument("--calibration-steps", type=int, default=10)
    tr.add_argument(
        "--long-edge",
        type=_long_edge_arg,
        required=True,
        help="image long edge in px (512/1024/1536/2048), or 'chosen' (resolution_choice)",
    )
    tr.add_argument("--gpu-hours-left", type=float, default=None)
    tr.add_argument(
        "--session-hours",
        type=float,
        default=11.0,
        help="stop cleanly (checkpoint uploaded) after this long; Kaggle ends sessions at 12 h",
    )
    tr.add_argument("--limit", type=int, default=None, help="smoke run on the first N train_fit")
    tr.add_argument("--stand-in", action="store_true", help="tiny random-weight model (pipeline)")
    tr.add_argument("--device", choices=["cuda", "cpu"], default=None)
    tr.add_argument("--dry-run", action="store_true", help="print the plan and GPU hours, exit")
    tr.set_defaults(fn=cmd_train)
    fe = sub.add_parser("finetune_eval", help="a fine-tuned adapter on dev or val")
    fe.add_argument("--run", type=_run_name_arg, required=True)
    fe.add_argument("--epoch", type=int, required=True)
    fe.add_argument("--split", choices=["dev", "val"], required=True)
    fe.add_argument("--gpu-hours-left", type=float, default=None)
    fe.add_argument("--batch-size", type=int, default=None)
    fe.add_argument("--calibration-items", type=int, default=32)
    fe.add_argument("--prefetch-workers", type=int, default=PREFETCH_ON)
    fe.add_argument("--limit", type=int, default=None, help="smoke run on the first N questions")
    fe.add_argument("--stand-in", action="store_true", help="tiny random-weight model (pipeline)")
    fe.add_argument("--device", choices=["cuda", "cpu"], default=None)
    fe.add_argument("--dry-run", action="store_true", help="print the GPU-hour plan and exit")
    fe.set_defaults(fn=cmd_finetune_eval)
    fc = sub.add_parser("finetune_choice", help="pick the config and epoch on dev (CPU)")
    fc.add_argument("--smoke", action="store_true", help="choose among smoke runs (pipeline)")
    fc.set_defaults(fn=cmd_finetune_choice)
    fp = sub.add_parser("finetune_compare", help="val table vs the zero-shot baseline (CPU)")
    fp.add_argument("--smoke", action="store_true", help="compare smoke runs (pipeline)")
    fp.set_defaults(fn=cmd_finetune_compare)
    pa = sub.add_parser("pin_adapter", help="print the pins.yaml entry for an epoch adapter")
    pa.add_argument("--run", type=_run_name_arg, required=True)
    pa.add_argument("--epoch", type=int, required=True)
    pa.set_defaults(fn=cmd_pin_adapter)
    ft = sub.add_parser("final_test", help="the single sealed VizWiz test evaluation (GPU)")
    ft.add_argument(
        "--final-test-evaluation",
        action="store_true",
        help="opt in to the one final test run; it can never be repeated",
    )
    ft.add_argument("--gpu-hours-left", type=float, default=None)
    ft.add_argument("--batch-size", type=int, default=None)
    ft.add_argument("--dry-run", action="store_true", help="print the plan; claims nothing")
    ft.set_defaults(fn=cmd_final_test, stand_in=False, device=None)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
