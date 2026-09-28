#!/usr/bin/env python3
"""Kaggle runner for Lumen. All dataset downloads and all GPU work happen here, never on the Mac.

Stages (each writes result JSON under results/ and copies it to the download folder):

    env         print and save the machine, GPU, and library versions
    data        download + verify the pinned VizWiz files, validate annotations against the
                image zips, build the dev split and the Mac sample ID lists      (CPU only)
    mac_sample  package the ~300 pinned val images + their annotations for the Mac   (CPU only)

    baseline    zero-shot baselines (Stage 2) for the pinned models on dev or val        (GPU)
                --dry-run prints the GPU-hour plan without loading anything

Later stages: train and eval (Stage 3). Every GPU stage goes through the quota guard in
lumen.budget. The final test evaluation is sealed (lumen.eval.test_seal) and needs
--final-test-evaluation.

Only files in the download folder leave Kaggle: result JSONs, split ID lists, and the Mac sample.
Image zips stay on Kaggle.

    uv run python kaggle/run_stages.py env
    uv run python kaggle/run_stages.py data
    uv run python kaggle/run_stages.py mac_sample
    uv run python kaggle/run_stages.py baseline --model all --split dev --prompt both --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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
from lumen.data.splits import make_dev_split, make_mac_sample, read_ids, write_ids  # noqa: E402
from lumen.pins import SPLITS_DIR, load_pins  # noqa: E402
from lumen.runmeta import hardware_info, run_metadata, write_result  # noqa: E402
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
SPLIT_FILES = {
    "train_fit": "train_fit.txt",
    "dev": "dev.txt",
    "mac_sample": "mac_sample_val.txt",
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
    generated = {"train_fit": fit, "dev": dev, "mac_sample": mac}
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
EXIT_BUDGET, EXIT_NON_FINITE, EXIT_STORAGE = 3, 4, 5
SESSION_START = time.monotonic()


def _load_split(split: str, pins, *, limit: int | None) -> tuple[list[Record], Path]:
    """Records and image zip for dev (committed IDs, from official train) or val (all)."""
    ann = fetch_pinned(pins.files["annotations"], DATA_DIR).path
    if split == "dev":
        ids = set(read_ids(SPLITS_DIR / SPLIT_FILES["dev"]))
        records = [
            r
            for r in parse_annotations(_annotation_json(ann, "train"), "train")
            if r["question_id"] in ids
        ]
        if len(records) != len(ids):
            raise SystemExit(f"dev: {len(ids) - len(records)} committed IDs not in train.json")
        zip_pin = pins.files["train_images"]
    elif split == "val":
        records = parse_annotations(_annotation_json(ann, "val"), "val")
        zip_pin = pins.files["val_images"]
    else:
        raise SystemExit(f"baseline runs on dev or val, not {split!r} (test is sealed)")
    records.sort(key=lambda r: r["question_id"])
    if limit is not None:
        records = records[:limit]
    return records, fetch_pinned(zip_pin, DATA_DIR).path


def _run_dir(model_key: str, smoke: bool) -> Path:
    return RESULTS / BASELINE_DIR / ("smoke" if smoke else "") / model_key


def _pred_path(model_key: str, split: str, prompt_key: str, smoke: bool) -> Path:
    return (
        PREDICTIONS_DIR
        / BASELINE_DIR
        / ("smoke" if smoke else "")
        / model_key
        / f"{split}-{prompt_key}.predictions.jsonl"
    )


def make_hub_api():
    """Tests replace this with a fake Hub client."""
    from lumen import hub

    return hub.make_api()


def _dev_selected_prompt(model_key: str) -> tuple[str, dict]:
    """The prompt with the best dev accuracy (ties: higher dev AP, then name). Dev only."""
    found = {}
    for key in PROMPTS:
        files = sorted(_run_dir(model_key, False).glob(f"dev-{key}.json")) + sorted(
            _run_dir(model_key, False).glob(f"dev-{key}.[0-9]*.json")
        )
        if not files:
            raise SystemExit(
                f"{model_key}: no dev result for prompt {key!r}; "
                "run --split dev --prompt both first"
            )
        doc = json.loads(files[-1].read_text())
        found[key] = {
            "file": str(files[-1].relative_to(REPO_ROOT)),
            "accuracy": doc["metrics"]["accuracy"]["overall"],
            "average_precision": doc["metrics"]["answerability"]["average_precision"],
        }
    best = max(sorted(found), key=lambda k: (found[k]["accuracy"], found[k]["average_precision"]))
    return best, {
        "rule": "max dev accuracy; ties by dev AP, then name",
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
    if args.stand_in:
        return tiny_stand_in(pin, device=device, gen=gen)
    dtype = "float16" if device == "cuda" else "float32"  # T4: no bf16
    return HFBackend.from_pin(pin, device=device, dtype=dtype, gen=gen)


_BACKENDS: dict[str, object] = {}


def _cached_backend(model_key: str, args: argparse.Namespace):
    """Load each model once per process, even when it runs several prompts."""
    if model_key not in _BACKENDS:
        _BACKENDS[model_key] = _make_backend(model_key, args)
    return _BACKENDS[model_key]


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
    smoke = args.limit is not None or args.stand_in
    records, zip_path = _load_split(split, pins, limit=args.limit)
    run_dir = _run_dir(model_key, smoke)
    pred_path = _pred_path(model_key, split, prompt_key, smoke)
    _, done = read_predictions(pred_path)
    remaining = len([r for r in records if r["question_id"] not in done])
    prior = estimate_run(model_key, split, prompt_key, remaining)
    used = (time.monotonic() - SESSION_START) / 3600
    left = None if args.gpu_hours_left is None else args.gpu_hours_left - used
    print(
        f"\n== {model_key} | {split} | prompt={prompt_key} | {remaining}/{len(records)} to go"
        f" | prior estimate {prior.hours:.2f} GPU h"
    )
    if not args.stand_in:
        check_budget(prior, left)

    backend = make_backend(model_key, args)
    batch_size = args.batch_size or MODEL_COSTS[model_key].batch_size
    header = predictions_header(backend, prompt_key, split)
    calibration: dict = {}

    def on_batch(n_done: int, seconds: float) -> None:
        if calibration or n_done < args.calibration_items or n_done >= remaining:
            return
        cal = calibrated_estimate(prior, n_done, seconds)
        rest = calibrated_estimate(
            estimate_run(model_key, split, prompt_key, remaining - n_done), n_done, seconds
        )
        calibration.update(
            estimate=cal.as_dict(), measured_items=n_done, measured_seconds=round(seconds, 2)
        )
        print(f"   calibrated: {seconds / n_done:.3f} s/question -> {cal.hours:.2f} GPU h total")
        if not args.stand_in and left is not None:
            check_budget(rest, left - (time.monotonic() - SESSION_START) / 3600 + used)

    t0 = time.monotonic()
    status, error = "completed", None
    try:
        preds, stats = run_predictions(
            backend,
            records,
            ZipImages(zip_path),
            prompt_key=prompt_key,
            split=split,
            out_path=pred_path,
            batch_size=batch_size,
            on_batch=on_batch,
        )
    except BudgetExceeded as e:
        print(f"STOPPED (budget): {e}\n   answered so far is saved in {pred_path}; rerun resumes.")
        return EXIT_BUDGET
    except NonFiniteOutput as e:
        status, error = "failed_non_finite", str(e)
        _, preds = read_predictions(pred_path)
        stats = None
    hours = (time.monotonic() - t0) / 3600

    payload = {
        "stage": "baseline",
        "status": status,
        "error": error,
        "smoke": smoke,
        "model": backend.describe(),
        "split": split,
        "prompt": header,
        "prompt_selection": prompt_selection,
        "batch_size": batch_size,
        "gpu_hours": {
            "prior_estimate": prior.as_dict(),
            "calibration": calibration or None,
            "actual_this_run": round(hours, 4),
        },
        "run": None
        if stats is None
        else {"new": stats.new, "resumed": stats.resumed, "seconds": round(stats.seconds, 1)},
    }
    if status == "completed":
        payload["metrics"] = score_predictions(records, preds)
    else:
        payload["answered"] = len(preds)
    payload["predictions"], storage_error = _store_predictions(
        pred_path, model_key, split, prompt_key, smoke=smoke or status != "completed"
    )
    result = write_result(run_dir / f"{split}-{prompt_key}.json", payload)
    collect(result, f"{BASELINE_DIR}/{'smoke/' if smoke else ''}{model_key}")
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
    return 0


def _store_predictions(
    pred_path: Path, model_key: str, split: str, prompt_key: str, *, smoke: bool
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
            f"{BASELINE_DIR}/{model_key}/{pred_path.name}",
            f"{model_key} {split} {prompt_key} predictions",
        )
    except Exception as e:  # recorded, never swallowed: the stage exits nonzero
        return {**local, "stored": "upload failed", "error": f"{type(e).__name__}: {e}"}, str(e)
    return {**pointer, "stored": "private Hugging Face dataset"}, None


def _baseline_plan(models: list[str], split: str, prompts: list[str], args) -> list[tuple]:
    from lumen.zeroshot.estimates import estimate_run
    from lumen.zeroshot.harness import read_predictions

    pins = load_pins()
    n_split = {"dev": len(read_ids(SPLITS_DIR / SPLIT_FILES["dev"]))}
    if split == "val":
        n_split["val"] = json.loads((RESULTS / "data" / "vizwiz_data.json").read_text())["summary"][
            "val"
        ]["n"]
    plan = []
    for m in models:
        for p in prompts:
            n = min(n_split[split], args.limit) if args.limit else n_split[split]
            pred = _pred_path(m, split, p, args.limit is not None or args.stand_in)
            done = len(read_predictions(pred)[1])
            plan.append((m, p, n - done, estimate_run(m, split, p, max(n - done, 0))))
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
    if args.split == "dev" and args.prompt == "dev-selected":
        raise SystemExit("--prompt dev-selected is for val")

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("mac_sample").set_defaults(fn=cmd_mac_sample)
    b = sub.add_parser("baseline", help="zero-shot baselines (Stage 2)")
    b.add_argument("--model", nargs="+", default=["all"], help="pinned model keys, or 'all'")
    b.add_argument("--split", choices=["dev", "val"], required=True)
    b.add_argument("--prompt", choices=[*PROMPTS, "both", "dev-selected"], required=True)
    b.add_argument(
        "--gpu-hours-left",
        type=float,
        default=None,
        help="remaining weekly Kaggle GPU quota (kaggle.com -> Settings); required",
    )
    b.add_argument("--batch-size", type=int, default=None)
    b.add_argument("--calibration-items", type=int, default=32)
    b.add_argument("--limit", type=int, default=None, help="smoke run on the first N questions")
    b.add_argument(
        "--stand-in",
        action="store_true",
        help="tiny random-weight model with the real processor (pipeline check)",
    )
    b.add_argument("--device", choices=["cuda", "cpu"], default=None)
    b.add_argument("--dry-run", action="store_true", help="print the GPU-hour plan and exit")
    b.set_defaults(fn=cmd_baseline)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
