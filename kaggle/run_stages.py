#!/usr/bin/env python3
"""Kaggle runner for Lumen. All dataset downloads and all GPU work happen here, never on the Mac.

Stages (each writes result JSON under results/ and copies it to the download folder):

    env         print and save the machine, GPU, and library versions
    data        download + verify the pinned VizWiz files, validate annotations against the
                image zips, build the dev split and the Mac sample ID lists      (CPU only)
    mac_sample  package the ~300 pinned val images + their annotations for the Mac   (CPU only)

Later stages (added in their build-plan stage, each behind the GPU-quota guard in
lumen.budget): baseline (Stage 2), train and eval (Stage 3). The final test evaluation is sealed
(lumen.eval.test_seal) and needs --final-test-evaluation.

Only files in the download folder leave Kaggle: result JSONs, split ID lists, and the Mac sample.
Image zips stay on Kaggle.

    uv run python kaggle/run_stages.py env
    uv run python kaggle/run_stages.py data
    uv run python kaggle/run_stages.py mac_sample
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("env").set_defaults(fn=cmd_env)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    sub.add_parser("mac_sample").set_defaults(fn=cmd_mac_sample)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
