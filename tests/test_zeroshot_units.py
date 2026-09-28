from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lumen.data.images import ZipImages
from lumen.eval.bootstrap import bootstrap_ci, mean_ci
from lumen.zeroshot.backends import FakeBackend
from lumen.zeroshot.extract import extract_answer
from lumen.zeroshot.harness import HeaderMismatch, read_predictions, run_predictions
from lumen.zeroshot.prompts import PROMPTS, chat_messages, prompt_text


def test_prompts_fix_structure() -> None:
    assert set(PROMPTS) == {"plain", "abstain"}
    msg = chat_messages("  what   is\nthis? ", "abstain")
    assert msg[0]["content"][0] == {"type": "image"}
    text = msg[0]["content"][1]["text"]
    assert text.startswith("what is this?\n") and text.endswith("answer exactly: unanswerable")
    with pytest.raises(KeyError):
        prompt_text("q", "creative")
    with pytest.raises(ValueError):
        prompt_text("   ", "plain")


@pytest.mark.parametrize(
    ("raw", "answer", "by_rule"),
    [
        ("dog", "dog", False),
        (" Answer: Red.\nIt looks red.", "Red.", False),
        ("Assistant: 3", "3", False),
        ('"Coca Cola"', "Coca Cola", False),
        ("\n\nyes", "yes", False),
        ("unanswerable", "unanswerable", False),
        ("Unanswerable.", "unanswerable", False),
        ("I can't tell from this image.", "unanswerable", True),
        ("Cannot be determined", "unanswerable", True),
        ("It is not possible to tell", "unanswerable", True),
        ("The image is too blurry", "unanswerable", True),
        ("sorry, I cannot read the label", "unanswerable", True),
        ("not red", "not red", False),  # contains a negation, but is an answer
        ("a-frame", "a-frame", False),
        ("unknown", "unknown", False),  # could be a real answer: not mapped
        ("", "", False),
    ],
)
def test_extract_answer(raw: str, answer: str, by_rule: bool) -> None:
    ex = extract_answer(raw)
    assert (ex.answer, ex.abstained_by_rule) == (answer, by_rule)


def test_bootstrap_ci_contains_estimate_and_is_reproducible() -> None:
    vals = np.r_[np.ones(70), np.zeros(30)]
    ci = mean_ci(vals, resamples=500)
    assert ci["low"] < 0.7 < ci["high"] and ci["high"] - ci["low"] < 0.25
    assert mean_ci(vals, resamples=500) == ci


def test_bootstrap_skips_undefined_resamples() -> None:
    def stat(idx: np.ndarray) -> float:
        if len(set(idx % 2)) < 2:
            raise ValueError("one class")
        return 1.0

    ci = bootstrap_ci(2, stat, resamples=200)
    assert ci["skipped"] > 0 and ci["low"] == ci["high"] == 1.0


@pytest.fixture
def small(tmp_path: Path):
    import zipfile

    from conftest import fake_records, jpeg

    recs = fake_records("val", 6)
    with zipfile.ZipFile(tmp_path / "val.zip", "w") as z:
        for r in recs:
            z.writestr(f"val/{r['image']}", jpeg())
    for r in recs:
        r["question_id"] = r["image"].removesuffix(".jpg")
    return recs, ZipImages(tmp_path / "val.zip"), tmp_path / "p.jsonl"


def test_resume_skips_answered_and_survives_a_torn_line(small) -> None:
    recs, images, path = small
    fb = FakeBackend()
    run_predictions(
        fb, recs[:3], images, prompt_key="plain", split="val", out_path=path, batch_size=2
    )
    with open(path, "a") as f:
        f.write('{"question_id": "VizWiz_val_000')  # session killed mid-write
    preds, stats = run_predictions(
        fb, recs, images, prompt_key="plain", split="val", out_path=path, batch_size=2
    )
    assert stats.resumed == 3 and stats.new == 3 and len(preds) == 6
    header, rows = read_predictions(path)
    assert header["prompt_key"] == "plain" and len(rows) == 6
    assert all(json.loads(line) for line in path.read_text().splitlines())  # valid JSONL again


def test_different_settings_never_mix(small) -> None:
    recs, images, path = small
    run_predictions(
        FakeBackend(), recs, images, prompt_key="plain", split="val", out_path=path, batch_size=4
    )
    with pytest.raises(HeaderMismatch):
        run_predictions(
            FakeBackend(),
            recs,
            images,
            prompt_key="abstain",
            split="val",
            out_path=path,
            batch_size=4,
        )
    with pytest.raises(HeaderMismatch, match="outside this split"):
        run_predictions(
            FakeBackend(),
            recs[:2],
            images,
            prompt_key="plain",
            split="val",
            out_path=path,
            batch_size=4,
        )


def test_fake_backend_sees_rgb_images(small) -> None:
    _, images, _ = small
    assert images.open(images.names()[0]).mode == "RGB"
    FakeBackend().run_batch([Image.new("RGB", (2, 2))], [chat_messages("q?", "plain")])


# ------------------------------------------------------------------ prefetching (thread pool)


@pytest.fixture
def many(tmp_path: Path):
    import zipfile

    from conftest import fake_records, jpeg

    recs = fake_records("val", 23)
    with zipfile.ZipFile(tmp_path / "val.zip", "w") as z:
        for r in recs:
            z.writestr(f"val/{r['image']}", jpeg())
    for r in recs:
        r["question_id"] = r["image"].removesuffix(".jpg")
    return recs, ZipImages(tmp_path / "val.zip"), tmp_path


def _run(recs, images, path, workers, backend=None, **kw):
    return run_predictions(
        backend or FakeBackend(),
        recs,
        images,
        prompt_key="plain",
        split="val",
        out_path=path,
        batch_size=3,
        prefetch_workers=workers,
        **kw,
    )


def test_prefetch_output_is_byte_identical(many) -> None:
    recs, images, tmp = many
    _run(recs, images, tmp / "serial.jsonl", 0)
    _run(recs, images, tmp / "prefetch.jsonl", 3)
    assert (tmp / "serial.jsonl").read_bytes() == (tmp / "prefetch.jsonl").read_bytes()


class SlowFirstBackend(FakeBackend):
    """Preparation of earlier batches takes longer, so workers finish out of order."""

    def prepare(self, images, messages):
        import time

        texts = super().prepare(images, messages)
        if any("00000000" in t or "00000001" in t for t in texts):
            time.sleep(0.2)
        return texts


def test_prefetch_preserves_order_when_workers_finish_out_of_order(many) -> None:
    recs, images, tmp = many
    _run(recs, images, tmp / "p.jsonl", 3, backend=SlowFirstBackend())
    _, rows = read_predictions(tmp / "p.jsonl")
    assert list(rows) == [r["question_id"] for r in recs]


class FailingPrepare(FakeBackend):
    def prepare(self, images, messages):
        texts = super().prepare(images, messages)
        if any("broken" in t for t in texts):
            raise OSError("corrupt image")
        return texts


def test_worker_error_surfaces_at_its_batch(many) -> None:
    import threading

    recs, images, tmp = many
    recs = [dict(r) for r in recs]
    recs[10]["question"] = "broken?"  # batch 3 (0-based) of size 3
    with pytest.raises(OSError, match="corrupt image"):
        _run(recs, images, tmp / "p.jsonl", 3, backend=FailingPrepare())
    _, rows = read_predictions(tmp / "p.jsonl")
    assert len(rows) == 9  # the three batches before it were answered and saved
    assert not [t for t in threading.enumerate() if t.name.startswith("lumen-prefetch")]


def test_early_stop_shuts_the_pool_down(many) -> None:
    import threading

    recs, images, tmp = many

    def stop(done: int, seconds: float) -> None:
        if done >= 6:
            raise RuntimeError("budget")

    with pytest.raises(RuntimeError, match="budget"):
        _run(recs, images, tmp / "p.jsonl", 3, on_batch=stop)
    assert len(read_predictions(tmp / "p.jsonl")[1]) == 6
    assert not [t for t in threading.enumerate() if t.name.startswith("lumen-prefetch")]
    preds, stats = _run(recs, images, tmp / "p.jsonl", 3)  # resume completes the rest
    assert stats.resumed == 6 and len(preds) == len(recs)


def test_zip_images_concurrent_reads(many) -> None:
    from concurrent.futures import ThreadPoolExecutor

    _, images, _ = many
    names = images.names() * 4
    expected = {n: images.read_bytes(n) for n in images.names()}
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(images.read_bytes, names))
    assert all(g == expected[n] for g, n in zip(got, names, strict=True))


def test_negative_workers_rejected(many) -> None:
    recs, images, tmp = many
    with pytest.raises(ValueError):
        _run(recs, images, tmp / "p.jsonl", -1)
