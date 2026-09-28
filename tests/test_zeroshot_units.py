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
