"""HFBackend on tiny stand-ins built from the pinned models' real configs and processors.

Needs the `gpu` dependency group (torch, transformers); CI installs core only, so these skip
there and run locally and on Kaggle. The first run downloads the pinned revisions' config and
tokenizer files (a few MB, no weights); a Hub rate limit skips cleanly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import skip_on_usage_cap  # noqa: E402
from PIL import Image  # noqa: E402

from lumen.pins import load_pins  # noqa: E402
from lumen.zeroshot.backends import GenerationConfig, tiny_stand_in  # noqa: E402
from lumen.zeroshot.prompts import chat_messages  # noqa: E402

PINS = load_pins()
STAND_INS = ["smolvlm2-500m", "lfm2.5-vl-450m"]


@pytest.fixture(scope="module", params=STAND_INS)
def backend(request: pytest.FixtureRequest):
    with skip_on_usage_cap("Hugging Face"):
        return tiny_stand_in(PINS.models[request.param], gen=GenerationConfig(max_new_tokens=4))


def batch(n: int = 3):
    images = [
        Image.new("RGB", size, color)
        for size, color in [
            ((640, 480), (200, 20, 20)),
            ((300, 500), (20, 200, 20)),
            ((32, 32), (0, 0, 0)),
        ]
    ][:n]
    questions = [
        "What is this?",
        "What color is the shirt on the chair next to the window?",
        "Is it blurry?",
    ][:n]
    return images, [chat_messages(q, "abstain") for q in questions]


def test_outputs_are_strings_and_probabilities(backend) -> None:
    images, msgs = batch()
    outs = backend.run_batch(images, msgs)
    assert len(outs) == 3
    for o in outs:
        assert isinstance(o.raw, str)
        assert 0.0 <= o.p_unanswerable <= 1.0


def test_batching_does_not_change_answers(backend) -> None:
    """Left padding must be handled: each question answered alone == answered in a batch."""
    images, msgs = batch()
    together = backend.run_batch(images, msgs)
    alone = [backend.run_batch([im], [m])[0] for im, m in zip(images, msgs, strict=True)]
    assert [o.raw for o in together] == [o.raw for o in alone]
    for t, a in zip(together, alone, strict=True):
        assert t.p_unanswerable == pytest.approx(a.p_unanswerable, rel=1e-3, abs=1e-9)


def test_cached_score_equals_full_forward(backend) -> None:
    """P(unanswerable) from prefill + cached continuations == a full forward per spelling."""
    images, msgs = batch(1)
    enc = backend._encode(images, msgs)
    cached = backend.p_unanswerable(enc)[0]
    total = 0.0
    with torch.inference_mode():
        for variant in backend._variants:
            ids = torch.cat([enc["input_ids"], torch.tensor([variant])], dim=1)
            mask = torch.cat(
                [enc["attention_mask"], torch.ones(1, len(variant), dtype=torch.long)], dim=1
            )
            full = {**enc, "input_ids": ids, "attention_mask": mask}
            logp = torch.log_softmax(backend.model(**full).logits.double(), dim=-1)
            start = enc["input_ids"].shape[1] - 1
            total += float(sum(logp[0, start + j, t] for j, t in enumerate(variant)).exp())
    assert cached == pytest.approx(total, rel=1e-4)


def test_describe_records_pin_and_decoding(backend) -> None:
    d = backend.describe()
    assert d["stand_in"] is True and len(d["revision"]) == 40
    assert d["generation"]["do_sample"] is False and d["generation"]["max_new_tokens"] == 4


def test_generation_prompt_is_applied(backend) -> None:
    """transformers 5.17 logs a 'Kwargs passed to processor.__call__' warning from inside
    apply_chat_template even for a bare call; this pins that the assistant turn is still opened."""
    text = backend.processor.apply_chat_template(
        chat_messages("Q?", "plain"), add_generation_prompt=True
    )
    expected = {
        "smolvlm2-500m-stand-in": "Assistant:",
        "lfm2.5-vl-450m-stand-in": "<|im_start|>assistant\n",
    }[backend.name]
    assert text.endswith(expected)


def test_only_the_known_noisy_warning_is_dropped(backend, caplog) -> None:
    import logging

    log = logging.getLogger("transformers.processing_utils")
    with caplog.at_level(logging.WARNING, logger="transformers.processing_utils"):
        log.propagate = True
        log.warning(
            "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, "
            "not in `**kwargs`"
        )
        log.warning("some other warning")
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["some other warning"]


def test_prefetch_threads_give_identical_answers(backend, tmp_path) -> None:
    """The real processor used from worker threads (one copy per thread) changes nothing."""
    import zipfile

    from conftest import fake_records

    from lumen.data.images import ZipImages
    from lumen.zeroshot.harness import read_predictions, run_predictions

    recs = fake_records("val", 9)
    colors = [(200, 20, 20), (20, 200, 20), (20, 20, 200)]
    with zipfile.ZipFile(tmp_path / "val.zip", "w") as z:
        for i, r in enumerate(recs):
            import io

            buf = io.BytesIO()
            Image.new("RGB", (640 - 40 * i, 480), colors[i % 3]).save(buf, "JPEG")
            z.writestr(f"val/{r['image']}", buf.getvalue())
    for r in recs:
        r["question_id"] = r["image"].removesuffix(".jpg")
    images = ZipImages(tmp_path / "val.zip")
    for workers, name in [(0, "serial"), (3, "prefetch")]:
        run_predictions(
            backend,
            recs,
            images,
            prompt_key="abstain",
            split="val",
            out_path=tmp_path / f"{name}.jsonl",
            batch_size=2,
            prefetch_workers=workers,
        )
    _, serial = read_predictions(tmp_path / "serial.jsonl")
    _, prefetch = read_predictions(tmp_path / "prefetch.jsonl")
    assert [r["raw"] for r in serial.values()] == [r["raw"] for r in prefetch.values()]
    for a, b in zip(serial.values(), prefetch.values(), strict=True):
        assert a["p_unanswerable"] == pytest.approx(b["p_unanswerable"], rel=1e-5, abs=1e-12)


# ----------------------------------------------------------- image resolution cap (--long-edge)


@pytest.fixture(scope="module")
def smolvlm_pin():
    return PINS.models["smolvlm2-500m"]


@pytest.mark.parametrize(
    ("long_edge", "tokens"), [(None, 832), (1536, 640), (1024, 320), (512, 64)]
)
def test_long_edge_caps_the_image_tokens(smolvlm_pin, long_edge, tokens) -> None:
    """A 3:4 portrait (most VizWiz photos): 13 / 10 / 5 / 1 tiles of 64 tokens. The estimates in
    lumen.zeroshot.estimates.PORTRAIT_TILES come from these counts."""
    from lumen.zeroshot.estimates import PORTRAIT_TILES

    with skip_on_usage_cap("Hugging Face"):
        b = tiny_stand_in(
            smolvlm_pin, gen=GenerationConfig(max_new_tokens=2), image_longest_edge=long_edge
        )
    outs = b.run_batch(
        [Image.new("RGB", (968, 1296), (90, 90, 90))], [chat_messages("Q?", "plain")]
    )
    assert len(outs) == 1
    assert b.image_tokens.as_dict() == {"questions": 1, "mean": float(tokens)}
    assert tokens == 64 * PORTRAIT_TILES[long_edge or 2048]
    assert b.describe().get("image_longest_edge") == long_edge


def test_long_edge_must_be_a_whole_number_of_tiles(smolvlm_pin) -> None:
    with skip_on_usage_cap("Hugging Face"), pytest.raises(ValueError, match="must be one of"):
        tiny_stand_in(smolvlm_pin, image_longest_edge=768)
    with skip_on_usage_cap("Hugging Face"), pytest.raises(ValueError, match="must be one of"):
        tiny_stand_in(smolvlm_pin, image_longest_edge=4096)


def test_long_edge_refused_for_other_tiling() -> None:
    with skip_on_usage_cap("Hugging Face"), pytest.raises(ValueError, match="SmolVLM2 models only"):
        tiny_stand_in(PINS.models["lfm2.5-vl-450m"], image_longest_edge=1024)
