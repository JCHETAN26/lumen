"""Training batches on SmolVLM2-500M's real processor: the loss sees the answer and nothing else.

Needs the `gpu` dependency group (torch, transformers); downloads only processor/tokenizer files at
the pinned revision. Skips cleanly in CI and on a Hub rate limit.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import skip_on_usage_cap  # noqa: E402
from PIL import Image  # noqa: E402

from lumen.pins import load_pins  # noqa: E402
from lumen.train.data import IGNORE, Example, LabelAlignmentError, TrainCollator  # noqa: E402
from lumen.zeroshot.prompts import chat_messages  # noqa: E402


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    pin = load_pins().models["smolvlm2-500m"]
    with skip_on_usage_cap("Hugging Face"):
        return AutoProcessor.from_pretrained(pin.repo, revision=pin.revision)


def batch():
    return [
        (Image.new("RGB", (480, 640), (200, 20, 20)), Example("q1", "What is this?", "dog", "a")),
        (
            Image.new("RGB", (640, 480), (20, 200, 20)),
            Example("q2", "What color is the shirt on the chair?", "dark blue", "b"),
        ),
        (
            Image.new("RGB", (32, 32), (0, 0, 0)),
            Example("q3", "Is it blurry?", "unanswerable", "c"),
        ),
    ]


def test_labels_are_exactly_the_answer_and_end_token(processor) -> None:
    col = TrainCollator(processor, "abstain", image_longest_edge=512)
    items = batch()
    out = col(items)
    tok = processor.tokenizer
    for row, (_, ex) in enumerate(items):
        lab = out["labels"][row]
        kept = lab[lab != IGNORE].tolist()
        assert kept[-1] == tok.eos_token_id
        assert tok.decode(kept[:-1]).strip() == ex.target
        assert out["attention_mask"][row][lab != IGNORE].all()  # never on padding


def test_answer_starts_where_generation_starts(processor) -> None:
    """The labeled span begins right after the prompt as encoded for generation at inference."""
    col = TrainCollator(processor, "abstain", image_longest_edge=512)
    items = batch()
    out = col(items)
    for row, (im, ex) in enumerate(items):
        prompt = processor.apply_chat_template(
            chat_messages(ex.question, "abstain"), add_generation_prompt=True
        )
        alone = processor(text=[prompt], images=[[im]], return_tensors="pt", **col.image_kwargs)
        n_prompt = alone["input_ids"].shape[1]
        first = int((out["labels"][row] != IGNORE).nonzero()[0])
        assert first == n_prompt
        assert out["input_ids"][row, :n_prompt].tolist() == alone["input_ids"][0].tolist()


def test_right_padding_and_shared_processor_untouched(processor) -> None:
    side = processor.tokenizer.padding_side
    out = TrainCollator(processor, "plain", image_longest_edge=512)(batch())
    assert processor.tokenizer.padding_side == side
    mask = out["attention_mask"]
    assert (mask[:, 0] == 1).all()  # right padding: every row starts with real tokens
    assert mask.shape == out["labels"].shape == out["input_ids"].shape


def test_long_edge_reaches_the_processor(processor) -> None:
    small = TrainCollator(processor, "plain", image_longest_edge=512)(batch()[:1])
    full = TrainCollator(processor, "plain")(batch()[:1])
    assert small["pixel_values"].shape[1] < full["pixel_values"].shape[1]


def test_missing_marker_fails_loudly(processor) -> None:
    col = TrainCollator(processor, "plain", image_longest_edge=512)
    with pytest.raises(LabelAlignmentError, match="not found"):
        col.answer_start([1, 2, 3])
