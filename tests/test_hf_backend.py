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
