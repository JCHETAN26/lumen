"""The training loop (lumen.train.loop) on a tiny random-weight SmolVLM2 stand-in, on CPU.

Checks LoRA placement, that loss goes down, exact resume, the NaN/Inf guard, and a clean stop from
the quota callback. Needs the `gpu` dependency group (torch, transformers, peft); downloads only
config and processor files at the pinned revision.
"""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from conftest import skip_on_usage_cap  # noqa: E402
from PIL import Image  # noqa: E402

from lumen.pins import load_pins  # noqa: E402
from lumen.train.data import Example, TrainCollator  # noqa: E402
from lumen.train.loop import (  # noqa: E402
    StopTraining,
    TrainConfig,
    attach_lora,
    index_stream,
    latest_checkpoint,
    train,
    trainable_state,
)
from lumen.zeroshot.backends import tiny_stand_in  # noqa: E402

PIN = load_pins().models["smolvlm2-500m"]
CFG = dict(
    image_longest_edge=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.0,  # deterministic, so resumed and uninterrupted runs match exactly
    lr=5e-3,
    warmup_steps=2,
    schedule_steps=100,
    micro_batch=2,
    grad_accum=2,
    fp16=False,  # CPU
    gradient_checkpointing=True,
    nonfinite_patience=3,
    checkpoint_every=2,
    dataloader_workers=0,
)


def cfg(**kw) -> TrainConfig:
    return TrainConfig(**(CFG | kw))


DATA = [
    (Image.new("RGB", (48, 64), (200, 20, 20)), Example("q1", "What color?", "red", "a")),
    (Image.new("RGB", (64, 48), (20, 200, 20)), Example("q2", "What color?", "green", "b")),
    (Image.new("RGB", (40, 40), (20, 20, 200)), Example("q3", "What color?", "blue", "c")),
    (Image.new("RGB", (32, 32), (0, 0, 0)), Example("q4", "Is it blurry?", "unanswerable", "d")),
]


def fresh(c: TrainConfig):
    with skip_on_usage_cap("Hugging Face"):
        b = tiny_stand_in(PIN, seed=0)
    model = attach_lora(b.model, c)
    return model, TrainCollator(b.processor, c.prompt_key, image_longest_edge=c.image_longest_edge)


def run(c: TrainConfig, steps: int, ckpt, **kw):
    model, col = fresh(c)
    rep = train(model, DATA, col, c, max_steps=steps, checkpoint_dir=ckpt, device="cpu", **kw)
    return model, rep


def test_index_stream_resumes_exactly() -> None:
    full = list(itertools.islice(index_stream(10, 7), 25))
    assert sorted(full[:10]) == list(range(10)) and full[:10] != full[10:20]  # new epoch order
    assert list(itertools.islice(index_stream(10, 7, start=13), 12)) == full[13:]


def test_lora_only_on_the_language_model_in_fp32() -> None:
    model, _ = fresh(cfg())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("text_model" in n and "lora_" in n for n, _ in trainable)
    assert all(p.dtype == torch.float32 for _, p in trainable)
    assert not any(p.requires_grad for n, p in model.named_parameters() if "vision_model" in n)


def test_loss_goes_down(tmp_path) -> None:
    _, rep = run(cfg(), 30, tmp_path)
    assert rep.status == "completed" and rep.final_step == 30
    # A random tiny LM with frozen embeddings/head starts near ln(vocab) ~ 10.8 and can only move so
    # far with rank-8 LoRA; a clear, steady drop is what shows the gradients reach the adapters.
    assert sum(rep.losses[-5:]) / 5 < sum(rep.losses[:5]) / 5 - 0.5
    assert rep.target_tokens > 0 and rep.examples_seen == 30 * 4


def test_resume_matches_an_uninterrupted_run(tmp_path) -> None:
    straight, _ = run(cfg(), 6, tmp_path / "a")
    run(cfg(), 4, tmp_path / "b")
    resumed, rep = run(cfg(), 6, tmp_path / "b")
    assert rep.start_step == 4 and rep.steps_done == 2
    a, b = trainable_state(straight), trainable_state(resumed)
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-6)
    assert latest_checkpoint(tmp_path / "b").name == "step-0000006"
    assert len(list((tmp_path / "b").glob("step-*"))) == 2  # keep_checkpoints


def nan_from(model, step_of_call, bad: set[int]):
    """Make the loss NaN on the given forward calls (counted from 0)."""
    real = model.forward
    calls = itertools.count()

    def forward(*a, **kw):
        out = real(*a, **kw)
        if next(calls) in bad:
            out.loss = out.loss * float("nan")
        return out

    model.forward = forward


def test_isolated_nan_is_skipped_and_training_continues(tmp_path) -> None:
    c = cfg()
    model, col = fresh(c)
    nan_from(model, None, {6})  # forward call 6 = step 3's first micro-batch
    rep = train(model, DATA, col, c, max_steps=6, checkpoint_dir=tmp_path, device="cpu")
    assert rep.status == "completed" and rep.nonfinite_losses == [3]
    assert len(rep.losses) == 5 and latest_checkpoint(tmp_path).name == "step-0000006"


def test_repeated_nan_stops_cleanly_keeping_the_last_good_checkpoint(tmp_path) -> None:
    c = cfg()
    model, col = fresh(c)
    nan_from(model, None, set(range(8, 100)))  # every forward from step 4 on
    rep = train(model, DATA, col, c, max_steps=20, checkpoint_dir=tmp_path, device="cpu")
    assert rep.status == "stopped_non_finite" and rep.nonfinite_losses == [4, 5, 6]
    assert rep.final_step == 7
    assert latest_checkpoint(tmp_path).name == "step-0000004"
    assert rep.last_good_checkpoint.endswith("step-0000004")


def test_on_step_can_stop_cleanly_with_a_checkpoint(tmp_path) -> None:
    def quota(step, report):
        if step == 3:
            raise StopTraining("budget")

    _, rep = run(cfg(), 10, tmp_path, on_step=quota)
    assert rep.status == "stopped:budget" and rep.final_step == 3
    assert latest_checkpoint(tmp_path).name == "step-0000003"
    _, again = run(cfg(), 5, tmp_path)
    assert again.start_step == 3 and again.final_step == 5
