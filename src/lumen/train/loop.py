"""LoRA training loop for SmolVLM2 on a T4: fp16 with loss scaling, a NaN/Inf guard, checkpoints.

- LoRA (PEFT) on the language model's attention and MLP projections only (`LORA_TARGETS`); the
  vision encoder and connector stay frozen. The frozen base runs in fp16 (the T4 has no bf16);
  LoRA weights are fp32 (GradScaler can't unscale fp16 gradients) and run under fp16 autocast.
- fp16 safety: `torch.amp.GradScaler` (dynamic loss scaling) skips steps whose gradients overflow.
  A non-finite *loss* skips the step too; `nonfinite_patience` consecutive non-finite losses stop
  training cleanly with status "stopped_non_finite". Checkpoints are written only after good
  steps, so the newest checkpoint is always the last good one, and the stop is in the report.
- Resumable: each checkpoint holds the LoRA weights, optimizer, scaler, scheduler, step, and RNG
  states; the data order is a pure function of (seed, epoch), so a resumed run sees exactly the
  examples an uninterrupted run would.
- Gradient accumulation averages the micro-batches' per-token mean losses (each micro-batch's loss
  is weighted equally, not by its token count).
"""

from __future__ import annotations

import math
import random
import shutil
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The language model's attention + MLP projections. The vision encoder has q/k/v_proj too, so
# the pattern is anchored on text_model.
LORA_TARGETS = r".*\.text_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
TRAINER_STATE = "trainer_state.pt"


@dataclass
class TrainConfig:
    """Everything that determines a training run; recorded in every result. Choices on dev."""

    prompt_key: str = "abstain"
    image_longest_edge: int | None = None
    include_unanswerable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: str = LORA_TARGETS
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 50
    schedule_steps: int = 1000  # length of the LR schedule (linear decay to 0 after warmup)
    micro_batch: int = 4
    grad_accum: int = 2
    max_grad_norm: float = 1.0
    fp16: bool = True  # autocast + GradScaler; only on CUDA
    gradient_checkpointing: bool = True
    nonfinite_patience: int = 10
    checkpoint_every: int = 100
    keep_checkpoints: int = 2
    seed: int = 20260928
    dataloader_workers: int = 2

    @property
    def batch_size(self) -> int:
        return self.micro_batch * self.grad_accum

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "effective_batch_size": self.batch_size}


class StopTraining(Exception):
    """Raised by an `on_step` callback (e.g. the GPU-quota guard) to stop after a checkpoint."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class TrainReport:
    status: str = "completed"  # completed | stopped_non_finite | stopped:<reason>
    start_step: int = 0
    steps_done: int = 0  # optimizer steps taken in this process (good or skipped)
    final_step: int = 0
    examples_seen: int = 0
    nonfinite_losses: list[int] = field(default_factory=list)  # steps whose loss was NaN/Inf
    overflow_skips: int = 0  # steps GradScaler skipped for inf/NaN gradients
    losses: list[float] = field(default_factory=list)  # mean loss per good step
    step_seconds: list[float] = field(default_factory=list)
    tokens: int = 0  # non-padding tokens processed (prompt + image + answer)
    target_tokens: int = 0  # tokens carrying loss
    peak_memory_gib: float | None = None
    last_good_checkpoint: str | None = None
    final_loss_scale: float | None = None

    def summary(self, warmup_exclude: int = 5) -> dict[str, Any]:
        steady = self.step_seconds[warmup_exclude:] or self.step_seconds
        total_s = sum(self.step_seconds)
        return {
            **{k: v for k, v in asdict(self).items() if k not in ("losses", "step_seconds")},
            "loss_first": _mean(self.losses[:20]),
            "loss_last": _mean(self.losses[-20:]),
            "losses_every_10": [round(x, 4) for x in self.losses[::10]],
            "sec_per_step_median": statistics.median(steady) if steady else None,
            "sec_per_step_p90": float(np.percentile(steady, 90)) if steady else None,
            "steady_state_excludes_first_steps": warmup_exclude,
            "tokens_per_second": self.tokens / total_s if total_s else None,
        }


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


# ------------------------------------------------------------------------------------ model


def attach_lora(model: Any, cfg: TrainConfig) -> Any:
    """Wrap `model` with LoRA adapters on the language model; only those weights train (fp32)."""
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            bias="none",
        ),
    )
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"LoRA matched no modules with {cfg.lora_targets!r}")
    stray = [n for n, _ in trainable if "text_model" not in n]
    if stray:
        raise RuntimeError(f"LoRA reached outside the language model: {stray[:3]}")
    for _, p in trainable:
        p.data = p.data.to(torch.float32)
    if cfg.gradient_checkpointing:
        peft_model.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(peft_model.config, "use_cache"):
        peft_model.config.use_cache = False
    return peft_model


def trainable_state(model: Any) -> dict[str, Any]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def parameter_counts(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train}


# ------------------------------------------------------------------------------- data order


def index_stream(n: int, seed: int, start: int = 0) -> Iterator[int]:
    """Example indices in training order from position `start`: one fresh permutation per epoch,
    a pure function of (seed, epoch), so a resumed run continues exactly where it stopped."""
    if n <= 0:
        raise ValueError("no examples")
    epoch, offset = divmod(start, n)
    while True:
        order = np.random.default_rng([seed, epoch]).permutation(n)
        yield from (int(i) for i in order[offset:])
        epoch, offset = epoch + 1, 0


# ------------------------------------------------------------------------------ checkpoints


def latest_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    done = sorted(d for d in directory.glob("step-*") if (d / TRAINER_STATE).exists())
    return done[-1] if done else None


def _save_checkpoint(
    directory: Path, step: int, model: Any, opt: Any, scaler: Any, sched: Any, keep: int
) -> Path:
    import torch

    final = directory / f"step-{step:07d}"
    tmp = directory / f".tmp-step-{step:07d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")  # the usable PEFT adapter
    torch.save(
        {
            "step": step,
            "trainable": trainable_state(model),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp / TRAINER_STATE,
    )
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)  # a checkpoint either exists completely or not at all
    old = sorted(d for d in directory.glob("step-*") if d != final)
    for d in old[: max(0, len(old) - (keep - 1))]:
        shutil.rmtree(d)
    return final


def _load_checkpoint(path: Path, model: Any, opt: Any, scaler: Any, sched: Any) -> int:
    import torch

    state = torch.load(path / TRAINER_STATE, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(state["trainable"]) != want:
        raise RuntimeError(f"{path}: checkpoint's trainable weights don't match this model")
    with torch.no_grad():
        for n, t in state["trainable"].items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    opt.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    sched.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(state["step"])


# -------------------------------------------------------------------------------- training


def train(
    model: Any,
    dataset: Any,
    collate: Callable[[list], dict],
    cfg: TrainConfig,
    *,
    max_steps: int,
    checkpoint_dir: Path,
    device: str,
    on_step: Callable[[int, TrainReport], None] | None = None,
) -> TrainReport:
    """Train `model` (already wrapped by `attach_lora`) until optimizer step `max_steps`.

    Resumes from the newest checkpoint in `checkpoint_dir` if there is one. `on_step(step, report)`
    runs after every optimizer step and may raise StopTraining to stop cleanly.
    """
    import torch
    from torch.utils.data import DataLoader

    use_amp = cfg.fp16 and device.startswith("cuda")
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_factor(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        rest = max(1, cfg.schedule_steps - cfg.warmup_steps)
        return max(0.0, 1.0 - (step - cfg.warmup_steps) / rest)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    report = TrainReport()
    ckpt = latest_checkpoint(checkpoint_dir)
    step = _load_checkpoint(ckpt, model, opt, scaler, sched) if ckpt is not None else 0
    if ckpt is None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    report.start_step = report.final_step = step
    report.last_good_checkpoint = str(ckpt) if ckpt else None
    if step >= max_steps:
        return report

    per_step = cfg.micro_batch * cfg.grad_accum
    order = index_stream(len(dataset), cfg.seed, start=step * per_step)
    sampler = [next(order) for _ in range((max_steps - step) * per_step)]
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch,
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.dataloader_workers,
        persistent_workers=False,
    )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    batches = iter(loader)
    consecutive_bad = 0
    try:
        while step < max_steps:
            t0 = time.monotonic()
            micro_losses, bad = [], False
            for _ in range(cfg.grad_accum):
                batch = next(batches)
                report.tokens += int(batch["attention_mask"].sum())
                report.target_tokens += int((batch["labels"] != -100).sum())
                report.examples_seen += int(batch["input_ids"].shape[0])
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    bad = True  # keep consuming this step's micro-batches, never train on them
                    continue
                if not bad:
                    scaler.scale(loss / cfg.grad_accum).backward()
                    micro_losses.append(float(loss.detach()))
            if bad:
                opt.zero_grad(set_to_none=True)
                report.nonfinite_losses.append(step)
                consecutive_bad += 1
            else:
                consecutive_bad = 0
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(opt)  # skipped by the scaler if gradients overflowed
                scaler.update()
                if use_amp and scaler.get_scale() < scale_before:
                    report.overflow_skips += 1
                opt.zero_grad(set_to_none=True)
                report.losses.append(sum(micro_losses) / len(micro_losses))
            sched.step()
            step += 1
            report.steps_done += 1
            report.final_step = step
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            report.step_seconds.append(time.monotonic() - t0)
            if consecutive_bad >= cfg.nonfinite_patience:
                report.status = "stopped_non_finite"
                break
            if not bad and (step % cfg.checkpoint_every == 0 or step == max_steps):
                path = _save_checkpoint(
                    checkpoint_dir, step, model, opt, scaler, sched, cfg.keep_checkpoints
                )
                report.last_good_checkpoint = str(path)
            if on_step is not None:
                on_step(step, report)
    except StopTraining as stop:
        report.status = f"stopped:{stop.reason}"
        if consecutive_bad == 0 and report.last_good_checkpoint != str(
            checkpoint_dir / f"step-{step:07d}"
        ):
            path = _save_checkpoint(
                checkpoint_dir, step, model, opt, scaler, sched, cfg.keep_checkpoints
            )
            report.last_good_checkpoint = str(path)
    if device.startswith("cuda"):
        report.peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
    report.final_loss_scale = float(scaler.get_scale()) if use_amp else None
    if report.losses and not all(math.isfinite(x) for x in report.losses):
        raise AssertionError("a non-finite loss was recorded as a good step")
    return report
