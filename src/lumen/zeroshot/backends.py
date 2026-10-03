"""Model backends for zero-shot answering: a real Hugging Face backend and a fake one for tests.

Every backend answers a batch and returns, per question, the raw generation and P(unanswerable):
the model's probability that its answer *starts with* "unanswerable" (summed over the spellings
in UNANSWERABLE_VARIANTS). Answerability AP uses `1 - P(unanswerable)` as the answerable score,
so every model gets a graded ranking, not just a hard label.

HFBackend computes that probability exactly and cheaply: one forward pass over the prompt (with
the image) builds the KV cache, then each spelling's remaining tokens are scored from a copy of
that cache, so the image is encoded once for scoring and once for generation.
tests/test_hf_backend.py checks the cached score against a full uncached forward pass.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from PIL import Image

from lumen.pins import ModelPin

if TYPE_CHECKING:
    import torch

UNANSWERABLE_VARIANTS = ("unanswerable", "Unanswerable", " unanswerable", " Unanswerable")

# transformers 5.17 logs this from inside apply_chat_template on every call, even a bare one
# (see tests/test_hf_backend.py::test_generation_prompt_is_applied). Once per question it would add
# ~16k identical lines to a Stage 2 notebook. Only this exact message is dropped.
_NOISY = "have to be in `processor_kwargs` dict, not in `**kwargs`"


class _DropProcessorKwargsWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _NOISY not in record.getMessage()


def _quiet_processor_kwargs_warning() -> None:
    log = logging.getLogger("transformers.processing_utils")
    if not any(isinstance(f, _DropProcessorKwargsWarning) for f in log.filters):
        log.addFilter(_DropProcessorKwargsWarning())


@dataclass(frozen=True)
class GenerationConfig:
    """Fixed decoding settings, identical for every model (recorded in every result)."""

    max_new_tokens: int = 16
    do_sample: bool = False  # greedy
    num_beams: int = 1
    repetition_penalty: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchOutput:
    raw: str
    p_unanswerable: float


class Backend(Protocol):
    """`prepare` is CPU-only and thread-safe, so the harness can prepare the next batches in
    worker threads while `run_prepared` keeps the GPU busy. `run_batch` = both, in sequence."""

    name: str

    def describe(self) -> dict[str, Any]: ...

    def prepare(self, images: list[Image.Image], messages: list[list[dict]]) -> Any: ...

    def run_prepared(self, prepared: Any) -> list[BatchOutput]: ...

    def run_batch(
        self, images: list[Image.Image], messages: list[list[dict]]
    ) -> list[BatchOutput]: ...


class NonFiniteOutput(RuntimeError):
    """fp16 overflow or a broken model: logits contained NaN/Inf. Never scored silently."""


@dataclass
class ImageTokenStats:
    """Image tokens per question over the questions a backend has answered (the main cost lever:
    at SmolVLM2's default 2,048 px cap a VizWiz portrait photo is ~830 of ~875 prompt tokens)."""

    questions: int = 0
    tokens: int = 0

    def add(self, per_question: list[int]) -> None:
        self.questions += len(per_question)
        self.tokens += sum(per_question)

    def as_dict(self) -> dict[str, Any] | None:
        if not self.questions:
            return None
        return {"questions": self.questions, "mean": round(self.tokens / self.questions, 1)}


# ----------------------------------------------------------------------------- fake backend


class FakeBackend:
    """Deterministic stand-in with no ML dependencies, for pipeline tests (and CI).

    Says "unanswerable" when the question mentions 'blur', otherwise answers `answer`. Its
    P(unanswerable) is high for the blur questions and a stable hash-derived low value otherwise.
    """

    name = "fake"

    def __init__(
        self, answer: str = "dog", nan_on: str | None = None, image_longest_edge: int | None = None
    ) -> None:
        self.answer = answer
        self.nan_on = nan_on
        self.image_longest_edge = image_longest_edge
        self.image_tokens = ImageTokenStats()
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"backend": "fake", "fixed_answer": self.answer}
        if self.image_longest_edge is not None:
            d["image_longest_edge"] = self.image_longest_edge
        return d

    def prepare(self, images: list[Image.Image], messages: list[list[dict]]) -> list[str]:
        for image in images:
            assert isinstance(image, Image.Image) and image.mode == "RGB"
        return [msg[0]["content"][1]["text"] for msg in messages]

    def run_batch(self, images: list[Image.Image], messages: list[list[dict]]) -> list[BatchOutput]:
        return self.run_prepared(self.prepare(images, messages))

    def run_prepared(self, texts: list[str]) -> list[BatchOutput]:
        self.calls += 1
        self.image_tokens.add([(self.image_longest_edge or 2048) // 8] * len(texts))
        out = []
        for text in texts:
            if self.nan_on and self.nan_on in text:
                raise NonFiniteOutput("fake backend: non-finite logits")
            h = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            if "blur" in text.split("\n")[0].lower():
                out.append(BatchOutput("Unanswerable.", 0.6 + 0.4 * h))
            else:
                out.append(BatchOutput(f"Answer: {self.answer}\nbecause reasons", 0.3 * h))
        return out


# ------------------------------------------------------------------------ Hugging Face backend


class HFBackend:
    """A pinned Hugging Face image-text-to-text model (SmolVLM2, LFM2.5-VL) with transformers."""

    def __init__(
        self,
        name: str,
        model: Any,
        processor: Any,
        *,
        device: str,
        gen: GenerationConfig,
        pin: ModelPin | None = None,
        stand_in: bool = False,
        image_longest_edge: int | None = None,
    ) -> None:
        self.name = name
        self.model = model.eval()
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"  # batched generation needs left padding
        _quiet_processor_kwargs_warning()
        self.device = device
        self.gen = gen
        self.pin = pin
        self.stand_in = stand_in
        self.image_longest_edge = image_longest_edge
        self._images_kwargs = _images_kwargs(processor, image_longest_edge)
        self.image_tokens = ImageTokenStats()
        tok = processor.tokenizer
        self._variants = sorted(
            {tuple(tok(v, add_special_tokens=False).input_ids) for v in UNANSWERABLE_VARIANTS}
        )
        if any(not v for v in self._variants):
            raise ValueError("a spelling of 'unanswerable' tokenized to nothing")
        self._local = threading.local()

    def _thread_processor(self) -> Any:
        """The processor for the calling thread. Fast tokenizers can raise "Already borrowed" when
        one instance is shared across threads, so each worker thread gets its own copy."""
        if threading.current_thread() is threading.main_thread():
            return self.processor
        proc = getattr(self._local, "processor", None)
        if proc is None:
            proc = self._local.processor = copy.deepcopy(self.processor)
        return proc

    @classmethod
    def from_pin(
        cls,
        pin: ModelPin,
        *,
        device: str,
        dtype: str,
        gen: GenerationConfig,
        image_longest_edge: int | None = None,
    ) -> HFBackend:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(pin.repo, revision=pin.revision)
        model = AutoModelForImageTextToText.from_pretrained(
            pin.repo, revision=pin.revision, dtype=getattr(torch, dtype)
        ).to(device)
        return cls(
            pin.key,
            model,
            processor,
            device=device,
            gen=gen,
            pin=pin,
            image_longest_edge=image_longest_edge,
        )

    def describe(self) -> dict[str, Any]:
        import torch

        params = sum(p.numel() for p in self.model.parameters())
        d = {
            "backend": "transformers",
            "model_key": self.name,
            "repo": self.pin.repo if self.pin else None,
            "revision": self.pin.revision if self.pin else None,
            "stand_in": self.stand_in,
            "parameters": params,
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "device": self.device,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "generation": self.gen.as_dict(),
            "unanswerable_token_variants": [list(v) for v in self._variants],
        }
        if self.image_longest_edge is not None:  # absent = the processor's own default
            d["image_longest_edge"] = self.image_longest_edge
        return d

    def prepare(self, images: list[Image.Image], messages: list[list[dict]]) -> dict:
        """CPU work only (chat template, image resize/tiling, tokenization): thread-safe."""
        proc = self._thread_processor()
        texts = [proc.apply_chat_template(m, add_generation_prompt=True) for m in messages]
        return dict(
            proc(
                text=texts,
                images=[[im] for im in images],
                return_tensors="pt",
                padding=True,
                **self._images_kwargs,
            )
        )

    def _to_device(self, prepared: dict) -> dict:
        return {k: v.to(self.device) for k, v in prepared.items()}

    def _encode(self, images: list[Image.Image], messages: list[list[dict]]) -> dict:
        return self._to_device(self.prepare(images, messages))

    def _check_finite(self, logits: torch.Tensor, what: str) -> None:
        import torch

        if not torch.isfinite(logits).all():
            raise NonFiniteOutput(f"{self.name}: non-finite logits in {what} (fp16 overflow?)")

    def p_unanswerable(self, enc: dict) -> list[float]:
        """P(answer starts with any spelling of 'unanswerable'), exact, from one prefill."""
        import torch

        with torch.inference_mode():
            out = self.model(**enc, use_cache=True)
            first = out.logits[:, -1, :].float()
            self._check_finite(first, "scoring prefill")
            logp_first = torch.log_softmax(first, dim=-1)
            total = torch.zeros(first.shape[0], dtype=torch.float64, device=first.device)
            mask = enc["attention_mask"]
            for variant in self._variants:
                logp = logp_first[:, variant[0]].double()
                if len(variant) > 1:
                    cont = torch.tensor([variant[:-1]], device=mask.device).repeat(mask.shape[0], 1)
                    cont_mask = torch.cat([mask, torch.ones_like(cont)], dim=1)
                    step = self.model(
                        input_ids=cont,
                        attention_mask=cont_mask,
                        past_key_values=copy.deepcopy(out.past_key_values),
                        use_cache=True,
                    )
                    # `cont` feeds tokens 0..k-2; logits at position j predict token j+1.
                    lp = torch.log_softmax(step.logits.float(), dim=-1)
                    self._check_finite(lp, "scoring continuation")
                    for j in range(1, len(variant)):
                        logp = logp + lp[:, j - 1, variant[j]].double()
                total += logp.exp()
        return [min(1.0, float(p)) for p in total.cpu()]

    def generate(self, enc: dict) -> list[str]:
        import torch

        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=self.gen.do_sample,
                num_beams=self.gen.num_beams,
                repetition_penalty=self.gen.repetition_penalty,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1] :]
        return self.processor.batch_decode(new, skip_special_tokens=True)

    def run_batch(self, images: list[Image.Image], messages: list[list[dict]]) -> list[BatchOutput]:
        return self.run_prepared(self.prepare(images, messages))

    def run_prepared(self, prepared: dict) -> list[BatchOutput]:
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            self.image_tokens.add((prepared["input_ids"] == image_token_id).sum(dim=1).tolist())
        enc = self._to_device(prepared)
        probs = self.p_unanswerable(enc)
        raws = self.generate(enc)
        for p in probs:
            if not math.isfinite(p):
                raise NonFiniteOutput(f"{self.name}: non-finite P(unanswerable)")
        return [BatchOutput(r, p) for r, p in zip(raws, probs, strict=True)]


def _images_kwargs(processor: Any, longest_edge: int | None) -> dict[str, Any]:
    """Processor kwargs that cap the image's long edge, validated against the processor.

    SmolVLM resizes the long edge up to a multiple of its 512 px tile, then splits it into tiles
    plus a global view, so only multiples of the tile size are distinct settings (768 gives the
    same tiles as 1,024). Other model families tile differently and are not supported.
    """
    if longest_edge is None:
        return {}
    ip = processor.image_processor
    tile = (getattr(ip, "max_image_size", None) or {}).get("longest_edge")
    default = getattr(getattr(ip, "size", None), "longest_edge", None)
    if default is None and isinstance(getattr(ip, "size", None), dict):
        default = ip.size.get("longest_edge")
    if tile is None or default is None:
        raise ValueError(
            f"{type(ip).__name__} has no SmolVLM-style long-edge tiling; "
            "--long-edge applies to SmolVLM2 models only"
        )
    if longest_edge % tile or not tile <= longest_edge <= default:
        caps = list(range(tile, default + 1, tile))
        raise ValueError(f"long edge {longest_edge} px: must be one of {caps} (tile {tile} px)")
    return {"images_kwargs": {"size": {"longest_edge": longest_edge}}}


# --------------------------------------------------------------------------- tiny stand-in


def tiny_stand_in(
    pin: ModelPin,
    *,
    device: str = "cpu",
    gen: GenerationConfig | None = None,
    seed: int = 0,
    image_longest_edge: int | None = None,
) -> HFBackend:
    """The pinned model's real processor and chat template on a tiny random-weight model.

    Only config and tokenizer/processor files are downloaded (a few MB, at the pinned revision);
    the model is built from the real config with every size shrunk, so it runs on a laptop CPU in
    seconds while exercising the same processor, image tiling, chat template, padding, and
    generation code as the real model. Used to prove the pipeline before spending GPU time.
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    cfg = AutoConfig.from_pretrained(pin.repo, revision=pin.revision)
    text, vision = cfg.text_config, cfg.vision_config
    for k, v in {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "head_dim": 16,
        "block_ff_dim": 128,
    }.items():
        if hasattr(text, k):
            setattr(text, k, v)
    if getattr(text, "layer_types", None):  # hybrid LFM2: keep one of each layer kind
        text.layer_types = ["conv", "full_attention"]
    for k, v in {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
    }.items():
        setattr(vision, k, v)
    if hasattr(cfg, "projector_hidden_size"):
        cfg.projector_hidden_size = 64
    torch.manual_seed(seed)
    model = AutoModelForImageTextToText.from_config(cfg).to(device)
    processor = AutoProcessor.from_pretrained(pin.repo, revision=pin.revision)
    return HFBackend(
        f"{pin.key}-stand-in",
        model,
        processor,
        device=device,
        gen=gen or GenerationConfig(),
        pin=pin,
        stand_in=True,
        image_longest_edge=image_longest_edge,
    )
