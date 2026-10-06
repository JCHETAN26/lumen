"""SmolVLM2 on Apple silicon with mlx-vlm, behind the same Backend interface as the Kaggle runs.

`lumen.zeroshot.harness.run_predictions` drives it unchanged, so the Mac answers with the same
prompt (the chat template applied to the same messages), the same greedy decoding and 16-token
cap, the same answer extractor, and the same P(unanswerable): one prefill fills the KV cache,
then each spelling's remaining tokens are scored and trimmed off again. `self_test` checks that
cached score against an uncached forward pass, and the prompt's token IDs against the
transformers processor the Kaggle runs used, before any benchmark is trusted.

Two differences from Kaggle, measured rather than assumed: mlx-vlm's SmolVLM processor uses the
slow (PIL) image path, and quantization leaves the vision encoder in fp16. So quantization's cost
is measured against MLX fp16 on the Mac; Kaggle fp16 on the same questions is a reference.

Batch size is 1: the app answers one photo at a time.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from PIL import Image

from lumen.zeroshot.backends import (
    UNANSWERABLE_VARIANTS,
    BatchOutput,
    GenerationConfig,
    ImageTokenStats,
    NonFiniteOutput,
)


def set_long_edge(processor: Any, long_edge: int | None) -> None:
    """mlx-vlm calls the image processor with no size kwargs, so the cap is set on it directly.
    Same validation as the Kaggle side: a multiple of the 512 px tile, up to the default."""
    if long_edge is None:
        return
    ip = processor.image_processor
    tile = (getattr(ip, "max_image_size", None) or {}).get("longest_edge")
    default = (getattr(ip, "size", None) or {}).get("longest_edge")
    if tile is None or default is None:
        raise ValueError(f"{type(ip).__name__} has no SmolVLM-style long-edge tiling")
    if long_edge % tile or not tile <= long_edge <= default:
        raise ValueError(f"long edge {long_edge} px must be a multiple of {tile} up to {default}")
    ip.size = {**ip.size, "longest_edge": long_edge}


class MLXBackend:
    name = "mlx"

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        describe: dict[str, Any],
        gen: GenerationConfig | None = None,
        image_longest_edge: int | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.gen = gen or GenerationConfig()
        self.image_longest_edge = image_longest_edge
        set_long_edge(processor, image_longest_edge)
        self._describe = describe
        self.image_tokens = ImageTokenStats()
        tok = processor.tokenizer
        self._variants = sorted(
            {tuple(tok(v, add_special_tokens=False).input_ids) for v in UNANSWERABLE_VARIANTS}
        )
        if any(not v for v in self._variants):
            raise ValueError("a spelling of 'unanswerable' tokenized to nothing")
        self.image_token_id = getattr(model.config, "image_token_index", None)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> MLXBackend:
        import mlx.core as mx
        from mlx_vlm import load

        model, processor = load(str(path))
        mx.eval(model.parameters())
        return cls(model, processor, **kwargs)

    def describe(self) -> dict[str, Any]:
        d = {
            **self._describe,
            "backend": "mlx",
            "stand_in": False,
            "generation": self.gen.as_dict(),
            "unanswerable_token_variants": [list(v) for v in self._variants],
        }
        if self.image_longest_edge is not None:
            d["image_longest_edge"] = self.image_longest_edge
        return d

    # ------------------------------------------------------------------------------ inputs

    def prompt(self, messages: list[dict]) -> str:
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def encode(self, image: Image.Image, messages: list[dict]) -> dict[str, Any]:
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        inputs = prepare_inputs(
            self.processor,
            images=[image],
            prompts=[self.prompt(messages)],
            image_token_index=self.image_token_id,
            add_special_tokens=should_add_special_tokens(
                self.model.config.model_type, self.processor
            ),
        )
        return dict(inputs)

    def prepare(self, images: list[Image.Image], messages: list[list[dict]]) -> list[dict]:
        return [self.encode(im, m) for im, m in zip(images, messages, strict=True)]

    @staticmethod
    def _split(enc: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, Any]]:
        extra = {
            k: v for k, v in enc.items() if k not in ("input_ids", "pixel_values", "attention_mask")
        }
        return enc["input_ids"], enc.get("pixel_values"), enc.get("attention_mask"), extra

    # ----------------------------------------------------------------------------- scoring

    def p_unanswerable(self, enc: dict[str, Any]) -> float:
        """P(answer starts with any spelling of 'unanswerable'), from one prefill + KV cache."""
        import mlx.core as mx
        from mlx_vlm.models import cache as mcache

        ids, pixels, _, extra = self._split(enc)
        lm = self.model.language_model
        cache = mcache.make_prompt_cache(lm)
        emb = self.model.get_input_embeddings(ids, pixels, **extra).inputs_embeds
        first = lm(ids, inputs_embeds=emb, cache=cache).logits[0, -1].astype(mx.float32)
        logp_first = first - mx.logsumexp(first)
        total = 0.0
        for variant in self._variants:
            logp = logp_first[variant[0]]
            if len(variant) > 1:
                cont = mx.array([list(variant[:-1])])
                lp = lm(cont, cache=cache).logits[0].astype(mx.float32)
                lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
                for j in range(1, len(variant)):
                    logp = logp + lp[j - 1, variant[j]]
                for c in cache:
                    c.trim(len(variant) - 1)
            total += math.exp(float(logp))
        if not math.isfinite(total):
            raise NonFiniteOutput("mlx: non-finite P(unanswerable)")
        return min(1.0, total)

    def p_unanswerable_uncached(self, enc: dict[str, Any]) -> float:
        """The same probability from full forward passes with no cache (self-test only)."""
        import mlx.core as mx

        ids, pixels, _, extra = self._split(enc)
        total = 0.0
        for variant in self._variants:
            full = mx.concatenate([ids, mx.array([list(variant[:-1])], dtype=ids.dtype)], axis=1)
            emb = self.model.get_input_embeddings(full, pixels, **extra).inputs_embeds
            logits = self.model.language_model(full, inputs_embeds=emb).logits[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            n = ids.shape[1]
            logp = sum(float(lp[n - 1 + j, variant[j]]) for j in range(len(variant)))
            total += math.exp(logp)
        return min(1.0, total)

    # -------------------------------------------------------------------------- generation

    def _gen_kwargs(self, enc: dict[str, Any]) -> dict[str, Any]:
        ids, pixels, mask, extra = self._split(enc)
        if self.gen.do_sample or self.gen.num_beams != 1:
            raise ValueError("the MLX backend implements greedy decoding only")
        return dict(
            input_ids=ids,
            pixel_values=pixels,
            mask=mask,
            max_tokens=self.gen.max_new_tokens,
            temperature=0.0,
            repetition_penalty=self.gen.repetition_penalty
            if self.gen.repetition_penalty != 1.0
            else None,
            skip_special_tokens=True,
            **extra,
        )

    def generate(self, enc: dict[str, Any], prompt: str = "") -> str:
        from mlx_vlm import generate

        return generate(self.model, self.processor, prompt, **self._gen_kwargs(enc)).text

    def timed_answer(self, image: Image.Image, messages: list[dict]) -> dict[str, Any]:
        """One question the way the app answers it: preprocessing, then streamed generation.
        Times to first token and to the last token are wall-clock, preprocessing included."""
        import mlx.core as mx
        from mlx_vlm import stream_generate

        tok = self.processor.tokenizer
        if hasattr(tok, "stopping_criteria"):  # what mlx_vlm.generate does before streaming
            tok.stopping_criteria.reset(self.model.config.eos_token_id)
        t0 = time.perf_counter()
        enc = self.encode(image, messages)
        t_prep = time.perf_counter()
        ttft, last = None, None
        for chunk in stream_generate(self.model, self.processor, "", **self._gen_kwargs(enc)):
            if ttft is None:
                ttft = time.perf_counter() - t0
            last = chunk
        total = time.perf_counter() - t0
        mx.clear_cache()
        if last is None:
            raise RuntimeError("generation produced no output")
        return {
            "preprocess_s": t_prep - t0,
            "ttft_s": ttft,
            "total_s": total,
            "prompt_tokens": last.prompt_tokens,
            "generation_tokens": last.generation_tokens,
            "prompt_tps": last.prompt_tps,
            "generation_tps": last.generation_tps,
            "peak_memory_gb": last.peak_memory,  # mlx-vlm reports 1e9 bytes
        }

    def run_prepared(self, prepared: list[dict]) -> list[BatchOutput]:
        out = []
        for enc in prepared:
            if self.image_token_id is not None:
                self.image_tokens.add([int((enc["input_ids"] == self.image_token_id).sum())])
            p = self.p_unanswerable(enc)
            out.append(BatchOutput(self.generate(enc), p))
        return out

    def run_batch(self, images: list[Image.Image], messages: list[list[dict]]) -> list[BatchOutput]:
        return self.run_prepared(self.prepare(images, messages))
