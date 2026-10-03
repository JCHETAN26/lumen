"""Training batches: the zero-shot prompt plus the target answer, loss on the answer only.

Each example is the user turn the zero-shot runs use (image + the dev-chosen prompt) followed by an
assistant turn holding the target. Labels are -100 everywhere except the answer tokens and the
end-of-turn token that follows them, so the model learns what to say and when to stop, and nothing
about the prompt or the image tokens.

The answer's start is found in the token ids, not by string offsets: the chat template's generation
marker (SmolVLM2: "\\nAssistant:") is tokenized once and its last occurrence in each sequence marks
where the answer begins. `tests/test_train_data.py` checks that this equals the length of the
prompt encoded on its own, so labels line up exactly with what generation sees at inference.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from lumen.data.annotations import Record
from lumen.data.images import ZipImages
from lumen.zeroshot.backends import long_edge_kwargs
from lumen.zeroshot.prompts import chat_messages

if TYPE_CHECKING:
    import torch

IGNORE = -100


class LabelAlignmentError(RuntimeError):
    """The answer couldn't be located in a tokenized example; never trained on silently."""


@dataclass(frozen=True)
class Example:
    question_id: str
    question: str
    target: str
    image: str  # file name inside the image zip


def examples_from_targets(pairs: Sequence[tuple[Record, str]]) -> list[Example]:
    return [Example(r["question_id"], r["question"], t, r["image"]) for r, t in pairs]


class TrainDataset:
    """Map-style dataset of (PIL image, Example); images are read from the zip on demand."""

    def __init__(self, examples: Sequence[Example], images: ZipImages) -> None:
        self.examples = list(examples)
        self.images = images

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[Image.Image, Example]:
        ex = self.examples[i]
        return self.images.open(ex.image), ex


class TrainCollator:
    """Turn (image, Example) pairs into a right-padded batch with answer-only labels."""

    def __init__(
        self, processor: Any, prompt_key: str, *, image_longest_edge: int | None = None
    ) -> None:
        self.processor = copy.deepcopy(processor)  # own copy: right padding, never shared
        self.processor.tokenizer.padding_side = "right"
        self.prompt_key = prompt_key
        self.image_kwargs = long_edge_kwargs(self.processor, image_longest_edge)
        tok = self.processor.tokenizer
        probe = chat_messages("Q?", prompt_key)
        with_marker = self.processor.apply_chat_template(probe, add_generation_prompt=True)
        without = self.processor.apply_chat_template(probe, add_generation_prompt=False)
        if not with_marker.startswith(without) or with_marker == without:
            raise LabelAlignmentError("chat template has no separable generation marker")
        self.marker_text = with_marker[len(without) :]
        self.marker = tok(self.marker_text, add_special_tokens=False).input_ids
        self.end_ids = {tok.eos_token_id}
        if not self.marker or None in self.end_ids:
            raise LabelAlignmentError("tokenizer has no generation marker or end-of-turn token")

    def texts(self, ex: Example) -> str:
        msgs = [
            *chat_messages(ex.question, self.prompt_key),
            {"role": "assistant", "content": [{"type": "text", "text": ex.target}]},
        ]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=False)

    def answer_start(self, ids: list[int]) -> int:
        k = len(self.marker)
        for i in range(len(ids) - k, -1, -1):
            if ids[i : i + k] == self.marker:
                return i + k
        raise LabelAlignmentError(f"generation marker {self.marker_text!r} not found")

    def __call__(self, batch: Sequence[tuple[Image.Image, Example]]) -> dict[str, torch.Tensor]:
        import torch

        texts = [self.texts(ex) for _, ex in batch]
        enc = self.processor(
            text=texts,
            images=[[im] for im, _ in batch],
            return_tensors="pt",
            padding=True,
            **self.image_kwargs,
        )
        labels = torch.full_like(enc["input_ids"], IGNORE)
        for row, (_, ex) in enumerate(batch):
            n = int(enc["attention_mask"][row].sum())
            ids = enc["input_ids"][row, :n].tolist()  # right padding: real tokens come first
            start = self.answer_start(ids)
            end = next((j for j in range(start, n) if ids[j] in self.end_ids), None)
            if end is None or end == start:
                raise LabelAlignmentError(f"{ex.question_id}: no answer tokens before end of turn")
            labels[row, start : end + 1] = enc["input_ids"][row, start : end + 1]
        out = dict(enc)
        out["labels"] = labels
        return out
