"""Prompt templates for zero-shot VizWiz answering. Fixed before any model output was seen.

`plain` is the standard short-answer VQA instruction. `abstain` adds VizWiz's own convention
for questions a photo can't answer. Which prompt a model uses on val is chosen on dev only
(kaggle/run_stages.py baseline --prompt dev-selected).

Changing any template changes PROMPT_VERSION, which is recorded in every prediction file and
result, so outputs from different templates can never be mixed or compared by accident.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "v1"

_SHORT = "Answer the question using a single word or phrase."
PROMPTS: dict[str, str] = {
    "plain": "{question}\n" + _SHORT,
    "abstain": (
        "{question}\n" + _SHORT + " If the photo cannot answer the question (for example, it is "
        "too blurry or dark, or the object is out of frame), answer exactly: unanswerable"
    ),
}


def prompt_text(question: str, prompt_key: str) -> str:
    if prompt_key not in PROMPTS:
        raise KeyError(f"unknown prompt {prompt_key!r}; choose from {sorted(PROMPTS)}")
    question = " ".join(question.split())  # VizWiz questions are speech transcripts
    if not question:
        raise ValueError("empty question")
    return PROMPTS[prompt_key].format(question=question)


def chat_messages(question: str, prompt_key: str) -> list[dict[str, Any]]:
    """One user turn: the image, then the text. Each backend applies its model's chat template."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text(question, prompt_key)},
            ],
        }
    ]
