"""Turn a model's raw generation into the answer string that gets scored.

The rules are fixed, applied identically to every model, and versioned (EXTRACTOR_VERSION is
recorded with every prediction). The raw text is always kept next to the extracted answer.

1. Drop a leading role or answer label ("Assistant:", "Answer:"), then keep the first non-empty
   line. VizWiz answers are short, and later lines are commentary.
2. Strip surrounding quotes and whitespace. Punctuation and case are left to the official
   metric's normalization, so we don't double-normalize.
3. Map clear abstentions ("I can't tell", "cannot be determined", ...) to "unanswerable",
   VizWiz's label for them. Without this, a model that abstains in its own words would be scored
   wrong on exactly the questions it handled well. Only whole-answer abstentions match: an answer
   that merely contains "not" is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EXTRACTOR_VERSION = "v1"
UNANSWERABLE = "unanswerable"

_LABEL = re.compile(r"^\s*(?:assistant|answer)\s*:\s*", re.IGNORECASE)
_ABSTAIN = re.compile(
    r"^(?:sorry,?\s*)?(?:"
    r"unanswerable|not answerable|"
    r"(?:i\s*)?(?:can'?not|can'?t|am unable to|unable to|don'?t|do not)\s+"
    r"(?:tell|see|answer|determine|know|read|identify|be (?:answered|determined))|"
    r"(?:it\s+is\s+|it'?s\s+)?(?:not possible|impossible) to (?:tell|answer|determine|see)|"
    r"(?:the\s+)?(?:image|photo|picture) is too (?:blurry|dark|blurred)|"
    r"cannot be (?:answered|determined)|can'?t be (?:answered|determined)"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Extracted:
    answer: str
    abstained_by_rule: bool


def extract_answer(raw: str) -> Extracted:
    text = _LABEL.sub("", raw.strip())
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first = first.strip("\"'“”‘’` ").strip()
    if _ABSTAIN.match(first.rstrip(".! ")):
        return Extracted(UNANSWERABLE, first.lower().rstrip(".! ") != UNANSWERABLE)
    return Extracted(first, False)
