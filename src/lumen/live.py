"""Recognize usage-cap and rate-limit errors from the live services Lumen talks to.

Live tests (pytest mark `live`) call real services: Hugging Face Hub, Kaggle, the VizWiz host,
and later Anthropic if an LLM judge is added. A usage cap or rate limit is not a test failure, so
tests/conftest.py turns errors recognized here into a clean skip. Anything else is re-raised: a
genuine failure must still fail.
"""

from __future__ import annotations

import urllib.error

# Status codes services use for "you've hit a limit": 429 everywhere; Kaggle and some HF
# endpoints answer quota exhaustion with 403 + a quota message, so 403 counts only with one.
_RATE_LIMIT_STATUS = {429}
_QUOTA_STATUS = {402, 403}
_CAP_PHRASES = (
    "rate limit", "rate-limit", "ratelimit", "too many requests", "quota", "usage limit",
    "regain access", "exceeded your", "credit balance",
)  # fmt: skip


def _status(exc: BaseException) -> int | None:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)  # requests / huggingface_hub / httpx errors
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def usage_cap_reason(exc: BaseException) -> str | None:
    """A short reason if `exc` is a usage cap or rate limit, else None."""
    status = _status(exc)
    text = str(exc).lower()
    if isinstance(exc, urllib.error.HTTPError):
        text += " " + str(exc.reason).lower()
    has_phrase = any(p in text for p in _CAP_PHRASES)
    if status in _RATE_LIMIT_STATUS:
        return f"HTTP {status} (rate limited)"
    if status in _QUOTA_STATUS and has_phrase:
        return f"HTTP {status} (quota or usage cap)"
    if status is None and has_phrase:
        return "usage cap or rate limit"
    return None
