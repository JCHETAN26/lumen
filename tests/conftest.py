"""Shared test helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest

from lumen.live import usage_cap_reason


@contextlib.contextmanager
def skip_on_usage_cap(service: str) -> Iterator[None]:
    """Turn a usage cap or rate limit from `service` into a visible skip; re-raise anything else."""
    try:
        yield
    except Exception as exc:
        reason = usage_cap_reason(exc)
        if reason is None:
            raise
        pytest.skip(f"SKIPPED: {service} usage cap: {reason}")
