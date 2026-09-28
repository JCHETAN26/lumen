"""The usage-cap detector decides skip vs. fail for live tests, so it is unit-tested offline."""

from __future__ import annotations

import io
import urllib.error

import pytest

from lumen.live import usage_cap_reason


def http_error(code: int, msg: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, msg, {}, io.BytesIO())  # type: ignore[arg-type]


class Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class HubError(Exception):
    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.response = Resp(status)


@pytest.mark.parametrize(
    "exc",
    [
        http_error(429, "Too Many Requests"),
        http_error(403, "Quota exceeded"),
        HubError("429 Client Error: Too Many Requests", 429),
        HubError("You have exceeded your monthly quota", 402),
        RuntimeError("You have reached your usage limit. You will regain access on Oct 1"),
    ],
)
def test_caps_are_recognized(exc: Exception) -> None:
    assert usage_cap_reason(exc) is not None


@pytest.mark.parametrize(
    "exc",
    [
        http_error(404, "Not Found"),
        http_error(403, "Forbidden"),  # a permissions problem is a real failure
        http_error(500, "Internal Server Error"),
        HubError("Repository not found", 404),
        ValueError("sha256 mismatch"),
    ],
)
def test_real_failures_are_not_skipped(exc: Exception) -> None:
    assert usage_cap_reason(exc) is None
