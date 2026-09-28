"""Live checks against the real services (pytest -m live). Excluded from CI.

Each test skips cleanly on a usage cap or rate limit (tests/conftest.py) and on missing
credentials; any other error fails, because it means a pin or a service changed.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

import pytest
from conftest import skip_on_usage_cap

from lumen.pins import load_pins

pytestmark = pytest.mark.live
PINS = load_pins()
TIMEOUT = 30


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


@pytest.mark.parametrize("key", sorted(PINS.models))
def test_pinned_model_revision_resolves_on_hub(key: str) -> None:
    m = PINS.models[key]
    url = f"https://huggingface.co/api/models/{m.repo}/revision/{m.revision}"
    with (
        skip_on_usage_cap("Hugging Face"),
        urllib.request.urlopen(
            urllib.request.Request(url, headers=_hf_headers()), timeout=TIMEOUT
        ) as r,
    ):
        info = json.load(r)
    assert info["sha"] == m.revision
    assert any(s["rfilename"] == "config.json" for s in info.get("siblings", []))


@pytest.mark.parametrize("name", sorted(PINS.files))
def test_vizwiz_file_size_matches_pin(name: str) -> None:
    pin = PINS.files[name]
    req = urllib.request.Request(
        pin.url, method="HEAD", headers={"User-Agent": "lumen-research-download/0.1"}
    )
    with skip_on_usage_cap("VizWiz host"), urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        size = int(r.headers["Content-Length"])
    assert size == pin.size, f"{name} changed upstream: {size} != pinned {pin.size}"


def _kaggle_credentials() -> tuple[str, str] | None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    for p in (
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ):
        if p.exists():
            d = json.loads(p.read_text())
            return d["username"], d["key"]
    return None


def test_kaggle_api_reachable_with_credentials() -> None:
    creds = _kaggle_credentials()
    if creds is None:
        pytest.skip("SKIPPED: no Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json)")
    auth = base64.b64encode(f"{creds[0]}:{creds[1]}".encode()).decode()
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/kernels/list?user={creds[0]}&pageSize=1",
        headers={"Authorization": f"Basic {auth}"},
    )
    with skip_on_usage_cap("Kaggle"), urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        assert r.status == 200
