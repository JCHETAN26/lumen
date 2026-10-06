"""The local app server (lumen.app.server) with the fake backend: the API contract the web app
relies on, and its privacy and safety guards."""

from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from lumen.app.server import (
    MAX_IMAGE_BYTES,
    create_app,
    is_unanswerable,
    read_photo,
)
from lumen.zeroshot.backends import FakeBackend

H = {"X-Lumen": "1"}


def png(size=(8, 6), exif_rotate: bool = False) -> bytes:
    buf = io.BytesIO()
    im = Image.new("RGB", size, (200, 10, 10))
    if exif_rotate:
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees on display
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path) -> TestClient:
    web = tmp_path / "dist"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>Lumen</title>")
    app = create_app(FakeBackend(), prompt_key="abstain", threshold=0.9, static_dir=web)
    return TestClient(app, base_url="http://localhost:8765")


def ask(client: TestClient, question: str, data: bytes | None = None, headers=H):
    files = {"image": ("photo.png", data if data is not None else png(), "image/png")}
    return client.post("/api/ask", files=files, data={"question": question}, headers=headers)


def test_answer_and_cant_tell(client) -> None:
    r = ask(client, "What is this?")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "dog" and body["unanswerable"] is False
    assert 0 <= body["p_unanswerable"] <= 1 and body["seconds"] >= 0
    r = ask(client, "Is it blurry?")  # the fake abstains on 'blur'
    assert r.json()["unanswerable"] is True and r.json()["answer"] is None
    assert r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]


def test_health_and_static_page(client) -> None:
    h = client.get("/api/health", headers=H).json()
    assert h["ready"] and h["unanswerable_threshold"] == 0.9
    page = client.get("/")
    assert page.status_code == 200 and "Lumen" in page.text
    assert page.headers["x-content-type-options"] == "nosniff"


def test_guards(client) -> None:
    assert ask(client, "What?", headers={}).status_code == 400  # no X-Lumen: cross-site forms
    evil = TestClient(client.app, base_url="http://evil.example:8765")
    assert evil.get("/api/health", headers=H).status_code == 403  # DNS rebinding
    assert evil.get("/").status_code == 403
    v6 = TestClient(client.app, base_url="http://[::1]:8765")
    assert v6.get("/api/health", headers=H).status_code == 200
    assert ask(client, "   ").status_code == 422
    assert ask(client, "x" * 501).status_code == 422
    assert ask(client, "What?", data=b"not an image").status_code == 415
    assert ask(client, "What?", data=b"0" * (MAX_IMAGE_BYTES + 1)).status_code == 413


def test_photo_is_turned_upright() -> None:
    im = read_photo(png((8, 6), exif_rotate=True))
    assert im.size == (6, 8) and im.mode == "RGB"
    with pytest.raises(ValueError, match="photo"):
        read_photo(b"\x00\x01")


def test_cant_tell_decision() -> None:
    assert is_unanswerable("Unanswerable", 0.0, None)
    assert not is_unanswerable("dog", 0.95, None)
    assert is_unanswerable("dog", 0.95, 0.9) and not is_unanswerable("dog", 0.9, 0.9)


def test_nothing_is_logged(client, caplog) -> None:
    with caplog.at_level("DEBUG"):
        ask(client, "What colour is my secret note?")
    assert "secret" not in caplog.text
