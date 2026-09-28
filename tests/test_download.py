from __future__ import annotations

import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from lumen.data.download import IntegrityError, fetch_pinned, verify
from lumen.pins import FilePin

PAYLOAD = bytes(range(256)) * 400  # 100 KiB
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class RangeHandler(http.server.BaseHTTPRequestHandler):
    requests: list[str | None] = []  # noqa: RUF012 (test-only shared log)

    def do_GET(self) -> None:
        rng = self.headers.get("Range")
        RangeHandler.requests.append(rng)
        start = int(rng.removeprefix("bytes=").rstrip("-")) if rng else 0
        body = PAYLOAD[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    RangeHandler.requests = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), RangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/data.zip"
    httpd.shutdown()


def pin(url: str, sha: str | None = SHA, size: int = len(PAYLOAD)) -> FilePin:
    return FilePin("data", url, size, sha)


def test_download_and_verify(server: str, tmp_path: Path) -> None:
    v = fetch_pinned(pin(server), tmp_path)
    assert v.pinned and v.sha256 == SHA and (tmp_path / "data.zip").read_bytes() == PAYLOAD


def test_resumes_partial_download(server: str, tmp_path: Path) -> None:
    (tmp_path / "data.zip.part").write_bytes(PAYLOAD[:1000])
    fetch_pinned(pin(server), tmp_path)
    assert RangeHandler.requests == ["bytes=1000-"]
    assert (tmp_path / "data.zip").read_bytes() == PAYLOAD


def test_trust_on_first_use_reports_hash_unpinned(server: str, tmp_path: Path) -> None:
    v = fetch_pinned(pin(server, sha=None), tmp_path)
    assert not v.pinned and v.sha256 == SHA


def test_hash_mismatch_stops(server: str, tmp_path: Path) -> None:
    with pytest.raises(IntegrityError, match="sha256"):
        fetch_pinned(pin(server, sha="0" * 64), tmp_path)


def test_size_mismatch_stops(tmp_path: Path) -> None:
    f = tmp_path / "data.zip"
    f.write_bytes(b"short")
    with pytest.raises(IntegrityError, match="size"):
        verify(f, pin("https://example.invalid/data.zip"))


def test_reuses_existing_file_without_network(tmp_path: Path) -> None:
    (tmp_path / "data.zip").write_bytes(PAYLOAD)
    assert fetch_pinned(pin("https://example.invalid/data.zip"), tmp_path).pinned


def test_verified_file_is_not_rehashed_until_it_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lumen.data.download as dl

    f = tmp_path / "data.zip"
    f.write_bytes(PAYLOAD)
    p = pin("https://example.invalid/data.zip")
    calls = []
    real = dl.sha256_file
    monkeypatch.setattr(dl, "sha256_file", lambda path: calls.append(path) or real(path))
    verify(f, p)
    verify(f, p)
    assert len(calls) == 1  # second call trusted the sidecar
    f.write_bytes(bytes(reversed(PAYLOAD)))  # same size, different content, new mtime
    with pytest.raises(IntegrityError, match="sha256"):
        verify(f, p)
    assert len(calls) == 2


def test_unpinned_file_gets_no_sidecar(tmp_path: Path) -> None:
    f = tmp_path / "data.zip"
    f.write_bytes(PAYLOAD)
    verify(f, pin("https://example.invalid/data.zip", sha=None))
    assert not (tmp_path / "data.zip.verified.json").exists()
