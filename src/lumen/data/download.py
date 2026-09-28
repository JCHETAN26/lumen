"""Download pinned dataset files: resumable, size-checked, and SHA-256 verified.

A file whose pin has `sha256: null` is trusted on first use: size must match, the hash is computed
and reported for committing to configs/pins.yaml, and the result is marked unpinned. Once a hash
is pinned, any mismatch deletes nothing but raises, so the run stops before using the data.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from lumen.pins import FilePin

_CHUNK = 8 << 20
_USER_AGENT = "lumen-research-download/0.1"


class IntegrityError(RuntimeError):
    pass


@dataclass
class Verified:
    path: Path
    size: int
    sha256: str
    pinned: bool  # False: trust-on-first-use; commit `sha256` to configs/pins.yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def ensure_free_space(directory: Path, needed_bytes: int, margin: float = 1.05) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < needed_bytes * margin:
        raise OSError(
            f"{directory} has {free / 2**30:.1f} GiB free; need {needed_bytes * margin / 2**30:.1f}"
            " GiB. Free space or point LUMEN_DATA_DIR at a larger disk."
        )


def _fetch(url: str, dest: Path, expected_size: int) -> None:
    """Download to dest.part with HTTP Range resume, then rename to dest."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > expected_size:
        part.unlink()
        have = 0
    if have == expected_size:
        part.rename(dest)
        return
    ensure_free_space(dest.parent, expected_size - have)
    headers = {"User-Agent": _USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored Range: start over
            have = 0
        mode = "ab" if have else "wb"
        done, t0, last = have, time.monotonic(), 0.0
        with open(part, mode) as f:
            while chunk := resp.read(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last > 30:
                    rate = (done - have) / max(now - t0, 1e-9) / 2**20
                    print(f"  {dest.name}: {done / 2**30:.2f}/{expected_size / 2**30:.2f} GiB"
                          f" ({rate:.0f} MiB/s)", file=sys.stderr, flush=True)
                    last = now
    if part.stat().st_size != expected_size:
        raise IntegrityError(
            f"{dest.name}: got {part.stat().st_size} bytes, expected {expected_size}; rerun to resume"
        )
    part.rename(dest)


def verify(path: Path, pin: FilePin) -> Verified:
    size = path.stat().st_size
    if size != pin.size:
        raise IntegrityError(f"{path.name}: size {size} != pinned {pin.size}")
    digest = sha256_file(path)
    if pin.sha256 is not None and digest != pin.sha256:
        raise IntegrityError(f"{path.name}: sha256 {digest} != pinned {pin.sha256}")
    return Verified(path=path, size=size, sha256=digest, pinned=pin.sha256 is not None)


def fetch_pinned(pin: FilePin, directory: Path) -> Verified:
    """Download (or reuse) the pinned file in `directory` and verify it."""
    dest = directory / pin.filename
    if not dest.exists():
        print(f"downloading {pin.url} ({pin.size / 2**30:.2f} GiB)", file=sys.stderr, flush=True)
        _fetch(pin.url, dest, pin.size)
    return verify(dest, pin)
