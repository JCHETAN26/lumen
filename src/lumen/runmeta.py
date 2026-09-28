"""Metadata recorded in every result file, and the one function that writes results.

Every result JSON carries: git commit (and whether the tree was dirty), hardware, OS, Python and
library versions, and a UTC timestamp. Results are never overwritten: rerunning writes a new file.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from lumen.pins import REPO_ROOT

# Libraries whose versions can change a number. Missing ones are recorded as null.
_TRACKED_LIBS = (
    "lumen", "numpy", "scikit-learn", "pillow", "torch", "transformers", "peft", "trl",
    "accelerate", "bitsandbytes", "mlx", "mlx-vlm", "mlx-lm",
)  # fmt: skip


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info() -> dict[str, Any]:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    # Result files themselves don't affect what was measured.
    status = _run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            ":!results",
        ]
    )
    return {"commit": commit or None, "dirty": bool(status)}


def _nvidia_gpus() -> list[dict[str, Any]]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory_mib": int(parts[1]), "driver": parts[2]})
    return gpus


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cpus": os.cpu_count(),
        "load_average_1m_5m_15m": [round(x, 2) for x in os.getloadavg()],
    }
    if platform.system() == "Darwin":
        info["chip"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["memory_gib"] = round(int(mem) / 2**30, 1) if mem else None
        info["os_version"] = _run(["sw_vers", "-productVersion"])
        info["model"] = _run(["sysctl", "-n", "hw.model"])
        # The M2 Air is fanless: power source and thermal state change latency numbers.
        batt = _run(["pmset", "-g", "batt"])
        info["power_source"] = (
            "ac" if "AC Power" in batt else "battery" if "Battery Power" in batt else None
        )
        therm = _run(["pmset", "-g", "therm"])
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
        info["cpu_speed_limit_pct"] = int(m.group(1)) if m else None
    elif platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["cpu"] = m.group(1).strip() if m else platform.processor()
        m = re.search(r"MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text())
        info["memory_gib"] = round(int(m.group(1)) / 2**20, 1) if m else None
    info["gpus"] = _nvidia_gpus()
    info["kaggle"] = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").is_dir()
    return info


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for lib in _TRACKED_LIBS:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def run_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "hardware": hardware_info(),
        "python": sys.version.split()[0],
        "libraries": library_versions(),
        "argv": sys.argv,
    }


def write_result(path: Path, payload: dict[str, Any]) -> Path:
    """Write `payload` plus run metadata to `path` as JSON. Refuses to overwrite.

    If `path` exists, a numbered sibling (`name.1.json`, ...) is written instead, so a rerun can
    never silently replace a committed result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    target, i = path, 0
    while target.exists():
        i += 1
        target = path.with_name(f"{path.stem}.{i}{path.suffix}")
    doc = {"meta": run_metadata(), **payload}
    with open(target, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    return target
