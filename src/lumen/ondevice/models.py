"""Which MLX model to load, and with what settings: a pinned adapter key, or "base"."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from lumen.pins import REPO_ROOT, adapter_pointer, load_pins

MODELS = REPO_ROOT / "models"


def model_settings(key: str) -> dict[str, Any]:
    """Identity (for results and prediction headers), image long edge, and prompt of a model."""
    pins = load_pins()
    if key == "base":
        pin = pins.models["smolvlm2-500m"]
        override = REPO_ROOT / "configs" / "resolution_override.yaml"
        return {
            "describe": {"model_key": pin.key, "repo": pin.repo, "revision": pin.revision},
            "long_edge": int(yaml.safe_load(override.read_text())["long_edge"]),
            "prompt_key": "abstain",  # SmolVLM2-500M's dev_select choice
        }
    a = adapter_pointer(key, pins)
    pin = pins.models[a["base"]]
    return {
        "describe": {
            "model_key": pin.key,
            "repo": pin.repo,
            "revision": pin.revision,
            "adapter": {k: a[k] for k in ("key", "revision", "path")},
        },
        "long_edge": a["image_longest_edge"],
        "prompt_key": a["prompt_key"],
    }


def load_backend(key: str, variant: str) -> tuple[Any, dict[str, Any], float, Path]:
    """(MLXBackend, settings, load seconds, model folder) for a prepared model (mac_prepare.py)."""
    from lumen.ondevice.mlx_backend import MLXBackend

    s = model_settings(key)
    path = MODELS / "mlx" / key / variant
    if not path.is_dir():
        raise SystemExit(f"{path} missing: run scripts/mac_prepare.py first")
    t0 = time.perf_counter()
    backend = MLXBackend.load(
        path,
        describe={**s["describe"], "variant": variant},
        image_longest_edge=s["long_edge"],
    )
    return backend, s, time.perf_counter() - t0, path
