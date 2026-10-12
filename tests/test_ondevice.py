"""Stage 4 logic that doesn't need a Mac: the pinned sample, latency/agreement stats, the thermal
guard, the accuracy/latency/summary flow (with a fake backend), adapter pins, and pin_adapter.

The MLX code itself (lumen.ondevice.mlx_backend) runs only on Apple silicon; `scripts/mac_bench.py
self-test` checks it there before any benchmark.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import pytest
import yaml
from conftest import fake_records, jpeg, run_stages

from lumen.ondevice.bench import Throttled, accuracy_run, latency_run, summarize, wait_until_cool
from lumen.ondevice.mlx_backend import set_long_edge
from lumen.ondevice.sample import SampleError, load_mac_sample
from lumen.ondevice.stats import agreement, latency_summary
from lumen.pins import PinError, Pins, adapter_pointer, load_pins
from lumen.zeroshot.backends import FakeBackend


def make_sample(tmp: Path, n: int = 12) -> tuple[Path, Path]:
    recs = fake_records("val", n)
    for i, r in enumerate(recs):
        r["question_id"] = f"VizWiz_val_{i:08d}"
    root = tmp / "mac_sample" / "mac_sample_val_n12_seed5"
    (root / "images").mkdir(parents=True)
    manifest = {}
    for r in recs:
        data = jpeg()
        (root / "images" / r["image"]).write_bytes(data)
        manifest[r["image"]] = hashlib.sha256(data).hexdigest()
    (root / "annotations.json").write_text(json.dumps(recs))
    (root / "manifest.json").write_text(json.dumps({"images_sha256": manifest}))
    ids = tmp / "mac_sample_val.txt"
    ids.write_text("\n".join(sorted(r["question_id"] for r in recs)) + "\n")
    return tmp / "mac_sample", ids


def test_sample_is_verified(tmp_path: Path) -> None:
    sample, ids = make_sample(tmp_path)
    records, images = load_mac_sample(sample, ids)
    assert len(records) == 12 and records == sorted(records, key=lambda r: r["question_id"])
    assert images.open(records[0]["image"]).mode == "RGB"
    img = next((sample / "mac_sample_val_n12_seed5" / "images").iterdir())
    img.write_bytes(img.read_bytes() + b"x")
    with pytest.raises(SampleError, match="sha256"):
        load_mac_sample(sample, ids)


def test_sample_ids_must_match_the_committed_list(tmp_path: Path) -> None:
    sample, ids = make_sample(tmp_path)
    ids.write_text(ids.read_text().splitlines()[0] + "\n")
    with pytest.raises(SampleError, match="differ"):
        load_mac_sample(sample, ids)
    with pytest.raises(SampleError, match="extract"):
        load_mac_sample(tmp_path / "nowhere", ids)


def test_latency_summary_and_agreement() -> None:
    s = latency_summary([1.0, 2.0, 3.0, 4.0, 10.0])
    assert s["median"] == 3.0 and s["p90"] == pytest.approx(7.6) and s["n"] == 5
    for bad in ([], [1.0, float("nan")], [-1.0]):
        with pytest.raises(ValueError):
            latency_summary(bad)
    a = {"q1": "Dog", "q2": "unanswerable", "q3": "red"}
    b = {"q1": " dog ", "q2": "cat", "q3": "red"}
    g = agreement(a, b)
    assert g["exact_answer_agreement"] == pytest.approx(2 / 3)
    assert g["abstention_agreement"] == pytest.approx(2 / 3)
    assert "Dog" not in json.dumps(g)  # rates only, never answer text
    with pytest.raises(ValueError):
        agreement(a, {"q1": "dog"})


def test_wait_until_cool() -> None:
    readings = iter([80, 90, 100])
    slept = []
    out = wait_until_cool(lambda: next(readings), max_wait_s=600, poll_s=30, sleep=slept.append)
    assert out == {"cpu_speed_limit_pct": 100, "waited_s": 60.0} and slept == [30, 30]
    with pytest.raises(Throttled, match="cool"):
        wait_until_cool(lambda: 70, max_wait_s=60, poll_s=30, sleep=lambda s: None)
    assert wait_until_cool(lambda: None, max_wait_s=0)["waited_s"] == 0  # not a Mac


class FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = argparse.Namespace(
            max_image_size={"longest_edge": 512}, size={"longest_edge": 2048}
        )


def test_set_long_edge_validates_like_kaggle() -> None:
    p = FakeProcessor()
    set_long_edge(p, 1024)
    assert p.image_processor.size == {"longest_edge": 1024}
    for bad in (768, 4096, 256):
        with pytest.raises(ValueError):
            set_long_edge(FakeProcessor(), bad)
    q = FakeProcessor()
    set_long_edge(q, None)
    assert q.image_processor.size == {"longest_edge": 2048}


def test_set_long_edge_on_a_transformers_size_dict() -> None:
    @dataclasses.dataclass
    class SizeDict:  # transformers 5: a dataclass with get(), not a mapping
        longest_edge: int | None = None
        height: int | None = None

        def get(self, key, default=None):
            return getattr(self, key, None) or default

    p = FakeProcessor()
    p.image_processor.size = SizeDict(longest_edge=2048)
    set_long_edge(p, 1024)
    assert p.image_processor.size == SizeDict(longest_edge=1024)


class TimedFake(FakeBackend):
    """The fake backend plus the app-style timed answer the latency run uses."""

    def __init__(self, answer: str = "dog", variant: str = "fp16") -> None:
        super().__init__(answer=answer)
        self.variant = variant
        self.rng = random.Random(0)

    def describe(self):
        return {**super().describe(), "model_key": "smolvlm2-500m", "variant": self.variant}

    def timed_answer(self, image, messages):
        t = 1.0 + self.rng.random()
        return {
            "preprocess_s": 0.05,
            "ttft_s": t,
            "total_s": t + 0.2,
            "prompt_tokens": 400,
            "generation_tokens": 3,
            "prompt_tps": 400 / t,
            "generation_tps": 60.0,
            "peak_memory_gb": 1.5,
        }


def test_accuracy_latency_and_summary(tmp_path: Path) -> None:
    sample, ids = make_sample(tmp_path)
    records, images = load_mac_sample(sample, ids)
    benches, files = {}, {}
    for variant, answer in (("fp16", "dog"), ("4bit", "cat")):
        backend = TimedFake(answer, variant)
        pred = tmp_path / "predictions" / f"{variant}.predictions.jsonl"
        acc = accuracy_run(backend, records, images, prompt_key="abstain", pred_path=pred)
        lat = latency_run(backend, records, images, prompt_key="abstain", n=4, repeats=2)
        assert acc["metrics"]["n"] == 12 and acc["predictions"]["stored"].startswith("local only")
        assert lat["questions"] == 4 and lat["ttft_s"]["n"] == 8
        assert len(lat["total_s_median_by_repeat"]) == 2
        header = json.loads(pred.read_text().splitlines()[0])["_header"]
        assert header["model"]["variant"] == variant and header["split"] == "mac_sample"
        benches[variant] = {"accuracy": acc, "latency": lat, "size_on_disk_bytes": 100}
        files[variant] = pred
    out = summarize(records, benches, files)
    fp16, q4 = out["variants"]
    assert fp16["accuracy"] == pytest.approx(100.0) and "vs_fp16" not in fp16
    # The fake "4bit" answers "cat" on every answerable question: all of its loss is there.
    assert q4["vs_fp16"]["accuracy_diff"] < 0
    assert q4["vs_fp16"]["accuracy_diff_unanswerable"] == pytest.approx(0.0)
    assert 0 < q4["agreement_with_fp16"]["exact_answer_agreement"] < 1
    # Kaggle reference on the same questions (more questions in the file are ignored).
    from lumen.zeroshot.harness import read_predictions

    kaggle = read_predictions(files["fp16"])[1] | {"VizWiz_val_99999999": {"answer": "x"}}
    out = summarize(records, benches, files, kaggle_preds=kaggle)
    assert out["variants"][0]["agreement_with_kaggle_fp16"]["exact_answer_agreement"] == 1.0
    files["4bit"].write_text(files["4bit"].read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        summarize(records, benches, files)
    with pytest.raises(ValueError, match="no fp16"):
        summarize(records, {"4bit": benches["4bit"]}, files)


ADAPTER = {
    "repo": "me/lumen-adapter",
    "revision": "a" * 40,
    "path": "runs/lr2e-4/epoch-1",
    "base": "smolvlm2-500m",
    "image_longest_edge": 1024,
    "prompt_key": "abstain",
    "files": {"adapter_model.safetensors": "b" * 64},
}


def pins_with(adapters: dict) -> Pins:
    p = load_pins()
    return Pins(**{**p.__dict__, "raw": {**p.raw, "adapters": adapters}})


def test_adapter_pointer_validation() -> None:
    a = adapter_pointer("x", pins_with({"x": ADAPTER}))
    assert a["repo_type"] == "model" and a["key"] == "x" and a["path"] == ADAPTER["path"]
    with pytest.raises(PinError, match="no adapter"):
        adapter_pointer("y", pins_with({"x": ADAPTER}))
    for bad, match in (
        ({**ADAPTER, "revision": "main"}, "40-hex"),
        ({**ADAPTER, "base": "nope"}, "not a pinned model"),
        ({**ADAPTER, "files": {"a": "short"}}, "sha256"),
        ({k: v for k, v in ADAPTER.items() if k != "path"}, "missing"),
    ):
        with pytest.raises(PinError, match=match):
            adapter_pointer("x", pins_with({"x": bad}))


def test_pin_adapter_prints_a_valid_pin(env: Path, capsys) -> None:
    from lumen import hub

    epoch_dir = env / "e1"
    epoch_dir.mkdir()
    (epoch_dir / "adapter_model.safetensors").write_bytes(b"w")
    spec = {
        "model": {"model_key": "smolvlm2-500m", "revision": load_pins().models["smolvlm2-500m"]},
        "config": {"image_longest_edge": 1024, "prompt_key": "abstain"},
    }
    spec["model"]["revision"] = spec["model"]["revision"].revision
    doc = {
        "spec": spec,
        "epochs_published": [
            {
                "epoch": 1,
                "step": 2309,
                "adapter": {
                    "repo": "cjitendr/lumen-adapter",
                    "repo_type": "model",
                    "path": "runs/lr2e-4/epoch-1",
                    "revision": "c" * 40,
                    "files": hub.folder_digest(epoch_dir),
                },
            }
        ],
    }
    out = env / "results" / "finetune" / "lr2e-4" / "train.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(doc))
    assert run_stages.cmd_pin_adapter(argparse.Namespace(run="lr2e-4", epoch=1)) == 0
    entry = yaml.safe_load(capsys.readouterr().out)
    a = adapter_pointer("lr2e-4-e1", pins_with(entry))
    assert a["revision"] == "c" * 40 and a["image_longest_edge"] == 1024 and a["step"] == 2309
    assert a["files"] == hub.folder_digest(epoch_dir)
