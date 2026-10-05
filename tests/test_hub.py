from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import FakeHubApi

from lumen import hub

REPO = "someone/lumen-predictions"


def test_predictions_repo_is_pinned() -> None:
    assert hub.predictions_repo() == "cjitendr/lumen-predictions"
    assert hub.adapter_repo() == "cjitendr/lumen-adapter"


@pytest.mark.parametrize(
    ("api", "match"),
    [
        (FakeHubApi(exists=False), "can't access"),
        (FakeHubApi(private=False), "PUBLIC"),
        (FakeHubApi(role="read"), "can't write"),
    ],
)
def test_preflight_refuses(api: FakeHubApi, match: str) -> None:
    with pytest.raises(hub.HubError, match=match):
        hub.preflight(api, REPO)


@pytest.mark.parametrize(
    ("scoped", "ok"),
    [
        ([{"entity": {"type": "dataset", "name": REPO}, "permissions": ["repo.write"]}], True),
        ([{"entity": {"type": "user", "name": "someone"}, "permissions": ["repo.write"]}], True),
        (
            [{"entity": {"type": "dataset", "name": REPO}, "permissions": ["repo.content.read"]}],
            False,
        ),
        (
            [
                {
                    "entity": {"type": "model", "name": "someone/adapter"},
                    "permissions": ["repo.write"],
                }
            ],
            False,
        ),
    ],
)
def test_fine_grained_token_scopes(scoped: list, ok: bool) -> None:
    who = {"auth": {"accessToken": {"role": "fineGrained", "fineGrained": {"scoped": scoped}}}}
    assert hub._can_write(who, REPO) is ok


def test_fine_grained_model_repo_scope() -> None:
    scoped = [
        {"entity": {"type": "model", "name": "someone/adapter"}, "permissions": ["repo.write"]}
    ]
    who = {"auth": {"accessToken": {"role": "fineGrained", "fineGrained": {"scoped": scoped}}}}
    assert hub._can_write(who, "someone/adapter", "model")
    assert not hub._can_write(who, "someone/adapter", "dataset")


def test_upload_returns_pinned_pointer(tmp_path: Path) -> None:
    f = tmp_path / "val-plain.predictions.jsonl"
    f.write_text('{"_header": {}}\n{"question_id": "a"}\n')
    api = FakeHubApi()
    ptr = hub.upload(api, REPO, f, "baseline/m/val-plain.predictions.jsonl", "msg")
    assert ptr["revision"] == f"{1:040x}" and ptr["lines"] == 2
    assert ptr["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert api.files["baseline/m/val-plain.predictions.jsonl"] == f.read_bytes()


def test_upload_refuses_public_repo(tmp_path: Path) -> None:
    f = tmp_path / "p.jsonl"
    f.write_text("{}\n")
    with pytest.raises(hub.HubError, match="PUBLIC"):
        hub.upload(FakeHubApi(private=False), REPO, f, "p.jsonl", "msg")


def test_missing_token_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(hub.HubError, match="HF_TOKEN"):
        hub.make_api()


def test_download_verifies_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf = pytest.importorskip("huggingface_hub")
    f = tmp_path / "p.jsonl"
    f.write_text("{}\n")
    monkeypatch.setattr(hf, "hf_hub_download", lambda *a, **k: str(f))
    good = {
        "repo": REPO,
        "path": "p.jsonl",
        "revision": "a" * 40,
        "sha256": hashlib.sha256(b"{}\n").hexdigest(),
    }
    assert hub.download(good) == f
    with pytest.raises(hub.HubError, match="sha256"):
        hub.download({**good, "sha256": "0" * 64})


def test_list_files_reads_one_exact_commit() -> None:
    api = FakeHubApi()
    api.files = {"baseline/m/a.predictions.jsonl": b"", "baseline/n/b.predictions.jsonl": b""}
    api.commits = 7
    sha, files = hub.list_files(api, "me/preds", "baseline/m/")
    assert sha == f"{7:040x}" and files == ["baseline/m/a.predictions.jsonl"]
    with pytest.raises(hub.HubError, match="not private"):
        hub.list_files(FakeHubApi(private=False), "me/preds", "baseline/")


def test_fetch_refuses_a_branch_name() -> None:
    with pytest.raises(hub.HubError, match="full commit SHA"):
        hub.fetch("me/preds", "baseline/m/a.predictions.jsonl", "main")


def test_pointer_for_an_existing_file(tmp_path) -> None:
    f = tmp_path / "x.predictions.jsonl"
    f.write_text('{"_header": {}}\n{"question_id": "q"}\n')
    p = hub.pointer("me/preds", "baseline/m/x.predictions.jsonl", "a" * 40, f)
    assert p["lines"] == 2 and len(p["sha256"]) == 64 and p["revision"] == "a" * 40
