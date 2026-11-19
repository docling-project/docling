# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docling.models.utils import hf_model_download, model_manifest
from docling.models.utils.model_manifest import (
    record_downloads,
    record_hf_snapshot,
    record_url_file,
    write_cyclonedx,
)
from docling.utils import model_downloader

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FILES = ["model.safetensors", "sub/config.json"]


def _snapshot(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "sub" / "config.json").write_bytes(b"{}")
    # huggingface_hub's bookkeeping next to a local-dir download, and a file
    # an earlier download of another commit left behind.
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "x.metadata").write_bytes(b"meta")
    (root / "deleted-upstream.bin").write_bytes(b"old")
    return root


def test_manifest_lists_repository_commit_and_file_digests(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "org--model")
    ocr_file = tmp_path / "rec.onnx"
    ocr_file.write_bytes(b"onnx")

    with record_downloads() as record:
        record_hf_snapshot("org/model", COMMIT, snapshot, FILES)
        record_url_file("https://example.org/models/rec.onnx", ocr_file)
    path = write_cyclonedx(record, tmp_path, "9.9.9")

    bom = json.loads(path.read_text())
    assert path.name == "models.cdx.json"
    assert bom["bomFormat"] == "CycloneDX"
    assert "serialNumber" not in bom and "timestamp" not in bom.get("metadata", {})
    hf, url = bom["components"]
    assert hf["type"] == "machine-learning-model"
    assert hf["purl"] == f"pkg:huggingface/org/model@{COMMIT}"
    assert hf["version"] == COMMIT
    assert {f["name"]: f["hashes"][0]["content"] for f in hf["components"]} == {
        "model.safetensors": _sha256(b"weights"),
        "sub/config.json": _sha256(b"{}"),
    }
    assert url["externalReferences"][0]["url"] == "https://example.org/models/rec.onnx"
    assert url["hashes"][0]["content"] == _sha256(b"onnx")


def test_manifest_does_not_depend_on_download_order(tmp_path: Path) -> None:
    a = _snapshot(tmp_path / "a")
    b = _snapshot(tmp_path / "b")

    with record_downloads() as first:
        record_hf_snapshot("org/a", COMMIT, a, FILES)
        record_hf_snapshot("org/b", COMMIT, b, FILES)
        record_hf_snapshot("org/a", COMMIT, a, FILES)
    with record_downloads() as second:
        record_hf_snapshot("org/b", COMMIT, b, FILES)
        record_hf_snapshot("org/a", COMMIT, a, FILES)

    one = write_cyclonedx(first, tmp_path, "1").read_bytes()
    two = write_cyclonedx(second, tmp_path, "1").read_bytes()
    assert one == two
    assert [c["name"] for c in json.loads(one)["components"]] == ["org/a", "org/b"]


def test_recorded_download_fetches_the_commit_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moving revision is resolved first, and that commit is downloaded."""
    import huggingface_hub

    snapshot = _snapshot(tmp_path / "org--model")
    asked: list[dict] = []

    class FakeApi:
        def model_info(self, repo_id: str, revision: str | None = None):
            assert (repo_id, revision) == ("org/model", None)
            return SimpleNamespace(
                sha=COMMIT,
                siblings=[SimpleNamespace(rfilename=name) for name in FILES],
            )

    def fake_snapshot_download(**kwargs):
        asked.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    with record_downloads() as record:
        hf_model_download.download_hf_model("org/model", local_dir=snapshot)

    assert asked[0]["revision"] == COMMIT
    (model,) = record.hf_models
    assert (model.repo_id, model.commit) == ("org/model", COMMIT)
    # Only the commit's files: not the bookkeeping, not the leftover.
    assert set(model.files) == set(FILES)

    # Without a record, nothing is resolved or recorded.
    hf_model_download.download_hf_model("org/model", local_dir=snapshot)
    assert asked[1]["revision"] is None
    assert model_manifest.current_record() is None


def test_distil_whisper_is_listed_at_its_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    name, (repo_id, filename, revision) = next(
        iter(model_downloader._DISTIL_WHISPER_OPENAI_CHECKPOINTS.items())
    )
    cached = tmp_path / filename
    cached.write_bytes(b"whisper")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_: str(cached))

    with record_downloads() as record:
        model_downloader.download_models(
            output_dir=tmp_path,
            with_layout=False,
            with_tableformer=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_rapidocr=False,
            whisper_distil_models=[name],
        )

    (component,) = json.loads(write_cyclonedx(record, tmp_path, "1").read_text())[
        "components"
    ]
    assert component["purl"] == f"pkg:huggingface/{repo_id}@{revision}"
    assert component["components"] == [
        {
            "type": "file",
            "name": filename,
            "hashes": [{"alg": "SHA-256", "content": _sha256(b"whisper")}],
        }
    ]


def test_a_file_of_the_commit_missing_on_disk_stops_the_record(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "org--model")
    with record_downloads(), pytest.raises(FileNotFoundError, match=r"absent\.bin"):
        record_hf_snapshot("org/model", COMMIT, snapshot, [*FILES, "absent.bin"])
