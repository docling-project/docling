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


OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


def test_a_later_download_into_the_directory_adds_its_models(tmp_path: Path) -> None:
    a = _snapshot(tmp_path / "a")
    b = _snapshot(tmp_path / "b")
    ocr_file = tmp_path / "rec.onnx"
    ocr_file.write_bytes(b"onnx")

    with record_downloads() as first:
        record_hf_snapshot("org/a", COMMIT, a, FILES)
        record_url_file("https://example.org/rec.onnx", ocr_file)
    write_cyclonedx(first, tmp_path, "1")
    with record_downloads() as second:
        record_hf_snapshot("org/b", COMMIT, b, FILES)
    path = write_cyclonedx(second, tmp_path, "1")

    names = [c["name"] for c in json.loads(path.read_text())["components"]]
    assert names == ["org/a", "org/b", "rec.onnx"]

    # The same models in one run give the same file.
    with record_downloads() as both:
        record_hf_snapshot("org/b", COMMIT, b, FILES)
        record_hf_snapshot("org/a", COMMIT, a, FILES)
        record_url_file("https://example.org/rec.onnx", ocr_file)
    once = tmp_path / "once"
    once.mkdir()
    assert write_cyclonedx(both, once, "1").read_bytes() == path.read_bytes()


def test_a_model_downloaded_again_replaces_its_entry(tmp_path: Path) -> None:
    a = _snapshot(tmp_path / "a")
    ocr_file = tmp_path / "rec.onnx"
    ocr_file.write_bytes(b"onnx")

    with record_downloads() as first:
        record_hf_snapshot("org/a", COMMIT, a, FILES)
        record_url_file("https://example.org/rec.onnx", ocr_file)
    write_cyclonedx(first, tmp_path, "1")
    ocr_file.write_bytes(b"onnx v2")
    with record_downloads() as second:
        record_hf_snapshot("org/a", OTHER_COMMIT, a, FILES)
        record_url_file("https://example.org/rec.onnx", ocr_file)
    path = write_cyclonedx(second, tmp_path, "1")

    hf, url = json.loads(path.read_text())["components"]
    assert hf["version"] == OTHER_COMMIT
    assert url["hashes"][0]["content"] == _sha256(b"onnx v2")


def test_an_existing_file_that_is_not_a_model_list_stops_the_write(
    tmp_path: Path,
) -> None:
    (tmp_path / "models.cdx.json").write_text('{"bomFormat": "SPDX"}')
    with record_downloads() as record:
        record_hf_snapshot("org/a", COMMIT, _snapshot(tmp_path / "a"), FILES)

    with pytest.raises(ValueError, match=r"models\.cdx\.json"):
        write_cyclonedx(record, tmp_path, "1")


def test_an_offline_recorded_download_points_to_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import OfflineModeIsEnabled

    class OfflineApi:
        def model_info(self, *args, **kwargs):
            raise OfflineModeIsEnabled("offline")

    monkeypatch.setattr(huggingface_hub, "HfApi", OfflineApi)

    with (
        record_downloads(),
        pytest.raises(OfflineModeIsEnabled, match=r"--no-manifest"),
    ):
        hf_model_download.download_hf_model("org/model", local_dir=tmp_path)
