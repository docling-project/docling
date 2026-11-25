# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Record downloaded models and write them as ``models.cdx.json`` (CycloneDX).

SBOM tools that read SBOM files inside an image (syft, used by
``docker build --sbom=true``) then list the models with their repository,
commit or URL, and file digests.
"""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_FILENAME = "models.cdx.json"


@dataclass(frozen=True)
class HfModelDownload:
    repo_id: str
    commit: str
    files: dict[str, Path]  # path in the repository -> local file


@dataclass(frozen=True)
class UrlDownload:
    url: str
    path: Path


@dataclass
class DownloadRecord:
    hf_models: list[HfModelDownload] = field(default_factory=list)
    urls: list[UrlDownload] = field(default_factory=list)


_current_record: ContextVar[Optional[DownloadRecord]] = ContextVar(
    "docling_model_download_record", default=None
)


@contextmanager
def record_downloads() -> Iterator[DownloadRecord]:
    """Collect every model download made inside the ``with`` block."""
    record = DownloadRecord()
    token = _current_record.set(record)
    try:
        yield record
    finally:
        _current_record.reset(token)


def current_record() -> Optional[DownloadRecord]:
    """The active record, or ``None`` when downloads are not recorded."""
    return _current_record.get()


def record_hf_snapshot(
    repo_id: str, commit: str, snapshot_dir: Path, filenames: list[str]
) -> None:
    """Record the files of ``commit``, as listed by the Hub, if recording.

    Files left in ``snapshot_dir`` by an earlier download are not listed.
    """
    record = current_record()
    if record is None:
        return
    files = {name: snapshot_dir / name for name in sorted(filenames)}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{repo_id}@{commit}: not downloaded to {snapshot_dir}: {missing}"
        )
    record.hf_models.append(HfModelDownload(repo_id, commit, files))


def record_hf_file(repo_id: str, commit: str, filename: str, path: Path) -> None:
    """Record one file downloaded from a repository, if recording."""
    record = current_record()
    if record is not None:
        record.hf_models.append(HfModelDownload(repo_id, commit, {filename: path}))


def record_url_file(url: str, path: Path) -> None:
    """Record a model file downloaded from ``url`` to ``path``, if recording."""
    record = current_record()
    if record is not None:
        record.urls.append(UrlDownload(url, path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _CdxHash(BaseModel):
    alg: Literal["SHA-256"] = "SHA-256"
    content: str


class _CdxExternalReference(BaseModel):
    type: Literal["distribution"] = "distribution"
    url: str


class _CdxComponent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["machine-learning-model", "file"]
    bom_ref: Optional[str] = Field(default=None, alias="bom-ref")
    name: str
    version: Optional[str] = None
    purl: Optional[str] = None
    hashes: Optional[list[_CdxHash]] = None
    externalReferences: Optional[list[_CdxExternalReference]] = None
    components: Optional[list["_CdxComponent"]] = None


class _CdxTool(BaseModel):
    type: Literal["application"] = "application"
    name: str
    version: str


class _CdxTools(BaseModel):
    components: list[_CdxTool]


class _CdxMetadata(BaseModel):
    tools: _CdxTools


class _CdxBom(BaseModel):
    bomFormat: Literal["CycloneDX"] = "CycloneDX"
    specVersion: Literal["1.6"] = "1.6"
    version: int = 1
    metadata: _CdxMetadata
    components: list[_CdxComponent]


def _hf_purl(repo_id: str, commit: str) -> str:
    return f"pkg:huggingface/{repo_id}@{commit.lower()}"


def _hf_component(model: HfModelDownload) -> _CdxComponent:
    purl = _hf_purl(model.repo_id, model.commit)
    return _CdxComponent(
        type="machine-learning-model",
        bom_ref=purl,
        name=model.repo_id,
        version=model.commit,
        purl=purl,
        externalReferences=[
            _CdxExternalReference(
                url=f"https://huggingface.co/{model.repo_id}/tree/{model.commit}"
            )
        ],
        components=[
            _CdxComponent(
                type="file",
                name=name,
                hashes=[_CdxHash(content=sha256_file(path))],
            )
            for name, path in sorted(model.files.items())
        ],
    )


def _url_component(download: UrlDownload) -> _CdxComponent:
    return _CdxComponent(
        type="machine-learning-model",
        bom_ref=download.url,
        name=download.url.rsplit("/", 1)[-1],
        hashes=[_CdxHash(content=sha256_file(download.path))],
        externalReferences=[_CdxExternalReference(url=download.url)],
    )


def _component_key(component: _CdxComponent) -> tuple[int, str, str]:
    """Hugging Face models first, by repository and commit, then URL downloads."""
    if component.purl is not None:
        return (0, component.name, component.version or "")
    return (1, component.bom_ref or component.name, "")


def _read_existing(path: Path) -> list[_CdxComponent]:
    """The components of an earlier ``models.cdx.json`` at ``path``, if any."""
    if not path.exists():
        return []
    try:
        return _CdxBom.model_validate_json(path.read_text(encoding="utf-8")).components
    except ValueError as error:
        raise ValueError(
            f"{path} exists but is not a list of models written by docling; "
            f"move or delete it: {error}"
        ) from error


def write_cyclonedx(
    record: DownloadRecord, output_dir: Path, tool_version: str
) -> Path:
    """Write ``record`` to ``output_dir/models.cdx.json`` and return the path.

    Models listed by an earlier download into the same directory stay listed,
    and a model downloaded again replaces its entry: a Hugging Face model by
    repository, a file by its URL. Duplicates are merged and components sorted,
    and there is no serial number or timestamp, so the same models always give
    the same file.
    """
    hf_models: dict[tuple[str, str], HfModelDownload] = {}
    for model in record.hf_models:
        key = (model.repo_id, model.commit)
        if key in hf_models:
            files = {**hf_models[key].files, **model.files}
            model = HfModelDownload(model.repo_id, model.commit, files)
        hf_models[key] = model
    urls = {download.url: download for download in record.urls}

    components = [_hf_component(model) for model in hf_models.values()]
    components += [_url_component(download) for download in urls.values()]
    path = output_dir / MANIFEST_FILENAME
    listed = {repo_id for repo_id, _ in hf_models} | set(urls)
    components += [
        component
        for component in _read_existing(path)
        if (component.name if component.purl else component.bom_ref) not in listed
    ]

    bom = _CdxBom(
        metadata=_CdxMetadata(
            tools=_CdxTools(components=[_CdxTool(name="docling", version=tool_version)])
        ),
        components=sorted(components, key=_component_key),
    )
    path.write_text(
        json.dumps(bom.model_dump(by_alias=True, exclude_none=True), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
