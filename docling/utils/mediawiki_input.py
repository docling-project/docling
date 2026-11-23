# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recognize and open MediaWiki source files and archive members."""

from __future__ import annotations

import gzip
import tarfile
import zlib
from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from io import BufferedReader, BytesIO, RawIOBase
from pathlib import Path
from typing import Protocol
from xml.parsers import expat

from typing_extensions import Buffer

from docling.exceptions import DocumentLoadError

_HEAD_BYTES = 8192
_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz")
_WIKITEXT_SUFFIXES = (".wiki", ".wikitext")
_GZIP_MAGIC = b"\x1f\x8b"
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10_000


class MediaWikiSourceKind(Enum):
    XML = "application/xml"
    WIKITEXT = "text/plain"


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class MediaWikiSource:
    name: str
    kind: MediaWikiSourceKind
    stream: BinaryReader
    mimetype: str
    archive_member: bool = False


class _XmlRootFound(Exception):
    pass


def is_mediawiki_xml_name(name: str) -> bool:
    namespace, _, local_name = name.rpartition("}")
    namespace = namespace.removeprefix("{")
    return local_name == "mediawiki" and (
        not namespace
        or namespace.startswith(
            (
                "http://www.mediawiki.org/xml/export-",
                "https://www.mediawiki.org/xml/export-",
            )
        )
    )


def is_mediawiki_xml(head: bytes) -> bool:
    if b"<" not in head:
        return False
    parser = expat.ParserCreate(namespace_separator="}")

    def stop_at_root(name: str, *_args: object) -> None:
        raise _XmlRootFound(name)

    parser.StartElementHandler = stop_at_root
    # Stop before processing a DTD. The export parser reports it as invalid.
    parser.StartDoctypeDeclHandler = stop_at_root
    try:
        parser.Parse(head, False)
    except _XmlRootFound as exc:
        return is_mediawiki_xml_name(str(exc))
    except expat.ExpatError:
        return False
    return False


class _SourceReader(RawIOBase):
    """Bound decompression, including bytes skipped by tarfile."""

    def __init__(self, stream: BinaryReader, limit: int | None) -> None:
        super().__init__()
        self.stream = stream
        self.remaining = limit

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer) -> int:
        view = memoryview(buffer).cast("B")
        if self.remaining is None:
            data = self.stream.read(len(view))
            view[: len(data)] = data
            return len(data)
        data = self.stream.read(min(len(view), self.remaining + 1))
        if len(data) > self.remaining:
            raise DocumentLoadError(
                f"MediaWiki archive exceeds the {_MAX_ARCHIVE_BYTES}-byte limit."
            )
        view[: len(data)] = data
        self.remaining -= len(data)
        return len(data)


def _source_kind(name: str, head: bytes) -> MediaWikiSourceKind | None:
    if is_mediawiki_xml(head):
        return MediaWikiSourceKind.XML
    if name.lower().endswith(_WIKITEXT_SUFFIXES):
        return MediaWikiSourceKind.WIKITEXT
    return None


def _tar_sources(
    archive: BufferedReader[_SourceReader],
) -> Generator[MediaWikiSource, None, None]:
    with tarfile.open(fileobj=archive, mode="r|") as tar:
        for index, member in enumerate(tar, start=1):
            if index > _MAX_ARCHIVE_MEMBERS:
                raise DocumentLoadError(
                    f"MediaWiki archive exceeds the {_MAX_ARCHIVE_MEMBERS}-member limit."
                )
            if member.size > _MAX_ARCHIVE_BYTES:
                raise DocumentLoadError(
                    f"MediaWiki archive member {member.name!r} exceeds the archive size limit."
                )
            if not member.isreg():
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            # tarfile's streaming member has read(), but no readinto().
            with (
                source,
                BufferedReader(_SourceReader(source, _MAX_ARCHIVE_BYTES)) as payload,
            ):
                kind = _source_kind(member.name, payload.peek(_HEAD_BYTES))
                if kind is not None:
                    yield MediaWikiSource(
                        member.name,
                        kind,
                        payload,
                        "application/x-tar",
                        archive_member=True,
                    )


@contextmanager
def open_mediawiki_sources(
    source: Path | BytesIO, name: str
) -> Iterator[Iterator[MediaWikiSource]]:
    """Open sources for sequential reading without extracting to the filesystem."""
    with source.open("rb") if isinstance(source, Path) else nullcontext(source) as raw:
        position = raw.tell()
        try:
            raw.seek(0)
            raw_head = raw.read(512)
            compressed = raw_head.startswith(_GZIP_MAGIC)
            archive = raw_head[257:262] == b"ustar" or name.lower().endswith(
                _TAR_SUFFIXES
            )
            raw.seek(0)
            with gzip.GzipFile(fileobj=raw) if compressed else nullcontext(raw) as data:
                limit = _MAX_ARCHIVE_BYTES if compressed or archive else None
                with BufferedReader(_SourceReader(data, limit)) as reader:
                    head = reader.peek(_HEAD_BYTES)
                    if archive or (compressed and head[257:262] == b"ustar"):
                        members = _tar_sources(reader)
                        try:
                            yield members
                        finally:
                            members.close()
                        return
                    payload_name = name
                    if compressed and name.lower().endswith(".gz"):
                        payload_name = name[:-3]
                    kind = _source_kind(payload_name, head)
                    if kind is None:
                        yield iter(())
                    else:
                        yield iter(
                            (MediaWikiSource(payload_name, kind, reader, kind.value),)
                        )
        finally:
            if isinstance(source, BytesIO):
                source.seek(position)


def is_mediawiki_archive(source: Path | BytesIO, name: str) -> bool:
    try:
        with open_mediawiki_sources(source, name) as members:
            return next(members, None) is not None
    except (DocumentLoadError, OSError, EOFError, tarfile.TarError, zlib.error):
        return False


def is_wikitext_name(name: str) -> bool:
    return name.lower().endswith(_WIKITEXT_SUFFIXES)
