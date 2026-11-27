# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Recognize and open single MediaWiki source files."""

from __future__ import annotations

import gzip
import tarfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol
from xml.parsers import expat

from docling.exceptions import DocumentLoadError

_HEAD_BYTES = 8192
_GZIP_PROBE_BYTES = 64 * 1024
_WIKITEXT_SUFFIXES = (".wiki", ".wikitext")
_GZIP_MAGIC = b"\x1f\x8b"


class MediaWikiSourceKind(Enum):
    XML = "application/xml"
    WIKITEXT = "text/plain"


class BinaryReader(Protocol):
    def read(self, size: int, /) -> bytes: ...


@dataclass(frozen=True)
class MediaWikiSource:
    name: str
    kind: MediaWikiSourceKind
    stream: BinaryReader


@dataclass(frozen=True)
class GzipProbe:
    mediawiki: MediaWikiSourceKind | None = None
    is_tar: bool = False


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


def is_mediawiki_xml_file(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return is_mediawiki_xml(file.read(_HEAD_BYTES))
    except OSError:
        return False


class _SourceReader:
    def __init__(self, stream: BinaryReader, limit: int) -> None:
        self._stream = stream
        self._limit = self._remaining = limit

    def read(self, size: int, /) -> bytes:
        data = self._stream.read(min(size, self._remaining + 1))
        if len(data) > self._remaining:
            raise DocumentLoadError(
                f"MediaWiki input exceeds max_total_bytes ({self._limit})."
            )
        self._remaining -= len(data)
        return data


def is_wikitext_name(name: str) -> bool:
    return name.lower().endswith(_WIKITEXT_SUFFIXES)


def _source_kind(name: str, head: bytes) -> MediaWikiSourceKind | None:
    if is_mediawiki_xml(head):
        return MediaWikiSourceKind.XML
    if is_wikitext_name(name):
        return MediaWikiSourceKind.WIKITEXT
    return None


def _is_tar_header(head: bytes) -> bool:
    try:
        tarfile.TarInfo.frombuf(head[:512], "utf-8", "surrogateescape")
    except tarfile.HeaderError:
        return False
    return True


def _gzip_payload_name(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def _probe_gzip(raw: BinaryIO, name: str, *, mediawiki_allowed: bool) -> GzipProbe:
    # Use zlib so that a large gzip header, empty members, or a malformed
    # stream cannot cause detection to read beyond the compressed prefix.
    remaining = _GZIP_PROBE_BYTES
    head = bytearray()
    member = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        while remaining and len(head) < _HEAD_BYTES:
            if not (compressed := raw.read(min(1024, remaining))):
                break
            remaining -= len(compressed)
            while compressed and len(head) < _HEAD_BYTES:
                if member.eof:
                    compressed = compressed.lstrip(b"\x00")
                    if not compressed:
                        break
                    member = zlib.decompressobj(16 + zlib.MAX_WBITS)
                head.extend(member.decompress(compressed, _HEAD_BYTES - len(head)))
                compressed = member.unused_data
    except zlib.error:
        return GzipProbe()
    if _is_tar_header(bytes(head)):
        return GzipProbe(is_tar=True)
    if mediawiki_allowed:
        return GzipProbe(mediawiki=_source_kind(_gzip_payload_name(name), bytes(head)))
    return GzipProbe()


def probe_gzip(
    source: Path | BytesIO, name: str, *, mediawiki_allowed: bool
) -> GzipProbe:
    """Classify a gzip prefix without opening an archive or moving the input stream."""
    try:
        with _open_source(source) as raw:
            return _probe_gzip(raw, name, mediawiki_allowed=mediawiki_allowed)
    except OSError:
        return GzipProbe()


@contextmanager
def _open_source(source: Path | BytesIO) -> Iterator[BinaryIO]:
    with source.open("rb") if isinstance(source, Path) else nullcontext(source) as raw:
        position = raw.tell()
        try:
            raw.seek(0)
            yield raw
        finally:
            if isinstance(source, BytesIO):
                source.seek(position)


@contextmanager
def open_mediawiki_source(
    source: Path | BytesIO, name: str, max_total_bytes: int
) -> Iterator[MediaWikiSource]:
    """Open one source and bound bytes read after optional decompression."""
    with _open_source(source) as raw:
        compressed = raw.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
        raw.seek(0)
        if compressed:
            kind = _probe_gzip(raw, name, mediawiki_allowed=True).mediawiki
            name = _gzip_payload_name(name)
        else:
            head = raw.read(_HEAD_BYTES)
            kind = None if _is_tar_header(head) else _source_kind(name, head)
        if kind is None:
            raise DocumentLoadError(
                "Expected one MediaWiki XML export or wikitext file."
            )
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw) if compressed else nullcontext(raw) as data:
            yield MediaWikiSource(name, kind, _SourceReader(data, max_total_bytes))
