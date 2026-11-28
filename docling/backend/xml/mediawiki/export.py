# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read pages and their latest revisions from a MediaWiki XML export."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from urllib.parse import urljoin, urlsplit
from xml.etree.ElementTree import ParseError
from xml.parsers import expat

from docling.exceptions import DocumentLoadError
from docling.utils.mediawiki_input import BinaryReader, is_mediawiki_xml_name

_XML_CHUNK_BYTES = 64 * 1024
_MAX_XML_DEPTH = 256
_MAX_XML_METADATA_CHARS = 8 * 1024
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class WikiPage:
    title: str
    wikitext: str
    wiki_base: str | None


@dataclass
class _Field:
    name: str
    depth: int
    destination: dict[str, str | None]
    keep: bool
    value: StringIO = field(default_factory=StringIO)
    has_child: bool = False


class _ExportReader:
    def __init__(self, max_page_chars: int) -> None:
        self.pages: list[WikiPage] = []
        self._depth = 0
        self._namespace = ""
        self._field: _Field | None = None
        self._siteinfo: dict[str, str | None] | None = None
        self._page: dict[str, str | None] | None = None
        self._revision: dict[str, str | None] | None = None
        self._slot: dict[str, str | None] | None = None
        self._main_slot: dict[str, str | None] | None = None
        self._wiki_base: str | None = None
        self._latest: tuple[datetime, int] = (_EARLIEST, -1)
        self._wikitext: str | None = None
        self._max_page_chars = max_page_chars
        self._revision_chars = 0

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        self._depth += 1
        if self._depth > _MAX_XML_DEPTH:
            raise DocumentLoadError("MediaWiki XML nesting exceeds the depth limit.")
        if self._depth == 1:
            if not is_mediawiki_xml_name(tag):
                raise DocumentLoadError("Expected a MediaWiki XML export.")
            self._namespace = tag.removesuffix("mediawiki")
            return
        if self._field is not None:
            self._field.has_child = True
            return
        if not tag.startswith(self._namespace):
            return
        name = tag.removeprefix(self._namespace)
        if self._depth == 2:
            if name == "page":
                self._page = {}
                self._latest = (_EARLIEST, -1)
                self._wikitext = None
            elif name == "siteinfo":
                self._siteinfo = {}
        elif self._depth == 3:
            if self._siteinfo is not None and name == "base":
                self._start_field(name, self._siteinfo)
            elif self._page is not None:
                if name == "title":
                    self._page.pop(name, None)
                    self._start_field(name, self._page)
                elif name == "revision":
                    self._revision = {}
                    self._main_slot = None
                    self._revision_chars = 0
        elif self._depth == 4 and self._revision is not None:
            if name == "content":
                self._slot = {}
            elif name in {"timestamp", "id", "model", "text"}:
                self._start_field(name, self._revision, attrs)
        elif self._depth == 5 and self._slot is not None:
            if name in {"role", "model", "text"}:
                self._start_field(name, self._slot, attrs)

    def _start_field(
        self,
        name: str,
        destination: dict[str, str | None],
        attrs: dict[str, str] | None = None,
    ) -> None:
        if name != "text" and name in destination:
            return
        keep = name not in destination and not (
            name == "text"
            and attrs is not None
            and ("deleted" in attrs or "location" in attrs)
        )
        self._field = _Field(name, self._depth, destination, keep)

    def end(self, _tag: str) -> None:
        current = self._field
        if current is not None and self._depth == current.depth:
            current.destination.setdefault(
                current.name, current.value.getvalue() if current.keep else None
            )
            self._field = None
        elif self._depth == 4 and self._slot is not None:
            if self._main_slot is None and self._slot.get("role") == "main":
                self._main_slot = self._slot
            self._slot = None
        elif self._depth == 3 and self._revision is not None:
            self._read_revision(self._revision)
            self._revision = self._main_slot = None
        elif self._depth == 2:
            if self._siteinfo is not None:
                self._read_base(self._siteinfo.get("base") or "")
                self._siteinfo = None
            elif self._page is not None:
                if self._wikitext is not None:
                    self.pages.append(
                        WikiPage(self._title(), self._wikitext, self._wiki_base)
                    )
                self._page = None
                self._wikitext = None
        self._depth -= 1

    def data(self, data: str) -> None:
        current = self._field
        if current is None:
            return
        if current.name == "text":
            self._revision_chars += len(data)
            if self._revision_chars > self._max_page_chars:
                raise DocumentLoadError(
                    f"MediaWiki revision for {self._title()!r} exceeds "
                    f"max_page_chars ({self._max_page_chars})."
                )
        if not current.keep or current.has_child:
            return
        if (
            current.name != "text"
            and current.value.tell() + len(data) > _MAX_XML_METADATA_CHARS
        ):
            raise DocumentLoadError(
                f"MediaWiki XML field {current.name!r} exceeds "
                f"the metadata limit ({_MAX_XML_METADATA_CHARS} characters)."
            )
        current.value.write(data)

    def doctype(self, *_args: object) -> None:
        raise DocumentLoadError("MediaWiki XML exports cannot contain a DTD.")

    def _title(self) -> str:
        assert self._page is not None
        return self._page.get("title") or ""

    def _read_base(self, base: str) -> None:
        base = base.strip()
        try:
            url = urlsplit(base)
        except ValueError:
            return
        if url.scheme in {"http", "https"} and url.netloc:
            self._wiki_base = urljoin(base, "./")

    def _read_revision(self, revision: dict[str, str | None]) -> None:
        stamp = (revision.get("timestamp") or "").strip()
        try:
            timestamp = datetime.fromisoformat(
                stamp.removesuffix("Z") + ("+00:00" if stamp.endswith("Z") else "")
            )
        except ValueError:
            timestamp = _EARLIEST
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        revision_id = (revision.get("id") or "").strip()
        try:
            identifier = int(revision_id)
        except ValueError:
            identifier = -1
        rank = timestamp, identifier
        if rank >= self._latest:
            self._latest = rank
            source = revision if "text" in revision else self._main_slot
            self._wikitext = None
            if source is not None:
                model = (source.get("model") or "").strip()
                if model in {"", "wikitext"}:
                    self._wikitext = source.get("text")


def read_wiki_pages(stream: BinaryReader, max_page_chars: int) -> Iterator[WikiPage]:
    """Stream an export without retaining completed pages or old revisions."""
    target = _ExportReader(max_page_chars)
    # Expat emits events without creating elements or interning unknown names.
    parser = expat.ParserCreate(namespace_separator="}", intern=None)
    parser.StartElementHandler = target.start
    parser.EndElementHandler = target.end
    parser.CharacterDataHandler = target.data
    parser.StartDoctypeDeclHandler = target.doctype
    try:
        while chunk := stream.read(_XML_CHUNK_BYTES):
            parser.Parse(chunk, False)
            yield from target.pages
            target.pages.clear()
        parser.Parse(b"", True)
    except expat.ExpatError as exc:
        raise ParseError(str(exc)) from exc
