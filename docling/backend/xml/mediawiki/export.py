# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read pages and their latest revisions from a MediaWiki XML export."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit
from xml.etree.ElementTree import Element, TreeBuilder, XMLParser

from docling.exceptions import DocumentLoadError
from docling.utils.mediawiki_input import BinaryReader, is_mediawiki_xml_name

_XML_CHUNK_BYTES = 64 * 1024
_MAX_XML_DEPTH = 256
_EARLIEST = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class WikiPage:
    title: str
    wikitext: str
    wiki_base: str | None


class _ExportReader:
    def __init__(self) -> None:
        self.pages: list[WikiPage] = []
        self._stack: list[str] = []
        self._namespace = ""
        self._capture_depth = 0
        self._tree: TreeBuilder | None = None
        self._title = ""
        self._wiki_base: str | None = None
        self._latest: tuple[datetime, int] = (_EARLIEST, -1)
        self._wikitext: str | None = None

    def start(self, tag: str, attrs: dict[str, str]) -> None:
        if not self._stack:
            if not is_mediawiki_xml_name(tag):
                raise DocumentLoadError("Expected a MediaWiki XML export.")
            self._namespace = tag.removesuffix("mediawiki")
        self._stack.append(tag)
        depth = len(self._stack)
        if depth > _MAX_XML_DEPTH:
            raise DocumentLoadError("MediaWiki XML nesting exceeds the depth limit.")

        if self._tree is None:
            if depth == 2 and tag == self._name("page"):
                self._title = ""
                self._latest = (_EARLIEST, -1)
                self._wikitext = None
            elif (depth == 2 and tag == self._name("siteinfo")) or (
                depth == 3
                and self._stack[-2] == self._name("page")
                and tag in (self._name("title"), self._name("revision"))
            ):
                self._tree = TreeBuilder()
                self._capture_depth = depth
        if self._tree is not None:
            self._tree.start(tag, attrs)

    def end(self, tag: str) -> None:
        depth = len(self._stack)
        if self._tree is not None:
            self._tree.end(tag)
            if depth == self._capture_depth:
                element = self._tree.close()
                self._tree = None
                self._capture(element)
        elif depth == 2 and tag == self._name("page") and self._wikitext is not None:
            self.pages.append(WikiPage(self._title, self._wikitext, self._wiki_base))
        self._stack.pop()

    def data(self, data: str) -> None:
        if self._tree is not None:
            self._tree.data(data)

    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise DocumentLoadError("MediaWiki XML exports cannot contain a DTD.")

    def close(self) -> None:
        pass

    def _name(self, local_name: str) -> str:
        return self._namespace + local_name

    def _capture(self, element: Element) -> None:
        if element.tag == self._name("title"):
            self._title = element.text or ""
        elif element.tag == self._name("siteinfo"):
            base = element.findtext(self._name("base"), "").strip()
            try:
                url = urlsplit(base)
            except ValueError:
                return
            if url.scheme in {"http", "https"} and url.netloc:
                self._wiki_base = urljoin(base, "./")
        else:
            self._read_revision(element)

    def _read_revision(self, revision: Element) -> None:
        stamp = revision.findtext(self._name("timestamp"), "").strip()
        try:
            timestamp = datetime.fromisoformat(
                stamp.removesuffix("Z") + ("+00:00" if stamp.endswith("Z") else "")
            )
        except ValueError:
            timestamp = _EARLIEST
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        revision_id = revision.findtext(self._name("id"), "").strip()
        try:
            identifier = int(revision_id)
        except ValueError:
            identifier = -1
        rank = timestamp, identifier
        if rank >= self._latest:
            self._latest = rank
            self._wikitext = self._revision_text(revision)

    def _revision_text(self, revision: Element) -> str | None:
        source = revision
        if revision.find(self._name("text")) is None:
            for content in revision.findall(self._name("content")):
                if content.findtext(self._name("role")) == "main":
                    source = content
                    break
        model = source.findtext(self._name("model"), "").strip()
        text = source.find(self._name("text"))
        if (
            model not in {"", "wikitext"}
            or text is None
            or "deleted" in text.attrib
            or "location" in text.attrib
        ):
            return None
        return text.text or ""


def read_wiki_pages(stream: BinaryReader) -> Iterator[WikiPage]:
    """Stream an export without retaining completed pages or old revisions."""
    target = _ExportReader()
    parser = XMLParser(target=target)
    while chunk := stream.read(_XML_CHUNK_BYTES):
        parser.feed(chunk)
        yield from target.pages
        target.pages.clear()
    parser.close()
