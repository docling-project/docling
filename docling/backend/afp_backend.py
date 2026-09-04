# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Minimal AFP MO:DCA backend.

The backend preserves MO:DCA page boundaries and extracts character strings
from PTOCA Transparent Data (TRN) control sequences. AFP is a compound print
format; rendering its IOCA, GOCA, BCOCA, and object-container resources is
deliberately outside this first, dependency-free implementation.
"""

import codecs
import logging
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Iterator

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ProvenanceItem,
    Size,
)
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.backend_options import AfpBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

_MIME_TYPE = "application/vnd.ibm.modcap"
_INTRODUCER = 0x5A
_BASE_INTRODUCER_LENGTH = 8
_MAX_STRUCTURED_FIELD_LENGTH = 32767
_CONTROL_SEQUENCE_PREFIX = b"\x2b\xd3"

_BEGIN_PAGE = b"\xd3\xa8\xaf"
_END_PAGE = b"\xd3\xa9\xaf"
_END_PRESENTATION_TEXT = b"\xd3\xa9\x9b"
_PRESENTATION_TEXT_DATA = b"\xd3\xee\x9b"

# Data-bearing structured fields from content architectures that the MVP does
# not render. Count the data records (rather than every descriptor and bracket)
# so each warning gives a useful measure of omitted content.
_UNSUPPORTED_CONTENT = {
    b"\xd3\xa8\x5f": "page-segment resource",
    b"\xd3\xa8\xdf": "overlay resource",
    b"\xd3\xee\xfb": "image data (IOCA)",
    b"\xd3\xee\x7b": "image raster data (IM)",
    b"\xd3\xee\xbb": "graphics data (GOCA)",
    b"\xd3\xee\xeb": "barcode data (BCOCA)",
    b"\xd3\xee\x92": "object-container data",
}

_TRN_TYPES = {0xDA, 0xDB}
_BEGIN_LINE_TYPES = {0xD8, 0xD9}


class AfpParseError(DocumentLoadError):
    """An AFP structured field or PTOCA control sequence is malformed."""


@dataclass(frozen=True)
class _StructuredField:
    identifier: bytes
    data: bytes
    offset: int


@dataclass
class _AfpPage:
    number: int
    ptoca: bytearray = field(default_factory=bytearray)
    text_blocks: list[str] = field(default_factory=list)


def _iter_structured_fields(content: bytes) -> Iterator[_StructuredField]:
    offset = 0
    while offset < len(content):
        remaining = len(content) - offset
        if remaining < _BASE_INTRODUCER_LENGTH + 1:
            raise AfpParseError(
                f"AFP input ends with a truncated structured-field introducer "
                f"at byte {offset}."
            )
        if content[offset] != _INTRODUCER:
            raise AfpParseError(
                f"Expected AFP structured-field introducer X'5A' at byte {offset}, "
                f"found X'{content[offset]:02X}'."
            )

        length = int.from_bytes(content[offset + 1 : offset + 3], byteorder="big")
        if length < _BASE_INTRODUCER_LENGTH:
            raise AfpParseError(
                f"AFP structured field at byte {offset} declares invalid length "
                f"{length}; the minimum is {_BASE_INTRODUCER_LENGTH}."
            )
        if length > _MAX_STRUCTURED_FIELD_LENGTH:
            raise AfpParseError(
                f"AFP structured field at byte {offset} declares invalid length "
                f"{length}; the maximum is {_MAX_STRUCTURED_FIELD_LENGTH}."
            )
        end = offset + 1 + length
        if end > len(content):
            raise AfpParseError(
                f"AFP structured field at byte {offset} declares {length} bytes, "
                f"but only {remaining - 1} remain."
            )

        identifier = content[offset + 3 : offset + 6]
        flags = content[offset + 6]
        data_start = offset + 9
        if flags & 0x01:  # Structured Field Introducer extension is present.
            if data_start >= end:
                raise AfpParseError(
                    f"AFP structured field at byte {offset} flags an introducer "
                    "extension but does not contain its length."
                )
            extension_length = content[data_start]
            if extension_length < 1 or data_start + extension_length > end:
                raise AfpParseError(
                    f"AFP structured field at byte {offset} has invalid introducer "
                    f"extension length {extension_length}."
                )
            data_start += extension_length

        data_end = end
        if flags & 0x10:  # Padding follows the structured-field data.
            padding_length = content[end - 1]
            if padding_length == 0 and end - data_start >= 3:
                padding_length = int.from_bytes(
                    content[end - 3 : end - 1], byteorder="big"
                )
            if padding_length < 1 or padding_length > end - data_start:
                raise AfpParseError(
                    f"AFP structured field at byte {offset} has invalid padding "
                    f"length {padding_length}."
                )
            data_end -= padding_length

        yield _StructuredField(
            identifier=identifier, data=content[data_start:data_end], offset=offset
        )
        offset = end


def _extract_ptoca_text(data: bytes, encoding: str) -> str:
    """Extract TRN character strings and BLN line boundaries from PTOCA data."""
    chunks: list[str] = []
    offset = 0

    while offset < len(data):
        if data.startswith(_CONTROL_SEQUENCE_PREFIX, offset):
            length_offset = offset + len(_CONTROL_SEQUENCE_PREFIX)
        elif offset + 2 <= len(data) and data[offset + 1] & 0x01:
            # A chained sequence omits X'2BD3' and uses the odd function-type
            # variant. Checking the type byte avoids treating free data as a
            # chained control sequence.
            length_offset = offset
        else:
            offset += 1
            continue

        if length_offset + 2 > len(data):
            raise AfpParseError(
                f"PTOCA data ends inside a control-sequence header at byte {offset}."
            )
        length = data[length_offset]
        if length < 2:
            raise AfpParseError(
                f"PTOCA control sequence at byte {offset} declares invalid length "
                f"{length}; the minimum is 2."
            )
        end = length_offset + length
        if end > len(data):
            raise AfpParseError(
                f"PTOCA control sequence at byte {offset} declares {length} bytes, "
                "but the presentation-text object ends first."
            )

        function_type = data[length_offset + 1]
        parameters = data[length_offset + 2 : end]
        if function_type in _TRN_TYPES:
            chunks.append(parameters.decode(encoding, errors="replace"))
        elif function_type in _BEGIN_LINE_TYPES and chunks and chunks[-1] != "\n":
            chunks.append("\n")

        offset = end

    text = "".join(chunks)
    return "".join(
        char for char in text if char in "\n\t" or unicodedata.category(char) != "Cc"
    )


class AfpDocumentBackend(DeclarativeDocumentBackend, PaginatedDocumentBackend):
    """Convert AFP MO:DCA pages and basic PTOCA text to a DoclingDocument."""

    options: AfpBackendOptions

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: AfpBackendOptions | None = None,
    ) -> None:
        if options is None:
            options = AfpBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.page_range = in_doc.limits.page_range
        try:
            codecs.lookup(options.encoding)
            self.content = (
                path_or_stream.getvalue()
                if isinstance(path_or_stream, BytesIO)
                else path_or_stream.read_bytes()
            )
        except (LookupError, OSError, ValueError) as exc:
            raise DocumentLoadError(
                "Could not initialize the AFP backend for file with hash "
                f"{self.document_hash}; check AfpBackendOptions.encoding."
            ) from exc
        self._pages, self._unsupported = self._parse()

    @override
    def is_valid(self) -> bool:
        if len(self.content) < 9:
            return False
        field_length = int.from_bytes(self.content[1:3], byteorder="big")
        return (
            self.content[0] == _INTRODUCER
            and _BASE_INTRODUCER_LENGTH <= field_length <= _MAX_STRUCTURED_FIELD_LENGTH
            and self.content[3] == 0xD3
        )

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.AFP}

    @override
    def page_count(self) -> int:
        return len(self._pages)

    def _flush_text_object(self, page: _AfpPage) -> None:
        if not page.ptoca:
            return
        text = _extract_ptoca_text(bytes(page.ptoca), self.options.encoding).strip()
        if text:
            page.text_blocks.append(text)
        page.ptoca.clear()

    def _parse(self) -> tuple[list[_AfpPage], Counter[str]]:
        pages: list[_AfpPage] = []
        current_page: _AfpPage | None = None
        unsupported: Counter[str] = Counter()

        def finish_page() -> None:
            nonlocal current_page
            if current_page is not None:
                self._flush_text_object(current_page)
                pages.append(current_page)
                current_page = None

        for structured_field in _iter_structured_fields(self.content):
            identifier = structured_field.identifier
            if identifier == _BEGIN_PAGE:
                if current_page is not None:
                    raise AfpParseError(
                        f"AFP Begin Page at byte {structured_field.offset} occurs "
                        "before the preceding page ends."
                    )
                current_page = _AfpPage(number=len(pages) + 1)
            elif identifier == _PRESENTATION_TEXT_DATA and current_page is not None:
                current_page.ptoca.extend(structured_field.data)
            elif identifier == _END_PRESENTATION_TEXT and current_page is not None:
                self._flush_text_object(current_page)
            elif identifier == _END_PAGE:
                if current_page is None:
                    raise AfpParseError(
                        f"AFP End Page at byte {structured_field.offset} has no "
                        "matching Begin Page."
                    )
                finish_page()
            elif content_type := _UNSUPPORTED_CONTENT.get(identifier):
                unsupported[content_type] += 1

        if current_page is not None:
            raise AfpParseError(
                f"AFP page {current_page.number} has no matching End Page."
            )
        return pages, unsupported

    def _warn_unsupported(self, unsupported: Counter[str]) -> None:
        if not self.options.warn_on_unsupported_content:
            return
        for content_type, count in sorted(unsupported.items()):
            warnings.warn(
                f"Skipped {count} AFP {content_type} structured field(s): this "
                "backend currently extracts PTOCA text and page boundaries but "
                f"does not render {content_type}.",
                UserWarning,
                stacklevel=2,
            )

    @override
    def convert(self) -> DoclingDocument:
        if not self.is_valid():
            raise DocumentLoadError(
                f"Cannot convert AFP document {self.document_hash}: the input does "
                "not start with a valid MO:DCA structured field."
            )

        self._warn_unsupported(self._unsupported)

        origin = DocumentOrigin(
            filename=self.file.name or "file.afp",
            mimetype=_MIME_TYPE,
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        start_page, end_page = self.page_range
        selected_pages = (
            page for page in self._pages if start_page <= page.number <= end_page
        )
        for page in selected_pages:
            doc.add_page(page_no=page.number, size=Size(width=0, height=0))
            parent = doc.add_group(name=f"page-{page.number}", label=GroupLabel.CHAPTER)
            for text_block in page.text_blocks:
                for paragraph in filter(
                    None, (line.strip() for line in text_block.splitlines())
                ):
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=paragraph,
                        parent=parent,
                        prov=ProvenanceItem(
                            page_no=page.number,
                            charspan=(0, len(paragraph)),
                            bbox=BoundingBox(l=0, t=0, r=0, b=0),
                        ),
                    )

        _log.debug("Converted AFP document with %d page(s)", len(doc.pages))
        return doc
