"""Backends to parse OpenDocument formats (ODT, ODS, ODP).

The backends leverage the ``odfdo`` library (https://github.com/jdum/odfdo) to
read the underlying XML structure and translate it into a :class:`DoclingDocument`.

The conventions used here mirror the corresponding Microsoft Office backends
(:mod:`docling.backend.msword_backend`, :mod:`docling.backend.mspowerpoint_backend`,
:mod:`docling.backend.msexcel_backend`).
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    NodeItem,
    TableCell,
    TableData,
)
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_ODFDO_AVAILABLE: bool = False
_ODFDO_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from odfdo import (
        Document as OdfDocument,
        DrawPage,
        Frame,
        Header,
        List as OdfList,
        ListItem,
        Paragraph,
        Table as OdfTable,
    )

    _ODFDO_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _ODFDO_IMPORT_ERROR = e

_log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "The 'odfdo' package is required to process OpenDocument files. "
    "Install it with `pip install 'docling[opendocument]'`."
)


def _load_odf_document(
    path_or_stream: BytesIO | Path, document_hash: str
) -> OdfDocument:
    """Load an ODF document from a path or in-memory stream."""
    try:
        if isinstance(path_or_stream, BytesIO):
            return OdfDocument(path_or_stream)
        return OdfDocument(str(path_or_stream))
    except Exception as e:
        raise RuntimeError(
            f"OpenDocument backend could not load document with hash {document_hash}"
        ) from e


class _OdfBaseBackend(DeclarativeDocumentBackend):
    """Shared loading / validation logic for ODT, ODS and ODP backends."""

    _odf_type: str = ""  # "text", "spreadsheet" or "presentation"

    @override
    def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path) -> None:
        if not _ODFDO_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _ODFDO_IMPORT_ERROR
        super().__init__(in_doc, path_or_stream)
        self.path_or_stream: BytesIO | Path = path_or_stream
        self.valid: bool = False
        self.odf_obj: OdfDocument = _load_odf_document(
            path_or_stream, self.document_hash
        )
        if self._odf_type and self.odf_obj.get_type() != self._odf_type:
            raise RuntimeError(
                f"Expected an OpenDocument {self._odf_type!r} but got "
                f"{self.odf_obj.get_type()!r}"
            )
        self.valid = True

    @override
    def is_valid(self) -> bool:
        return self.valid

    @override
    def unload(self):
        if isinstance(self.path_or_stream, BytesIO):
            self.path_or_stream.close()
        self.path_or_stream = None


def _table_data_from_odf(table: OdfTable) -> TableData | None:
    """Convert an ODF table to a :class:`TableData` object.

    Returns ``None`` when the table has no rows or columns.
    """
    width, height = table.width, table.height
    if width == 0 or height == 0:
        return None

    cells: list[TableCell] = []
    for row in table.traverse():
        for cell in row.traverse():
            if cell.tag == "table:covered-table-cell":
                # Spanned-over cells are skipped; the anchoring cell carries the span.
                continue
            attrs = cell.attributes
            row_span = int(attrs.get("table:number-rows-spanned") or 1)
            col_span = int(attrs.get("table:number-columns-spanned") or 1)
            text = "" if cell.value is None else str(cell.value)
            cells.append(
                TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=cell.y,
                    end_row_offset_idx=cell.y + row_span,
                    start_col_offset_idx=cell.x,
                    end_col_offset_idx=cell.x + col_span,
                    column_header=cell.y == 0,
                    row_header=False,
                )
            )

    return TableData(num_rows=height, num_cols=width, table_cells=cells)


class OdtDocumentBackend(_OdfBaseBackend):
    """Backend for OpenDocument Text (``.odt``) files."""

    _odf_type = "text"

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODT}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.text",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        self._walk(self.odf_obj.body.children, parent=None, doc=doc)
        return doc

    def _walk(
        self,
        elements: list[Any],
        parent: NodeItem | None,
        doc: DoclingDocument,
    ) -> None:
        for el in elements:
            if isinstance(el, Header):
                level = el.get_attribute_integer("text:outline-level") or 1
                text = el.text_recursive.strip()
                if text:
                    doc.add_heading(parent=parent, text=text, level=max(1, level))
            elif isinstance(el, Paragraph) and not isinstance(el, Header):
                text = el.text_recursive.strip()
                if text:
                    doc.add_text(label=DocItemLabel.TEXT, parent=parent, text=text)
            elif isinstance(el, OdfList):
                self._walk_list(el, parent=parent, doc=doc, enumerated=False)
            elif isinstance(el, OdfTable):
                data = _table_data_from_odf(el)
                if data is not None:
                    doc.add_table(parent=parent, data=data)
            else:
                _log.debug("Ignoring ODT element with tag: %s", el.tag)

    def _walk_list(
        self,
        odf_list: OdfList,
        parent: NodeItem | None,
        doc: DoclingDocument,
        enumerated: bool,
    ) -> None:
        list_group = doc.add_list_group(name="list", parent=parent)
        counter = 0
        for child in odf_list.children:
            if not isinstance(child, ListItem):
                continue
            counter += 1
            marker = f"{counter}." if enumerated else ""
            text_parts: list[str] = []
            nested: list[OdfList] = []
            for sub in child.children:
                if isinstance(sub, OdfList):
                    nested.append(sub)
                elif isinstance(sub, Paragraph):
                    text_parts.append(sub.text_recursive)
            text = " ".join(t for t in (s.strip() for s in text_parts) if t)
            item = doc.add_list_item(
                marker=marker,
                enumerated=enumerated,
                parent=list_group,
                text=text,
            )
            for nested_list in nested:
                self._walk_list(
                    nested_list, parent=item, doc=doc, enumerated=enumerated
                )


class OdpDocumentBackend(_OdfBaseBackend, PaginatedDocumentBackend):
    """Backend for OpenDocument Presentation (``.odp``) files."""

    _odf_type = "presentation"

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if not self.is_valid():
            return 0
        return len(self.odf_obj.body.get_draw_pages())

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODP}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.presentation",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        for slide_idx, page in enumerate(self.odf_obj.body.get_draw_pages()):
            slide_name = page.name or f"slide-{slide_idx + 1}"
            slide_group = doc.add_group(
                name=f"slide-{slide_idx}",
                label=GroupLabel.CHAPTER,
                parent=None,
            )
            doc.add_text(
                label=DocItemLabel.TITLE,
                parent=slide_group,
                text=slide_name,
            )
            self._walk_slide(page, parent=slide_group, doc=doc)
        return doc

    def _walk_slide(
        self, page: DrawPage, parent: NodeItem, doc: DoclingDocument
    ) -> None:
        for frame in page.get_frames():
            for tbl in frame.get_elements("descendant::table:table"):
                data = _table_data_from_odf(tbl)
                if data is not None:
                    doc.add_table(parent=parent, data=data)

            for textbox in frame.get_elements("descendant::draw:text-box"):
                self._walk_textbox_children(textbox.children, parent=parent, doc=doc)

    def _walk_textbox_children(
        self,
        elements: list[Any],
        parent: NodeItem,
        doc: DoclingDocument,
    ) -> None:
        for el in elements:
            if isinstance(el, Header):
                text = el.text_recursive.strip()
                if text:
                    level = el.get_attribute_integer("text:outline-level") or 1
                    doc.add_heading(parent=parent, text=text, level=max(1, level))
            elif isinstance(el, Paragraph):
                text = el.text_recursive.strip()
                if text:
                    doc.add_text(label=DocItemLabel.TEXT, parent=parent, text=text)
            elif isinstance(el, OdfList):
                list_group = doc.add_list_group(name="list", parent=parent)
                for child in el.children:
                    if not isinstance(child, ListItem):
                        continue
                    text_parts = [
                        p.text_recursive
                        for p in child.children
                        if isinstance(p, Paragraph)
                    ]
                    text = " ".join(t for t in (s.strip() for s in text_parts) if t)
                    doc.add_list_item(
                        marker="",
                        enumerated=False,
                        parent=list_group,
                        text=text,
                    )


class OdsDocumentBackend(_OdfBaseBackend, PaginatedDocumentBackend):
    """Backend for OpenDocument Spreadsheet (``.ods``) files.

    Each sheet becomes a separate page; sheet contents are emitted as a single
    table grouped under a section group named after the sheet.
    """

    _odf_type = "spreadsheet"

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if not self.is_valid():
            return 0
        return len(self.odf_obj.body.tables)

    @classmethod
    @override
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.ODS}

    @override
    def convert(self) -> DoclingDocument:
        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="application/vnd.oasis.opendocument.spreadsheet",
            binary_hash=self.document_hash,
        )
        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)
        if not self.is_valid():
            raise RuntimeError(
                f"Cannot convert doc with {self.document_hash} because the backend failed to init."
            )

        for sheet_idx, table in enumerate(self.odf_obj.body.tables, start=1):
            sheet_group = doc.add_group(
                parent=None,
                label=GroupLabel.SECTION,
                name=f"sheet: {table.name}",
            )
            data = _table_data_from_odf(table)
            if data is not None:
                doc.add_table(parent=sheet_group, data=data)
            _log.debug("Processed ODS sheet %s as page %s", table.name, sheet_idx)
        return doc
