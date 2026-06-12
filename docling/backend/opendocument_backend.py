"""Backends to parse OpenDocument formats (ODT, ODS, ODP).

The backends leverage the ``odfdo`` library (https://github.com/jdum/odfdo) to
read the underlying XML structure and translate it into a :class:`DoclingDocument`.

The conventions used here mirror the corresponding Microsoft Office backends
(:mod:`docling.backend.msword_backend`, :mod:`docling.backend.mspowerpoint_backend`,
:mod:`docling.backend.msexcel_backend`).

Known gaps to improve:
- rich text styling is not preserved yet;
- ODP table extraction and ordered-list markers are incomplete;
- ODS conversion focuses on cell/table content rather than spreadsheet rendering
  fidelity.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupLabel,
    ImageRef,
    NodeItem,
    ProvenanceItem,
    RichTableCell,
    Size,
    TableCell,
    TableData,
    TableItem,
)
from PIL import Image as PILImage
from typing_extensions import override

from docling.backend.abstract_backend import (
    DeclarativeDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.backend_options import OdsBackendOptions
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
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: OdsBackendOptions | None = None,
    ) -> None:
        if not _ODFDO_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _ODFDO_IMPORT_ERROR
        super().__init__(in_doc, path_or_stream, options)
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


def _find_true_data_bounds(table: OdfTable) -> tuple[int, int, int, int]:
    """Find the true data boundaries (min/max rows and columns) in an ODS table.

    This function scans all cells to find the smallest rectangular region that contains
    all non-empty cells or merged cell ranges, similar to the Excel backend approach.

    Args:
        table: The ODF table to analyze.

    Returns:
        A tuple (min_row, max_row, min_col, max_col) representing the 0-based indices
        of the data region. If the table is empty, returns (0, 0, 0, 0).
    """
    min_row, min_col = None, None
    max_row, max_col = 0, 0

    # Scan all rows and cells to find non-empty cells
    for row_idx, row in enumerate(table.traverse()):
        for col_idx, cell in enumerate(row.traverse()):
            # Check if cell has content (value or is part of a span)
            if _odf_cell_has_content(cell) or cell.tag == "table:covered-table-cell":
                if min_row is None:
                    min_row = row_idx
                if min_col is None or col_idx < min_col:
                    min_col = col_idx
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)

            # Also check for cells with spans (they define data regions)
            if cell.tag != "table:covered-table-cell":
                attrs = cell.attributes
                row_span = int(attrs.get("table:number-rows-spanned") or 1)
                col_span = int(attrs.get("table:number-columns-spanned") or 1)
                if row_span > 1 or col_span > 1:
                    if min_row is None:
                        min_row = row_idx
                    if min_col is None or col_idx < min_col:
                        min_col = col_idx
                    max_row = max(max_row, row_idx + row_span - 1)
                    max_col = max(max_col, col_idx + col_span - 1)

    # If no data found, return empty bounds
    if min_row is None or min_col is None:
        return (0, 0, 0, 0)

    return (min_row, max_row, min_col, max_col)


def _clean_odf_text_lines(text: str) -> list[str]:
    return [line for line in (line.strip() for line in text.splitlines()) if line]


def _odf_element_text_lines(element: Any) -> list[str]:
    if isinstance(element, OdfList):
        lines: list[str] = []
        for child in element.children:
            if isinstance(child, ListItem):
                lines.extend(_odf_element_text_lines(child))
        return lines

    if isinstance(element, ListItem):
        lines: list[str] = []
        for child in element.children:
            lines.extend(_odf_element_text_lines(child))
        if lines:
            return lines
        return _clean_odf_text_lines(element.text_recursive)

    if isinstance(element, (Header, Paragraph)):
        return _clean_odf_text_lines(element.text_recursive)

    child_lines: list[str] = []
    for child in getattr(element, "children", []):
        child_lines.extend(_odf_element_text_lines(child))
    if child_lines:
        return child_lines

    return _clean_odf_text_lines(element.text_recursive)


def _odf_list_item_content(item: ListItem) -> tuple[str, list[OdfList]]:
    text_parts: list[str] = []
    nested: list[OdfList] = []
    for child in item.children:
        if isinstance(child, OdfList):
            nested.append(child)
        elif isinstance(child, Paragraph):
            text_parts.extend(_clean_odf_text_lines(child.text_recursive))
    if not text_parts:
        text_parts.extend(_clean_odf_text_lines(item.text_recursive))
    return " ".join(text_parts), nested


def _odf_list_has_renderable_content(odf_list: OdfList) -> bool:
    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, nested = _odf_list_item_content(child)
        if text or any(_odf_list_has_renderable_content(item) for item in nested):
            return True
    return False


def _odf_table_has_content(table: OdfTable) -> bool:
    for row in table.traverse():
        for cell in row.traverse():
            if cell.tag == "table:covered-table-cell":
                return True
            if _odf_cell_has_content(cell):
                return True
    return False


def _odf_cell_has_rich_content(cell: Any) -> bool:
    if _odf_cell_has_images(cell):
        return True

    for child in cell.children:
        if isinstance(child, OdfList):
            if _odf_list_has_renderable_content(child):
                return True
        elif isinstance(child, (Header, Paragraph)):
            if _clean_odf_text_lines(child.text_recursive):
                return True
        elif isinstance(child, OdfTable):
            if _odf_table_has_content(child):
                return True

    return False


def _odf_cell_text(cell: Any) -> str:
    if cell.value is not None:
        return str(cell.value)

    lines: list[str] = []
    for child in cell.children:
        lines.extend(_odf_element_text_lines(child))
    if lines:
        return "\n".join(lines)
    if cell.children:
        return ""

    return "\n".join(_clean_odf_text_lines(cell.text_recursive))


def _odf_cell_has_images(cell: Any) -> bool:
    return len(cell.get_images()) > 0


def _odf_cell_has_content(cell: Any) -> bool:
    return _odf_cell_text(cell) != "" or _odf_cell_has_images(cell)


def _odf_cell_is_rich(cell: Any) -> bool:
    return cell.value is None and _odf_cell_has_rich_content(cell)


def _image_ref_from_odf_image(
    odf_obj: OdfDocument | None, image: Any
) -> ImageRef | None:
    image_data: bytes | None = None
    get_data = getattr(image, "get_data", None)
    if callable(get_data):
        image_data = get_data()

    image_url = getattr(image, "url", None)
    if image_data is None and odf_obj is not None and image_url:
        try:
            image_data = odf_obj.get_part(image_url)
        except Exception:
            image_data = None

    if image_data is None and image_url:
        image_path = Path(image_url)
        if image_path.is_file():
            image_data = image_path.read_bytes()

    if image_data is None:
        return None

    pil_image = PILImage.open(BytesIO(image_data))
    return ImageRef.from_pil(image=pil_image, dpi=72)


def _add_odf_images(
    doc: DoclingDocument,
    images: list[Any],
    parent: NodeItem,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    for image in images:
        try:
            image_ref = _image_ref_from_odf_image(odf_obj, image)
        except Exception as e:
            _log.debug("Could not extract OpenDocument image: %s", e)
            image_ref = None
        doc.add_picture(parent=parent, image=image_ref, content_layer=content_layer)


def _add_odf_list(
    doc: DoclingDocument,
    odf_list: OdfList,
    parent: NodeItem,
    content_layer: ContentLayer | None,
    enumerated: bool = False,
) -> None:
    if not _odf_list_has_renderable_content(odf_list):
        return

    list_group = doc.add_list_group(
        name="list", parent=parent, content_layer=content_layer
    )
    counter = 0
    for child in odf_list.children:
        if not isinstance(child, ListItem):
            continue
        text, nested = _odf_list_item_content(child)
        nested = [item for item in nested if _odf_list_has_renderable_content(item)]
        if not text and not nested:
            continue
        counter += 1
        marker = f"{counter}." if enumerated else ""
        item = doc.add_list_item(
            marker=marker,
            enumerated=enumerated,
            parent=list_group,
            text=text,
            content_layer=content_layer,
        )
        for nested_list in nested:
            _add_odf_list(
                doc,
                nested_list,
                parent=item,
                content_layer=content_layer,
                enumerated=enumerated,
            )


def _add_rich_cell_children(
    doc: DoclingDocument,
    cell: Any,
    parent: NodeItem,
    content_layer: ContentLayer | None,
    odf_obj: OdfDocument | None,
) -> None:
    for child in cell.children:
        if isinstance(child, Header):
            text = child.text_recursive.strip()
            if text:
                level = child.get_attribute_integer("text:outline-level") or 1
                doc.add_heading(
                    parent=parent,
                    text=text,
                    level=max(1, level),
                    content_layer=content_layer,
                )
        elif isinstance(child, Paragraph):
            text = child.text_recursive.strip()
            if text:
                doc.add_text(
                    label=DocItemLabel.TEXT,
                    parent=parent,
                    text=text,
                    content_layer=content_layer,
                )
        elif isinstance(child, OdfList):
            _add_odf_list(
                doc,
                child,
                parent=parent,
                content_layer=content_layer,
                enumerated=False,
            )
        elif isinstance(child, OdfTable):
            _add_table_from_odf(
                doc,
                child,
                parent=parent,
                content_layer=content_layer,
                odf_obj=odf_obj,
            )

    _add_odf_images(doc, cell.get_images(), parent, content_layer, odf_obj)


def _add_table_from_odf(
    doc: DoclingDocument,
    table: OdfTable,
    parent: NodeItem | None,
    *,
    min_row: int | None = None,
    max_row: int | None = None,
    min_col: int | None = None,
    max_col: int | None = None,
    prov: ProvenanceItem | None = None,
    content_layer: ContentLayer | None = None,
    odf_obj: OdfDocument | None = None,
) -> TableItem | None:
    if min_row is None or max_row is None or min_col is None or max_col is None:
        min_row, max_row, min_col, max_col = _find_true_data_bounds(table)

    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width == 0 or height == 0:
        return None

    data = TableData(num_rows=height, num_cols=width, table_cells=[])
    table_item = doc.add_table(
        parent=parent,
        data=data,
        prov=prov,
        content_layer=content_layer,
    )

    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue

        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue

            if cell.tag == "table:covered-table-cell":
                continue

            attrs = cell.attributes
            row_span = int(attrs.get("table:number-rows-spanned") or 1)
            col_span = int(attrs.get("table:number-columns-spanned") or 1)
            adjusted_row = row_idx - min_row
            adjusted_col = col_idx - min_col
            text = _odf_cell_text(cell)
            cell_kwargs = {
                "text": text,
                "row_span": row_span,
                "col_span": col_span,
                "start_row_offset_idx": adjusted_row,
                "end_row_offset_idx": adjusted_row + row_span,
                "start_col_offset_idx": adjusted_col,
                "end_col_offset_idx": adjusted_col + col_span,
                "column_header": adjusted_row == 0,
                "row_header": False,
            }

            if _odf_cell_is_rich(cell):
                group = doc.add_group(
                    label=GroupLabel.UNSPECIFIED,
                    name=f"rich_cell_group_{len(doc.tables) - 1}_{adjusted_col}_{adjusted_row}",
                    parent=table_item,
                    content_layer=content_layer,
                )
                _add_rich_cell_children(
                    doc,
                    cell,
                    parent=group,
                    content_layer=content_layer,
                    odf_obj=odf_obj,
                )
                table_cell = RichTableCell(**cell_kwargs, ref=group.get_ref())
            else:
                table_cell = TableCell(**cell_kwargs)

            doc.add_table_cell(table_item=table_item, cell=table_cell)

    return table_item


def _table_region_has_rich_cell(
    table: OdfTable,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
) -> bool:
    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue
        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue
            if cell.tag != "table:covered-table-cell" and _odf_cell_is_rich(cell):
                return True
    return False


def _table_data_from_odf(
    table: OdfTable,
    min_row: int | None = None,
    max_row: int | None = None,
    min_col: int | None = None,
    max_col: int | None = None,
) -> TableData | None:
    """Convert an ODF table to a :class:`TableData` object.

    This function finds the true data boundaries and only processes cells within
    that region, avoiding the inclusion of large numbers of empty cells that may
    exist beyond the actual data.

    Args:
        table: The ODF table to convert.
        min_row: Optional minimum row index (0-based). If None, will be computed.
        max_row: Optional maximum row index (0-based). If None, will be computed.
        min_col: Optional minimum column index (0-based). If None, will be computed.
        max_col: Optional maximum column index (0-based). If None, will be computed.

    Returns ``None`` when the table has no rows or columns.
    """
    # Find the actual data boundaries if not provided
    if min_row is None or max_row is None or min_col is None or max_col is None:
        min_row, max_row, min_col, max_col = _find_true_data_bounds(table)

    # Calculate the dimensions of the actual data region
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    if width == 0 or height == 0:
        return None

    cells: list[TableCell] = []

    # Only process rows and columns within the data bounds
    for row_idx, row in enumerate(table.traverse()):
        if row_idx < min_row or row_idx > max_row:
            continue

        for col_idx, cell in enumerate(row.traverse()):
            if col_idx < min_col or col_idx > max_col:
                continue

            if cell.tag == "table:covered-table-cell":
                # Spanned-over cells are skipped; the anchoring cell carries the span.
                continue

            attrs = cell.attributes
            row_span = int(attrs.get("table:number-rows-spanned") or 1)
            col_span = int(attrs.get("table:number-columns-spanned") or 1)
            text = _odf_cell_text(cell)

            # Adjust cell coordinates to be relative to the data region
            adjusted_row = row_idx - min_row
            adjusted_col = col_idx - min_col

            cells.append(
                TableCell(
                    text=text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=adjusted_row,
                    end_row_offset_idx=adjusted_row + row_span,
                    start_col_offset_idx=adjusted_col,
                    end_col_offset_idx=adjusted_col + col_span,
                    column_header=adjusted_row == 0,
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
                _add_table_from_odf(
                    doc,
                    el,
                    parent=parent,
                    odf_obj=self.odf_obj,
                )
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
                _add_table_from_odf(
                    doc,
                    tbl,
                    parent=parent,
                    odf_obj=self.odf_obj,
                )

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

    Each sheet becomes a separate page. The backend can detect multiple disconnected
    tables within a sheet and optionally treat singleton cells as text items (e.g., titles).
    """

    _odf_type = "spreadsheet"

    @override
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: BytesIO | Path,
        options: OdsBackendOptions | None = None,
    ) -> None:
        if options is None:
            options = OdsBackendOptions()
        super().__init__(in_doc, path_or_stream, options)

    @classmethod
    @override
    def supports_pagination(cls) -> bool:
        return True

    @override
    def page_count(self) -> int:
        if not self.is_valid():
            return 0
        sheet_names_filter: list[str] | None = (
            self.options.sheet_names
            if isinstance(self.options, OdsBackendOptions)
            else None
        )
        if sheet_names_filter is None:
            return len(self.odf_obj.body.tables)
        return sum(
            1 for table in self.odf_obj.body.tables if table.name in sheet_names_filter
        )

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

        sheet_names_filter: list[str] | None = (
            self.options.sheet_names
            if isinstance(self.options, OdsBackendOptions)
            else None
        )

        page_no = 0
        for sheet_idx, table in enumerate(self.odf_obj.body.tables):
            if sheet_names_filter is not None and table.name not in sheet_names_filter:
                _log.debug(f"Skipping sheet {sheet_idx}: {table.name} (filtered out)")
                continue

            page_no += 1
            _log.info(f"Processing sheet {sheet_idx}: {table.name} as page {page_no}")

            # Add page for this sheet
            page = doc.add_page(page_no=page_no, size=Size(width=0, height=0))

            # Determine content layer based on sheet visibility
            content_layer = self._get_sheet_content_layer(table)

            sheet_group = doc.add_group(
                parent=None,
                label=GroupLabel.SECTION,
                name=f"sheet: {table.name}",
                content_layer=content_layer,
            )

            # Convert table data with provenance
            self._convert_sheet_table(doc, table, sheet_group, page_no, content_layer)

            # Extract images from the sheet
            self._find_images_in_sheet(doc, table, sheet_group, page_no, content_layer)

            # Calculate and set page size based on content
            width, height = self._find_page_size(doc, page_no)
            page.size = Size(width=width, height=height)

            _log.debug("Processed ODS sheet %s as page %s", table.name, page_no)
        return doc

    def _convert_sheet_table(
        self,
        doc: DoclingDocument,
        table: OdfTable,
        parent: NodeItem,
        page_no: int,
        content_layer: ContentLayer | None,
    ) -> None:
        """Convert an ODS table and add it to the document with provenance.

        This method finds all disconnected data regions in the sheet and creates
        separate tables for each. Singleton cells can optionally be treated as text.
        """
        # Find all data tables in the sheet
        data_tables = self._find_data_tables_in_sheet(table)

        treat_singleton_as_text = (
            isinstance(self.options, OdsBackendOptions)
            and self.options.treat_singleton_as_text
        )

        for data_table in data_tables:
            min_row, max_row, min_col, max_col = data_table["bounds"]
            table_data = data_table["data"]
            has_rich_content = _table_region_has_rich_cell(
                table, min_row, max_row, min_col, max_col
            )

            # Check if this is a singleton (1x1 table)
            if (
                treat_singleton_as_text
                and len(table_data.table_cells) == 1
                and not has_rich_content
            ):
                # Treat as text item instead of table
                cell = table_data.table_cells[0]
                doc.add_text(
                    text=cell.text,
                    label=DocItemLabel.TEXT,
                    parent=parent,
                    prov=ProvenanceItem(
                        page_no=page_no,
                        charspan=(0, 0),
                        bbox=BoundingBox.from_tuple(
                            (min_col, min_row, max_col + 1, max_row + 1),
                            origin=CoordOrigin.TOPLEFT,
                        ),
                    ),
                    content_layer=content_layer,
                )
            else:
                # Add as table with provenance information
                _add_table_from_odf(
                    doc,
                    table,
                    parent,
                    min_row=min_row,
                    max_row=max_row,
                    min_col=min_col,
                    max_col=max_col,
                    prov=ProvenanceItem(
                        page_no=page_no,
                        charspan=(0, 0),
                        bbox=BoundingBox.from_tuple(
                            (min_col, min_row, max_col + 1, max_row + 1),
                            origin=CoordOrigin.TOPLEFT,
                        ),
                    ),
                    content_layer=content_layer,
                    odf_obj=self.odf_obj,
                )

    def _find_data_tables_in_sheet(
        self, table: OdfTable
    ) -> list[dict[str, tuple[int, int, int, int] | TableData]]:
        """Find all disconnected data tables in an ODS sheet using flood-fill.

        Returns a list of dictionaries, each containing:
        - 'bounds': (min_row, max_row, min_col, max_col)
        - 'data': TableData object
        """
        import collections

        # Get the overall data bounds
        overall_min_row, overall_max_row, overall_min_col, overall_max_col = (
            _find_true_data_bounds(table)
        )

        # Check if we found any data
        if (
            overall_min_row == 0
            and overall_max_row == 0
            and overall_min_col == 0
            and overall_max_col == 0
        ):
            first_cell = table.get_cell("A1")
            if not _odf_cell_has_content(first_cell):
                return []

        GAP_TOLERANCE = cast(OdsBackendOptions, self.options).gap_tolerance
        tables: list[dict[str, tuple[int, int, int, int] | TableData]] = []
        visited: set[tuple[int, int]] = set()

        # Build a map of cell contents for quick lookup
        cell_map: dict[tuple[int, int], bool] = {}
        for row_idx, row in enumerate(table.traverse()):
            for col_idx, cell in enumerate(row.traverse()):
                has_data = _odf_cell_has_content(cell) or (
                    cell.tag == "table:covered-table-cell"
                )
                cell_map[(row_idx, col_idx)] = has_data

        # Helper: Check if a cell has content
        def has_content(r: int, c: int) -> bool:
            if (
                r < overall_min_row
                or r > overall_max_row
                or c < overall_min_col
                or c > overall_max_col
            ):
                return False
            return cell_map.get((r, c), False)

        # Scan for table starts
        for ri in range(overall_min_row, overall_max_row + 1):
            for ci in range(overall_min_col, overall_max_col + 1):
                if (ri, ci) in visited:
                    continue

                if not has_content(ri, ci):
                    continue

                # Found a new table start - use flood fill to find its bounds
                table_cells: set[tuple[int, int]] = set()
                queue = collections.deque([(ri, ci)])
                table_cells.add((ri, ci))

                min_r, max_r = ri, ri
                min_c, max_c = ci, ci

                # Phase 1: Flood Fill
                while queue:
                    curr_r, curr_c = queue.popleft()

                    # Update bounds
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)

                    # Check neighbors in 4 directions with gap tolerance
                    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                    for dr, dc in directions:
                        for step in range(1, GAP_TOLERANCE + 2):
                            nr, nc = curr_r + (dr * step), curr_c + (dc * step)

                            if (nr, nc) in table_cells:
                                break

                            if has_content(nr, nc):
                                table_cells.add((nr, nc))
                                queue.append((nr, nc))
                                break

                # Mark all cells in this table as visited
                visited.update(table_cells)

                # Phase 2: Extract data for this table region
                # Create a sub-table with just this region
                data = _table_data_from_odf(
                    table, min_row=min_r, max_row=max_r, min_col=min_c, max_col=max_c
                )

                if data is not None:
                    tables.append(
                        {"bounds": (min_r, max_r, min_c, max_c), "data": data}
                    )

        return tables

    def _find_images_in_sheet(
        self,
        doc: DoclingDocument,
        table: OdfTable,
        parent: NodeItem,
        page_no: int,
        content_layer: ContentLayer | None,
    ) -> None:
        """Find and extract images from an ODS sheet."""
        try:
            # Get all images in the table
            images = table.get_images()
            for img in images:
                try:
                    # Get the image data
                    image_data = img.get_data()
                    if image_data:
                        # Convert to PIL Image
                        from io import BytesIO

                        pil_image = PILImage.open(BytesIO(image_data))

                        # Try to get position information
                        # ODF images are typically anchored to cells
                        # For now, use a default position
                        anchor = (0, 0, 1, 1)

                        doc.add_picture(
                            parent=parent,
                            image=ImageRef.from_pil(image=pil_image, dpi=72),
                            caption=None,
                            prov=ProvenanceItem(
                                page_no=page_no,
                                charspan=(0, 0),
                                bbox=BoundingBox.from_tuple(
                                    anchor, origin=CoordOrigin.TOPLEFT
                                ),
                            ),
                            content_layer=content_layer,
                        )
                except Exception as e:
                    _log.debug(f"Could not extract image from ODS sheet: {e}")
        except Exception as e:
            _log.debug(f"Could not find images in ODS sheet: {e}")

    @staticmethod
    def _find_page_size(doc: DoclingDocument, page_no: int) -> tuple[float, float]:
        """Calculate page size based on the bounding boxes of all items on the page."""
        left: float = -1.0
        top: float = -1.0
        right: float = -1.0
        bottom: float = -1.0

        for item, _ in doc.iterate_items(traverse_pictures=True, page_no=page_no):
            if not isinstance(item, DocItem):
                continue
            for provenance in item.prov:
                if provenance.bbox is None:
                    continue
                bbox = provenance.bbox
                left = min(left, bbox.l) if left != -1 else bbox.l
                right = max(right, bbox.r) if right != -1 else bbox.r
                top = min(top, bbox.t) if top != -1 else bbox.t
                bottom = max(bottom, bbox.b) if bottom != -1 else bbox.b

        # Return dimensions, defaulting to (0, 0) if no items found
        if left == -1 or right == -1:
            return (0.0, 0.0)
        return (right - left, bottom - top)

    @staticmethod
    def _get_sheet_content_layer(table: OdfTable) -> ContentLayer | None:
        """Determine if a sheet is hidden and should be marked as invisible."""
        # Check if the table has a display attribute indicating it's hidden
        # ODF uses table:display="false" for hidden sheets
        display = table.get_attribute("table:display")
        if display == "false":
            return ContentLayer.INVISIBLE
        return None
