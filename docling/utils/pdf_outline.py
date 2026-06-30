"""Extract a PDF's outline (bookmarks / table-of-contents).

The outline, when present, is the most authoritative heading-hierarchy signal in a PDF. Two
extractors are provided so each backend uses its own native capability:

* :func:`extract_outline_from_pdfium` -- for the pypdfium2 backend. Returns the richest data:
  title, depth, target page and vertical position.
* :func:`outline_from_docling_parse` -- for the docling-parse backends, using their native
  ``get_table_of_contents()`` (no pypdfium2 dependency). The native outline carries titles and
  hierarchy only, so page number and position are left unset.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pydantic import BaseModel
from pypdfium2._helpers.misc import PdfiumError

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling_parse.pdf_parser import (
        PdfDocument as DoclingParsePdfDocument,
        PdfTableOfContents,
    )

_log = logging.getLogger(__name__)


class _PdfOutlineItem(BaseModel):
    """A single PDF bookmark / table-of-contents entry (internal).

    Internal data-passing structure between a PDF backend's ``get_document_outline()`` and the
    heading-hierarchy stage; not part of the public datamodel or the serialized output. The list
    is kept flat and in document order; each entry carries its own ``level`` so no tree structure
    is needed for matching.
    """

    title: str
    # 0-based depth as reported by the PDF outline; compressed to contiguous levels downstream.
    level: int
    # 1-based target page; None when the entry has no resolvable page (e.g. docling-parse ToC).
    page_no: int | None = None
    # Top-left-origin vertical position of the target, when derivable from the destination view.
    y_top: float | None = None


# Destination view modes whose coordinates carry a usable vertical (top) position, mapped to
# the index of that coordinate within the position tuple. Coordinates are in PDF space
# (bottom-left origin). Modes not listed (FIT, FITV, FITB, FITBV, unknown) provide no top.
_VIEW_TOP_INDEX = {
    pdfium_c.PDFDEST_VIEW_XYZ: 1,  # [x, y, zoom]
    pdfium_c.PDFDEST_VIEW_FITH: 0,  # [y]
    pdfium_c.PDFDEST_VIEW_FITBH: 0,  # [y]
    pdfium_c.PDFDEST_VIEW_FITR: 3,  # [left, bottom, right, top]
}


def _dest_top_pdf(dest: pdfium.PdfDest) -> tuple[int | None, float | None]:
    """Return ``(0-based page index, vertical top in PDF bottom-left coords)`` for a dest.

    Either element may be ``None`` when the destination does not encode it.
    """
    page_index = dest.get_index()
    mode, pos = dest.get_view()
    idx = _VIEW_TOP_INDEX.get(mode)
    y_pdf = pos[idx] if idx is not None and idx < len(pos) else None
    return page_index, y_pdf


def extract_outline_from_pdfium(pdoc: pdfium.PdfDocument) -> list[_PdfOutlineItem]:
    """Extract the outline as a flat, document-ordered list of :class:`_PdfOutlineItem`.

    Vertical positions are converted to top-left origin (matching ``DocItem`` provenance) using
    the target page height. Returns an empty list when the document has no outline or it cannot
    be read.
    """
    items: list[_PdfOutlineItem] = []
    page_heights: dict[int, float] = {}

    with pypdfium2_lock:
        try:
            toc = list(pdoc.get_toc())
        except PdfiumError as exc:
            _log.debug("Could not read PDF outline: %s", exc)
            return []

        for bm in toc:
            title = (bm.get_title() or "").strip()
            if not title:
                continue

            page_no: int | None = None
            y_top: float | None = None
            try:
                dest = bm.get_dest()
            except PdfiumError:
                dest = None
            if dest is not None:
                page_index, y_pdf = _dest_top_pdf(dest)
                if page_index is not None:
                    page_no = page_index + 1
                    if y_pdf is not None:
                        if page_index not in page_heights:
                            page = pdoc[page_index]
                            page_heights[page_index] = page.get_height()
                            page.close()
                        y_top = page_heights[page_index] - y_pdf

            items.append(
                _PdfOutlineItem(
                    title=title, level=int(bm.level), page_no=page_no, y_top=y_top
                )
            )

    return items


def outline_from_docling_parse(
    dp_doc: DoclingParsePdfDocument,
) -> list[_PdfOutlineItem]:
    """Flatten docling-parse's native table-of-contents into ordered ``_PdfOutlineItem``\\ s.

    Walks the ``PdfTableOfContents`` tree returned by ``PdfDocument.get_table_of_contents()``,
    depth-first, assigning each node a 0-based ``level`` from its depth (top-level entries at
    level 0, matching the pypdfium2 extractor). The native outline exposes only the title and
    the tree structure -- no target page or vertical position -- so ``page_no`` and ``y_top``
    are left ``None`` and the heading matcher falls back to title-only matching.

    ``get_table_of_contents()`` returns ``None`` for PDFs without an embedded outline, in which
    case an empty list is returned.
    """
    toc = dp_doc.get_table_of_contents()
    if toc is None:
        return []

    items: list[_PdfOutlineItem] = []

    def _walk(node: PdfTableOfContents, level: int) -> None:
        for child in node.children or []:
            title = (child.text or child.orig or "").strip()
            if title:
                items.append(_PdfOutlineItem(title=title, level=level))
            _walk(child, level + 1)

    _walk(toc, 0)
    return items
