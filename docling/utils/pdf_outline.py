# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Extract a PDF's outline (bookmarks / table-of-contents).

The outline, when present, is the most authoritative heading-hierarchy signal in a PDF. Two
extractors are provided so each backend uses its own native capability:

* :func:`extract_outline_from_pdfium` -- for the pypdfium2 backend. Returns the richest data:
  title, depth, target page and vertical position.
* :func:`extract_outline_from_docling_parse` -- for the docling-parse backend, using its native
  document annotations (no pypdfium2 dependency). The native outline carries titles, hierarchy,
  and destinations.

``pypdfium2`` is imported lazily, inside the functions that use it, never at module level:
``datamodel.document`` imports this module for the ``_PdfOutlineItem`` model, which places it on
the ``docling.service_client`` import chain. That chain must stay importable on any docling-slim
install that does not enable the PDF pipeline (and therefore ships no ``pypdfium2``).
"""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    import pypdfium2 as pdfium
    from docling_parse.pdf_parser import PdfTableOfContents

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
    # 1-based target page; None when the entry has no resolvable page.
    page_no: int | None = None
    # Top-left-origin vertical position of the target, when derivable from the destination view.
    y_top: float | None = None


@cache
def _view_top_index() -> dict[int, int]:
    """Destination view modes whose coordinates carry a usable vertical (top) position.

    Coordinates are in PDF space (bottom-left origin). Modes not listed (FIT, FITV, FITB,
    FITBV, unknown) provide no top.

    Returns:
        A mapping of each supported ``PDFDEST_VIEW_*`` mode to the index of the vertical (top)
        coordinate within that mode's position tuple:

        * ``PDFDEST_VIEW_XYZ`` -> ``1`` (position is ``[x, y, zoom]``)
        * ``PDFDEST_VIEW_FITH`` -> ``0`` (position is ``[y]``)
        * ``PDFDEST_VIEW_FITBH`` -> ``0`` (position is ``[y]``)
        * ``PDFDEST_VIEW_FITR`` -> ``3`` (position is ``[left, bottom, right, top]``)
    """
    # lazy import (see module docstring)
    import pypdfium2.raw as pdfium_c

    return {
        pdfium_c.PDFDEST_VIEW_XYZ: 1,
        pdfium_c.PDFDEST_VIEW_FITH: 0,
        pdfium_c.PDFDEST_VIEW_FITBH: 0,
        pdfium_c.PDFDEST_VIEW_FITR: 3,
    }


def _dest_top_pdf(dest: pdfium.PdfDest) -> tuple[int | None, float | None]:
    """Return ``(0-based page index, vertical top in PDF bottom-left coords)`` for a dest.

    Either element may be ``None`` when the destination does not encode it.
    """
    page_index = dest.get_index()
    mode, pos = dest.get_view()
    idx = _view_top_index().get(mode)
    y_pdf = pos[idx] if idx is not None and idx < len(pos) else None
    return page_index, y_pdf


def extract_outline_from_pdfium(pdoc: pdfium.PdfDocument) -> list[_PdfOutlineItem]:
    """Extract the outline as a flat, document-ordered list of :class:`_PdfOutlineItem`.

    Vertical positions are converted to top-left origin (matching ``DocItem`` provenance) using
    the target page height. Returns an empty list when the document has no outline or it cannot
    be read.
    """
    # lazy import (see module docstring)
    from pypdfium2._helpers.misc import PdfiumError

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


def extract_outline_from_docling_parse(
    toc: PdfTableOfContents | None,
) -> list[_PdfOutlineItem]:
    """Flatten docling-parse's native table-of-contents into ordered ``_PdfOutlineItem``\\ s.

    The native iterator supplies each node's 0-based depth. Destination coordinates are
    normalized to top-left origin to match document-item provenance. ``None`` represents a PDF
    without an embedded outline.
    """
    if toc is None:
        return []

    items: list[_PdfOutlineItem] = []
    for level, node in toc.iterate():
        title = (node.text or node.orig or "").strip()
        if title:
            destination = node.destination
            page_no = destination.page_no if destination is not None else None
            y_top: float | None = None
            if destination is not None:
                top_left_destination = destination.to_top_left_origin()
                if top_left_destination.point is not None:
                    y_top = top_left_destination.point.y
            items.append(
                _PdfOutlineItem(
                    title=title,
                    level=level,
                    page_no=page_no,
                    y_top=y_top,
                )
            )

    return items
