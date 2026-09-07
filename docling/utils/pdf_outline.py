# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Extract a PDF's outline (bookmarks / table-of-contents).

The outline, when present, is the most authoritative heading-hierarchy signal in a PDF. Two
extractors are provided so each backend uses its own native capability. Both yield the same
data -- title, depth, target page and vertical position -- so bookmark matching behaves
identically whichever backend produced the outline:

* :func:`extract_outline_from_pdfium` -- for the pypdfium2 backend, reading PDFium's own
  bookmark destinations.
* :func:`extract_outline_from_docling_parse` -- for the docling-parse backends, flattening the
  ``PdfTableOfContents`` those backends read natively (no pypdfium2 dependency).

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
    from docling_core.types.doc.page import PdfTableOfContents

_log = logging.getLogger(__name__)

# Depth bound for the pypdfium2 outline walk. Its default of 15 silently drops deeper
# subtrees, which real documents do have; 100 covers them with room to spare. It is not
# raised further because ``get_toc()`` recurses (``yield from``) once per level, so an
# unbounded depth would trade silent truncation for a RecursionError on a maliciously
# nested outline. The docling-parse extractor walks an explicit stack and needs no bound.
_MAX_OUTLINE_DEPTH = 100


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
    # 1-based target page; None when the entry's destination could not be resolved.
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
            toc = list(pdoc.get_toc(max_depth=_MAX_OUTLINE_DEPTH))
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

    Takes the ``PdfTableOfContents`` root that both docling-parse backends read natively -- the
    lazy document's ``get_table_of_contents()`` and the threaded parser's
    ``get_annotations().table_of_contents`` return the same model -- and walks it depth-first in
    document order. Each entry's 0-based ``level`` is its depth below the synthetic root, so
    top-level entries are at level 0, matching the pypdfium2 extractor.

    Each node carries a ``destination``, from which the 1-based target page and, when the
    destination encodes one, the vertical position are taken. Destination coordinates are
    reported in the target page's own frame -- the frame that page's cells use -- so converting
    to a top-left origin makes ``y_top`` directly comparable with ``DocItem`` provenance.
    Destinations that encode no position (``FIT``, ``FIT_B``) leave ``y_top`` unset.

    ``None`` is accepted, and yields an empty list, for PDFs without an embedded outline.
    """
    if toc is None:
        return []

    items: list[_PdfOutlineItem] = []
    for level, node in toc.iterate():
        title = (node.text or node.orig or "").strip()
        if not title:
            continue

        page_no: int | None = None
        y_top: float | None = None
        dest = node.destination
        if dest is not None:
            page_no = dest.page_no
            point = dest.to_top_left_origin().point
            if point is not None:
                y_top = point.y

        items.append(
            _PdfOutlineItem(title=title, level=level, page_no=page_no, y_top=y_top)
        )

    return items
