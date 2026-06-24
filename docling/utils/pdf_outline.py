"""Extract a PDF's outline (bookmarks / table-of-contents) via pypdfium2.

The outline, when present, is the most authoritative heading-hierarchy signal in a PDF. This
helper is shared by every PDFium-backed document backend so the signal is available regardless
of which text backend (pypdfium2 or docling-parse) is selected.
"""

from __future__ import annotations

import logging

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdfium2._helpers.misc import PdfiumError

from docling.datamodel.base_models import PdfOutlineItem
from docling.utils.locks import pypdfium2_lock

_log = logging.getLogger(__name__)

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


def extract_outline_from_pdfium(pdoc: pdfium.PdfDocument) -> list[PdfOutlineItem]:
    """Extract the outline as a flat, document-ordered list of :class:`PdfOutlineItem`.

    Vertical positions are converted to top-left origin (matching ``DocItem`` provenance) using
    the target page height. Returns an empty list when the document has no outline or it cannot
    be read.
    """
    items: list[PdfOutlineItem] = []
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
                PdfOutlineItem(
                    title=title, level=int(bm.level), page_no=page_no, y_top=y_top
                )
            )

    return items
