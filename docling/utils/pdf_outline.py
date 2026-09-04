# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Extract a PDF's outline (bookmarks / table-of-contents).

The outline, when present, is the most authoritative heading-hierarchy signal in a PDF. Two
extractors are provided:

* :func:`extract_outline_from_pdfium` -- for the pypdfium2 backend. Returns the richest data:
  title, depth, target page and vertical position.
* :func:`extract_outline_from_docling_parse` -- fallback for the docling-parse backends, using
  their native ``get_table_of_contents()`` (no pypdfium2 dependency). The native outline carries
  titles, hierarchy, and an optional target page; position is left unset.

``pypdfium2`` is imported lazily, inside the functions that use it, never at module level:
``datamodel.document`` imports this module for the ``_PdfOutlineItem`` model, which places it on
the ``docling.service_client`` import chain. That chain must stay importable on any docling-slim
install that does not enable the PDF pipeline (and therefore ships no ``pypdfium2``).
"""

from __future__ import annotations

import logging
from functools import cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    import pypdfium2 as pdfium
    from docling_parse.pdf_parser import (
        PdfDocument as DoclingParsePdfDocument,
        PdfTableOfContentsWithPage,
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
            # pypdfium2 4.x defaults to depth 15; use an explicit generous
            # bound so deeply nested outlines are not silently truncated.
            toc = list(pdoc.get_toc(max_depth=1000))
        except PdfiumError as exc:
            _log.debug("Could not read PDF outline: %s", exc)
            return []

        for bm in toc:
            try:
                title = bm.get_title()
            except AttributeError:
                # pypdfium2 4.x exposes outline records as namedtuples.
                title = bm.title
            title = (title or "").strip()
            if not title:
                continue

            page_no: int | None = None
            y_top: float | None = None
            try:
                try:
                    dest = bm.get_dest()
                except AttributeError:
                    # pypdfium2 4.x stores the destination on the record.
                    dest = bm.dest
            except (AttributeError, PdfiumError):
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


def extract_outline_from_pdfium_path_or_stream(
    path_or_stream: BytesIO | Path,
    *,
    password: str | None = None,
) -> list[_PdfOutlineItem]:
    """Open a transient PDFium document and extract its outline.

    Used by backends that do not keep a PDFium document handle alive. ``BytesIO`` inputs are
    rewound for PDFium and restored to their original position before returning.
    """
    # lazy imports (see module docstring)
    import pypdfium2 as pdfium
    from pypdfium2._helpers.misc import PdfiumError

    stream_pos: int | None = None
    if isinstance(path_or_stream, BytesIO):
        stream_pos = path_or_stream.tell()
        path_or_stream.seek(0)

    pdoc: pdfium.PdfDocument | None = None
    try:
        with pypdfium2_lock:
            pdoc = pdfium.PdfDocument(path_or_stream, password=password)
        return extract_outline_from_pdfium(pdoc)
    except (PdfiumError, RuntimeError) as exc:
        _log.debug("Could not open PDF with PDFium for outline extraction: %s", exc)
        return []
    finally:
        if pdoc is not None:
            try:
                with pypdfium2_lock:
                    pdoc.close()
            except (PdfiumError, RuntimeError) as exc:
                _log.debug("Could not close PDFium outline document: %s", exc)
        if stream_pos is not None:
            path_or_stream.seek(stream_pos)


def extract_outline_from_docling_parse(
    dp_doc: DoclingParsePdfDocument,
) -> list[_PdfOutlineItem]:
    """Flatten docling-parse's native table-of-contents into ordered ``_PdfOutlineItem``\\ s.

    Walks the ``PdfTableOfContents`` tree returned by ``PdfDocument.get_table_of_contents()``,
    depth-first, assigning each node a 0-based ``level`` from its depth (top-level entries at
    level 0, matching the pypdfium2 extractor). Target pages are converted from docling-parse's
    zero-based representation to the 1-based page numbering used by Docling. Vertical position
    is left unset.

    ``get_table_of_contents()`` returns ``None`` for PDFs without an embedded outline, in which
    case an empty list is returned.
    """
    toc = dp_doc.get_table_of_contents()
    if toc is None:
        return []

    items: list[_PdfOutlineItem] = []

    # Iterative pre-order depth-first walk via an explicit stack, avoiding Python's
    # call-stack recursion limit. Some large real-world documents (technical manuals,
    # legal filings) legitimately nest headings hundreds of levels deep, and malformed
    # PDFs can nest further still; a naive recursive walk here raises RecursionError.
    stack: list[tuple[PdfTableOfContentsWithPage, int]] = [
        (child, 0) for child in reversed(toc.children or [])
    ]
    while stack:
        node, level = stack.pop()
        title = (node.text or node.orig or "").strip()
        if title:
            page_no = node.page + 1 if node.page is not None else None
            items.append(_PdfOutlineItem(title=title, level=level, page_no=page_no))
        stack.extend((child, level + 1) for child in reversed(node.children or []))

    return items
