# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read formula structure elements out of a tagged PDF's structure tree.

A tagged (accessible / PDF-UA) PDF marks each equation with a ``Formula`` structure element,
and producers may attach the author's own MathML to it. That representation is more faithful
than anything reconstructed from the rendered glyphs, so it is worth preferring over the
formula understanding model when it is there.

Only what PDFium's public API can reach is covered:

* **producer attributes** on the structure element -- Microsoft's PDF/UA output writes the
  MathML into an ``MSFT_MathML`` attribute, which is the mechanism seen in the wild;
* ``/ActualText`` and ``/Alt``, used as the item's text once the model is skipped.

MathML delivered as an *associated file* (``/AF`` on the structure element, the PDF 2.0 /
PDF-UA-2 route) is **not** reachable: PDFium exposes only document-level
``/Names/EmbeddedFiles``, with no way to tie an attachment back to a structure element.
Supporting it needs a raw object-model reader (pikepdf/pypdf), which is not a dependency.

``pypdfium2`` is imported lazily, inside the functions that use it, never at module level:
``datamodel.document`` imports this module for the ``_PdfFormulaStruct`` model, which places it
on the ``docling.service_client`` import chain. That chain must stay importable on any
docling-slim install that does not enable the PDF pipeline (and therefore ships no
``pypdfium2``) -- the same constraint documented in :mod:`docling.utils.pdf_outline`.
"""

from __future__ import annotations

import ctypes
import logging
import re
from typing import TYPE_CHECKING, Any, Iterable
from xml.etree import ElementTree

from docling_core.types.doc import BoundingBox, CoordOrigin
from pydantic import BaseModel

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    import pypdfium2 as pdfium

_log = logging.getLogger(__name__)

# Structure element type that marks an equation (PDF 32000-1, table 333).
_FORMULA_TAG = "Formula"

# Attribute keys known to carry MathML. Microsoft's PDF/UA export writes ``MSFT_MathML``;
# the suffix match leaves room for other producers using the same convention.
_MATHML_ATTR_RE = re.compile(r"(?:^|_)mathml$", re.IGNORECASE)

# Element names rejected outright: MathML that carries these is either not MathML or is
# being used to smuggle markup. The accepted string is emitted verbatim into exported HTML,
# so it must not become a script-injection vector.
_MATHML_FORBIDDEN_TAGS = {"script", "annotation-xml"}

# Cap on how much of a structure tree is walked per page, so a pathological (or hostile)
# document cannot spin here.
_MAX_STRUCT_ELEMENTS_PER_PAGE = 10_000


class _PdfFormulaStruct(BaseModel):
    """One ``Formula`` structure element found in a tagged PDF (internal).

    Internal data-passing structure between a PDF backend's ``get_formula_structures()`` and
    the native-formula stage; not part of the public datamodel or the serialized output.
    """

    # 1-based page number, matching ``ProvenanceItem.page_no``.
    page_no: int
    # Top-left origin, in 72-DPI document points; None when neither the ``BBox`` attribute
    # nor the element's marked content could give one.
    bbox: BoundingBox | None = None
    mathml: str | None = None
    actual_text: str | None = None
    alt_text: str | None = None


def _utf16_text(fn: Any, *args: Any) -> str:
    """Read one of PDFium's ``(buffer, buflen) -> byte length`` UTF-16LE string getters."""
    length = fn(*args, None, 0)
    if not length or length <= 0:
        return ""
    buf = ctypes.create_string_buffer(length)
    fn(*args, buf, length)
    # The reported length covers the NUL terminator, and some producers pad further.
    return buf.raw[:length].decode("utf-16-le", errors="replace").rstrip("\x00")


def _attr_names(pdfium_c: Any, attr: Any) -> list[str]:
    """Return the keys of one structure-element attribute dictionary."""
    count = pdfium_c.FPDF_StructElement_Attr_GetCount(attr)
    names: list[str] = []
    for i in range(max(count, 0)):
        out_len = ctypes.c_ulong(0)
        if not pdfium_c.FPDF_StructElement_Attr_GetName(
            attr, i, None, 0, ctypes.byref(out_len)
        ):
            continue
        buf = ctypes.create_string_buffer(out_len.value)
        pdfium_c.FPDF_StructElement_Attr_GetName(
            attr, i, buf, out_len.value, ctypes.byref(out_len)
        )
        names.append(buf.value.decode("utf-8", errors="replace"))
    return names


def _attr_string(pdfium_c: Any, attr: Any, name: str) -> str | None:
    """Read a string-valued attribute, or None when it is absent or not a string."""
    value = pdfium_c.FPDF_StructElement_Attr_GetValue(attr, name.encode())
    if not value:
        return None
    if pdfium_c.FPDF_StructElement_Attr_GetType(value) != pdfium_c.FPDF_OBJECT_STRING:
        return None
    out_len = ctypes.c_ulong(0)
    if not pdfium_c.FPDF_StructElement_Attr_GetStringValue(
        value, None, 0, ctypes.byref(out_len)
    ):
        return None
    buf = ctypes.create_string_buffer(out_len.value)
    pdfium_c.FPDF_StructElement_Attr_GetStringValue(
        value, buf, out_len.value, ctypes.byref(out_len)
    )
    text = buf.raw[: out_len.value].decode("utf-16-le", errors="replace").rstrip("\x00")
    return text or None


def _attr_rect(pdfium_c: Any, attr: Any, name: str) -> tuple[float, ...] | None:
    """Read a 4-number array attribute (e.g. ``BBox``) in PDF bottom-left coordinates."""
    value = pdfium_c.FPDF_StructElement_Attr_GetValue(attr, name.encode())
    if not value:
        return None
    if pdfium_c.FPDF_StructElement_Attr_GetType(value) != pdfium_c.FPDF_OBJECT_ARRAY:
        return None
    if pdfium_c.FPDF_StructElement_Attr_CountChildren(value) != 4:
        return None
    numbers: list[float] = []
    for i in range(4):
        child = pdfium_c.FPDF_StructElement_Attr_GetChildAtIndex(value, i)
        if not child:
            return None
        out = ctypes.c_float(0)
        if not pdfium_c.FPDF_StructElement_Attr_GetNumberValue(
            child, ctypes.byref(out)
        ):
            return None
        numbers.append(out.value)
    return tuple(numbers)


def _sanitize_mathml(text: str) -> str | None:
    """Return *text* when it is usable MathML, else None.

    The value comes from the PDF and is emitted verbatim by the HTML serializer, so it is
    accepted only when it parses as XML, is rooted at ``math``, and carries none of the
    elements that would let it smuggle non-MathML markup into an exported page.
    """
    candidate = text.strip()
    if not candidate:
        return None
    try:
        root = ElementTree.fromstring(candidate)
    except ElementTree.ParseError as exc:
        _log.debug("Ignoring malformed MathML in structure tree: %s", exc)
        return None

    def _local_name(tag: object) -> str:
        return str(tag).rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""

    if _local_name(root.tag) != "math":
        _log.debug("Ignoring structure-tree MathML with root %r", root.tag)
        return None
    for element in root.iter():
        if _local_name(element.tag) in _MATHML_FORBIDDEN_TAGS:
            _log.debug("Ignoring structure-tree MathML containing %r", element.tag)
            return None
    return candidate


def _mcid_boxes(pdfium_c: Any, page: pdfium.PdfPage) -> dict[int, BoundingBox]:
    """Map each marked-content id on *page* to the union of its page objects' bounds.

    This is the fallback for locating a structure element that carries no ``BBox``
    attribute: the element names its marked-content ids, and the page objects tagged with
    those ids are what it actually covers. Coordinates stay in PDF bottom-left space.
    """
    boxes: dict[int, BoundingBox] = {}
    for index in range(pdfium_c.FPDFPage_CountObjects(page.raw)):
        obj = pdfium_c.FPDFPage_GetObject(page.raw, index)
        if not obj:
            continue
        mcid = pdfium_c.FPDFPageObj_GetMarkedContentID(obj)
        if mcid < 0:
            continue
        left, bottom, right, top = (ctypes.c_float() for _ in range(4))
        if not pdfium_c.FPDFPageObj_GetBounds(
            obj,
            ctypes.byref(left),
            ctypes.byref(bottom),
            ctypes.byref(right),
            ctypes.byref(top),
        ):
            continue
        box = BoundingBox(
            l=left.value,
            b=bottom.value,
            r=right.value,
            t=top.value,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        previous = boxes.get(mcid)
        boxes[mcid] = (
            box if previous is None else BoundingBox.enclosing_bbox([previous, box])
        )
    return boxes


def _element_mcids(pdfium_c: Any, element: Any) -> list[int]:
    count = pdfium_c.FPDF_StructElement_GetMarkedContentIdCount(element)
    mcids = [
        pdfium_c.FPDF_StructElement_GetMarkedContentIdAtIndex(element, i)
        for i in range(max(count, 0))
    ]
    return [mcid for mcid in mcids if mcid >= 0]


def _formula_from_element(
    pdfium_c: Any,
    element: Any,
    *,
    page_no: int,
    page_height: float,
    mcid_boxes: dict[int, BoundingBox],
) -> _PdfFormulaStruct:
    """Build the record for one ``Formula`` structure element."""
    mathml: str | None = None
    bbox_pdf: tuple[float, ...] | None = None

    for i in range(max(pdfium_c.FPDF_StructElement_GetAttributeCount(element), 0)):
        attr = pdfium_c.FPDF_StructElement_GetAttributeAtIndex(element, i)
        if not attr:
            continue
        for name in _attr_names(pdfium_c, attr):
            if mathml is None and _MATHML_ATTR_RE.search(name):
                raw_value = _attr_string(pdfium_c, attr, name)
                if raw_value:
                    mathml = _sanitize_mathml(raw_value)
            elif bbox_pdf is None and name == "BBox":
                bbox_pdf = _attr_rect(pdfium_c, attr, name)

    if bbox_pdf is not None:
        left, bottom, right, top = bbox_pdf
        box: BoundingBox | None = BoundingBox(
            l=left, b=bottom, r=right, t=top, coord_origin=CoordOrigin.BOTTOMLEFT
        )
    else:
        covered = [
            mcid_boxes[mcid]
            for mcid in _element_mcids(pdfium_c, element)
            if mcid in mcid_boxes
        ]
        box = BoundingBox.enclosing_bbox(covered) if covered else None

    return _PdfFormulaStruct(
        page_no=page_no,
        bbox=box.to_top_left_origin(page_height=page_height)
        if box is not None
        else None,
        mathml=mathml,
        actual_text=_utf16_text(pdfium_c.FPDF_StructElement_GetActualText, element)
        or None,
        alt_text=_utf16_text(pdfium_c.FPDF_StructElement_GetAltText, element) or None,
    )


def _collect_page(
    pdfium_c: Any, page: pdfium.PdfPage, page_no: int
) -> list[_PdfFormulaStruct]:
    """Walk one page's structure tree and return its ``Formula`` elements."""
    tree = pdfium_c.FPDF_StructTree_GetForPage(page.raw)
    if not tree:
        return []

    found: list[_PdfFormulaStruct] = []
    try:
        page_height = page.get_height()
        # Marked-content bounds are only needed by elements without a BBox attribute, and
        # scanning every page object is not free -- so build the map on first use.
        mcid_boxes: dict[int, BoundingBox] | None = None

        # Iterative pre-order walk via an explicit stack, avoiding Python's call-stack
        # recursion limit on deeply nested (or malformed) structure trees.
        stack = [
            pdfium_c.FPDF_StructTree_GetChildAtIndex(tree, i)
            for i in reversed(range(pdfium_c.FPDF_StructTree_CountChildren(tree)))
        ]
        visited = 0
        while stack:
            element = stack.pop()
            if not element:
                continue
            visited += 1
            if visited > _MAX_STRUCT_ELEMENTS_PER_PAGE:
                _log.debug(
                    "Stopped structure-tree walk on page %d after %d elements",
                    page_no,
                    _MAX_STRUCT_ELEMENTS_PER_PAGE,
                )
                break

            if (
                _utf16_text(pdfium_c.FPDF_StructElement_GetType, element)
                == _FORMULA_TAG
            ):
                if mcid_boxes is None:
                    mcid_boxes = _mcid_boxes(pdfium_c, page)
                found.append(
                    _formula_from_element(
                        pdfium_c,
                        element,
                        page_no=page_no,
                        page_height=page_height,
                        mcid_boxes=mcid_boxes,
                    )
                )
                # A Formula element is a leaf as far as this stage cares; nested structure
                # below it is part of the same equation.
                continue

            stack.extend(
                pdfium_c.FPDF_StructElement_GetChildAtIndex(element, i)
                for i in reversed(
                    range(pdfium_c.FPDF_StructElement_CountChildren(element))
                )
            )
    finally:
        pdfium_c.FPDF_StructTree_Close(tree)

    return found


def extract_formula_structs_from_pdfium(
    pdoc: pdfium.PdfDocument, page_nos: Iterable[int]
) -> list[_PdfFormulaStruct]:
    """Return the ``Formula`` structure elements on the requested pages.

    Args:
        pdoc: An open pypdfium2 document.
        page_nos: 1-based page numbers to inspect.

    Returns:
        One record per ``Formula`` element found, in page order. Empty when the document
        is not tagged, has no formula elements, or could not be read.
    """
    # lazy import (see module docstring)
    import pypdfium2.raw as pdfium_c
    from pypdfium2._helpers.misc import PdfiumError

    results: list[_PdfFormulaStruct] = []
    with pypdfium2_lock:
        page_count = len(pdoc)
        for page_no in page_nos:
            if not 1 <= page_no <= page_count:
                continue
            try:
                page = pdoc[page_no - 1]
            except (PdfiumError, IndexError) as exc:
                _log.debug(
                    "Could not load page %d for structure tree: %s", page_no, exc
                )
                continue
            try:
                results.extend(_collect_page(pdfium_c, page, page_no))
            except (PdfiumError, OSError, ValueError) as exc:
                _log.debug("Could not read structure tree on page %d: %s", page_no, exc)
            finally:
                page.close()

    return results
