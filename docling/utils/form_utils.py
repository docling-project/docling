"""AcroForm widget-annotation extraction for interactive PDF forms.

Born-digital PDFs carry their form layer machine-readable: every field is a
widget annotation with a name (``/T``), a type (``/FT``), a current value
(``/V``), a checkbox/radio state, an accessible-name tooltip (``/TU``) and
behavioral flags. This module reads that layer through PDFium's annotation API
and represents it in a :class:`~docling_core.types.doc.DoclingDocument` as one
``FormItem`` per page, carrying a key/value ``GraphData`` — the structured-form
output tracked in discussion #301.

Extraction is lossless into :class:`FormFieldInfo`; the document representation
captures what the current docling-core schema can hold (field name, value/state,
widget geometry). Properties without a typed slot yet (tooltip, field type,
required/read-only) stay available on :class:`FormFieldInfo` for API users.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from docling_core.types.doc import BoundingBox, CoordOrigin, ProvenanceItem
from docling_core.types.doc.document import GraphCell, GraphData, GraphLink
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel

from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

_log = logging.getLogger(__name__)


class FormFieldType(str, Enum):
    """AcroForm field types (PDFium's ``FPDF_FORMFIELD_*`` taxonomy)."""

    UNKNOWN = "unknown"
    PUSHBUTTON = "pushbutton"
    CHECKBOX = "checkbox"
    RADIOBUTTON = "radiobutton"
    COMBOBOX = "combobox"
    LISTBOX = "listbox"
    TEXT = "text"
    SIGNATURE = "signature"


_FIELD_TYPES = {
    pdfium_c.FPDF_FORMFIELD_PUSHBUTTON: FormFieldType.PUSHBUTTON,
    pdfium_c.FPDF_FORMFIELD_CHECKBOX: FormFieldType.CHECKBOX,
    pdfium_c.FPDF_FORMFIELD_RADIOBUTTON: FormFieldType.RADIOBUTTON,
    pdfium_c.FPDF_FORMFIELD_COMBOBOX: FormFieldType.COMBOBOX,
    pdfium_c.FPDF_FORMFIELD_LISTBOX: FormFieldType.LISTBOX,
    pdfium_c.FPDF_FORMFIELD_TEXTFIELD: FormFieldType.TEXT,
    pdfium_c.FPDF_FORMFIELD_SIGNATURE: FormFieldType.SIGNATURE,
}

_CHECKABLE = {FormFieldType.CHECKBOX, FormFieldType.RADIOBUTTON}


@dataclass
class FormFieldInfo:
    """One widget-annotation form field, as read from the PDF."""

    page_no: int  # 1-based, matching DoclingDocument page numbering
    field_type: FormFieldType
    name: str  # fully qualified field name (/T chain)
    # Current value: /V for text-like fields; for checkable fields PDFium resolves
    # the state name (e.g. "Yes"/"Off") from /V or the appearance state (/AS).
    value: str
    tooltip: str  # alternate field name (/TU) — the PDF/UA accessible name
    checked: Optional[bool]  # checkbox/radio state; None for other types
    required: bool  # /Ff bit 2
    readonly: bool  # /Ff bit 1
    # Widget rectangle in PDF points, bottom-left origin: (left, bottom, right, top).
    rect: tuple[float, float, float, float]


def _annot_string(func: Callable, formenv_handle: object, annot: object) -> str:
    """Read a PDFium UTF-16LE string via the two-call buffer-length protocol."""
    n_bytes = func(formenv_handle, annot, None, 0)
    if n_bytes <= 2:  # empty string: just the NUL terminator
        return ""
    buffer = ctypes.create_string_buffer(n_bytes)
    func(
        formenv_handle,
        annot,
        ctypes.cast(buffer, ctypes.POINTER(pdfium_c.FPDF_WCHAR)),
        n_bytes,
    )
    return buffer.raw[: n_bytes - 2].decode("utf-16-le", errors="replace")


def extract_form_fields(pdoc: pdfium.PdfDocument) -> list[FormFieldInfo]:
    """Read all AcroForm widget fields of an open pypdfium2 document.

    Returns an empty list for documents without an interactive form. The
    caller keeps ownership of ``pdoc``; pages and annotations opened here are
    closed before returning.
    """
    fields: list[FormFieldInfo] = []
    with pypdfium2_lock:
        if pdoc.formenv is None:
            try:
                pdoc.init_forms()
            except pdfium.PdfiumError as err:  # malformed AcroForm: no form layer
                _log.warning("Could not initialize PDF form environment: %s", err)
                return fields
        if pdoc.formenv is None:  # document has no interactive form
            return fields
        formenv = pdoc.formenv.raw

        for page_index in range(len(pdoc)):
            page = pdoc.get_page(page_index)
            try:
                annot_count = pdfium_c.FPDFPage_GetAnnotCount(page.raw)
                for annot_index in range(annot_count):
                    annot = pdfium_c.FPDFPage_GetAnnot(page.raw, annot_index)
                    if not annot:  # pragma: no cover - defensive against API failure
                        continue
                    try:
                        subtype = pdfium_c.FPDFAnnot_GetSubtype(annot)
                        if subtype != pdfium_c.FPDF_ANNOT_WIDGET:
                            continue
                        field = _read_widget(formenv, annot, page_no=page_index + 1)
                        if field is not None:
                            fields.append(field)
                    finally:
                        pdfium_c.FPDFPage_CloseAnnot(annot)
            finally:
                page.close()
    return fields


def _read_widget(
    formenv: object, annot: object, page_no: int
) -> Optional[FormFieldInfo]:
    """Map one widget annotation to :class:`FormFieldInfo`."""
    raw_type = pdfium_c.FPDFAnnot_GetFormFieldType(formenv, annot)
    field_type = _FIELD_TYPES.get(raw_type, FormFieldType.UNKNOWN)

    rect_struct = pdfium_c.FS_RECTF()
    got_rect = pdfium_c.FPDFAnnot_GetRect(annot, rect_struct)
    if not got_rect:  # pragma: no cover - defensive against API failure
        return None
    # FS_RECTF is in page space (bottom-left origin); normalize the corner order.
    left = min(rect_struct.left, rect_struct.right)
    right = max(rect_struct.left, rect_struct.right)
    bottom = min(rect_struct.bottom, rect_struct.top)
    top = max(rect_struct.bottom, rect_struct.top)

    name = _annot_string(pdfium_c.FPDFAnnot_GetFormFieldName, formenv, annot)
    value = _annot_string(pdfium_c.FPDFAnnot_GetFormFieldValue, formenv, annot)
    tooltip = _annot_string(
        pdfium_c.FPDFAnnot_GetFormFieldAlternateName, formenv, annot
    )
    flags = pdfium_c.FPDFAnnot_GetFormFieldFlags(formenv, annot)

    checked: Optional[bool] = None
    if field_type in _CHECKABLE:
        checked = bool(pdfium_c.FPDFAnnot_IsChecked(formenv, annot))

    return FormFieldInfo(
        page_no=page_no,
        field_type=field_type,
        name=name,
        value=value,
        tooltip=tooltip,
        checked=checked,
        required=bool(flags & pdfium_c.FPDF_FORMFLAG_REQUIRED),
        readonly=bool(flags & pdfium_c.FPDF_FORMFLAG_READONLY),
        rect=(left, bottom, right, top),
    )


def attach_form_fields(doc: DoclingDocument, fields: list[FormFieldInfo]) -> None:
    """Represent extracted fields in ``doc`` as one ``FormItem`` per page.

    Each field becomes a ``key`` cell (the field name; no page provenance — the
    name is dictionary metadata, not page text) linked ``to_value`` to a value
    cell anchored at the widget rectangle. Checkable fields use the ``checkbox``
    cell label with an explicit ``checked``/``unchecked`` text so state survives
    every export path.
    """
    by_page: dict[int, list[FormFieldInfo]] = {}
    for field in fields:
        by_page.setdefault(field.page_no, []).append(field)

    for page_no in sorted(by_page):
        page_fields = by_page[page_no]
        cells: list[GraphCell] = []
        links: list[GraphLink] = []
        for field in page_fields:
            key_id = len(cells)
            cells.append(
                GraphCell(
                    label=GraphCellLabel.KEY,
                    cell_id=key_id,
                    text=field.name,
                    orig=field.name,
                    prov=None,
                )
            )
            if field.checked is not None:
                value_label = GraphCellLabel.CHECKBOX
                value_text = "checked" if field.checked else "unchecked"
            else:
                value_label = GraphCellLabel.VALUE
                value_text = field.value
            value_id = len(cells)
            cells.append(
                GraphCell(
                    label=value_label,
                    cell_id=value_id,
                    text=value_text,
                    orig=value_text,
                    prov=ProvenanceItem(
                        page_no=page_no,
                        bbox=BoundingBox(
                            l=field.rect[0],
                            b=field.rect[1],
                            r=field.rect[2],
                            t=field.rect[3],
                            coord_origin=CoordOrigin.BOTTOMLEFT,
                        ),
                        charspan=(0, len(value_text)),
                    ),
                )
            )
            links.append(
                GraphLink(
                    label=GraphLinkLabel.TO_VALUE,
                    source_cell_id=key_id,
                    target_cell_id=value_id,
                )
            )

        form_bbox = BoundingBox(
            l=min(f.rect[0] for f in page_fields),
            b=min(f.rect[1] for f in page_fields),
            r=max(f.rect[2] for f in page_fields),
            t=max(f.rect[3] for f in page_fields),
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        doc.add_form(
            graph=GraphData(cells=cells, links=links),
            prov=ProvenanceItem(page_no=page_no, bbox=form_bbox, charspan=(0, 0)),
        )
