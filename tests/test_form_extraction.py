from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc.document import DoclingDocument, FormItem
from docling_core.types.doc.labels import GraphCellLabel, GraphLinkLabel

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.image_backend import ImageDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.form_utils import (
    FormFieldType,
    attach_form_fields,
    extract_form_fields,
)

FORM_PDF = Path("./tests/data/pdf/sources/acroform_sample.pdf")
PLAIN_PDF = Path("./tests/data/pdf/sources/2305.03393v1-pg9.pdf")
PLAIN_IMAGE = Path("./tests/data/ocr/sources/old_newspaper.png")


def _get_backend(backend_cls, pdf_path: Path):
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=backend_cls,
    )
    return in_doc._backend


@pytest.mark.parametrize(
    "backend_cls", [PyPdfiumDocumentBackend, DoclingParseDocumentBackend]
)
def test_backend_reads_acroform_widget_fields(backend_cls):
    backend = _get_backend(backend_cls, FORM_PDF)
    try:
        fields = backend.get_form_fields()
    finally:
        backend.unload()

    # The fixture also carries a Link annotation; only widgets may surface here.
    assert [f.name for f in fields] == [
        "applicant_name",
        "agree_terms",
        "newsletter",
        "mail_optin",
    ]

    text_field, checked_box, unchecked_box, as_only_box = fields

    assert text_field.field_type == FormFieldType.TEXT
    assert text_field.value == "Ada Lovelace"
    assert text_field.tooltip == "Full legal name"  # /TU: the accessible name
    assert text_field.required is True
    assert text_field.readonly is False
    assert text_field.checked is None
    assert text_field.page_no == 1
    left, bottom, right, top = text_field.rect
    assert (left, bottom, right, top) == (150.0, 690.0, 400.0, 710.0)

    assert checked_box.field_type == FormFieldType.CHECKBOX
    assert checked_box.checked is True
    assert checked_box.tooltip == "I agree to the terms"

    assert unchecked_box.field_type == FormFieldType.CHECKBOX
    assert unchecked_box.checked is False
    assert unchecked_box.required is False

    # Checked only through /AS (no /V): PDFium resolves the state name from
    # the appearance dictionary, a common pattern in viewer-filled forms.
    assert as_only_box.field_type == FormFieldType.CHECKBOX
    assert as_only_box.checked is True
    assert as_only_box.value == "On"


@pytest.mark.parametrize(
    "backend_cls", [PyPdfiumDocumentBackend, DoclingParseDocumentBackend]
)
def test_backend_without_acroform_yields_no_fields(backend_cls):
    backend = _get_backend(backend_cls, PLAIN_PDF)
    try:
        assert backend.get_form_fields() == []
    finally:
        backend.unload()


def test_attach_form_fields_builds_one_form_item_per_page():
    backend = _get_backend(PyPdfiumDocumentBackend, FORM_PDF)
    try:
        fields = extract_form_fields(backend._pdoc)
    finally:
        backend.unload()

    doc = DoclingDocument(name="acroform_sample")
    attach_form_fields(doc, fields)

    assert len(doc.form_items) == 1
    form = doc.form_items[0]
    assert isinstance(form, FormItem)
    assert form.prov[0].page_no == 1

    cells = form.graph.cells
    links = form.graph.links
    assert len(cells) == 8  # key + value per field
    assert len(links) == 4
    assert all(link.label == GraphLinkLabel.TO_VALUE for link in links)

    by_key = {}
    for link in links:
        key = next(c for c in cells if c.cell_id == link.source_cell_id)
        value = next(c for c in cells if c.cell_id == link.target_cell_id)
        by_key[key.text] = value

    assert by_key["applicant_name"].label == GraphCellLabel.VALUE
    assert by_key["applicant_name"].text == "Ada Lovelace"
    # The value cell anchors at the widget rectangle; the key cell carries no
    # page provenance (a field name is dictionary metadata, not page text).
    assert by_key["applicant_name"].prov.page_no == 1
    key_cells = [c for c in cells if c.label == GraphCellLabel.KEY]
    assert all(c.prov is None for c in key_cells)

    assert by_key["agree_terms"].label == GraphCellLabel.CHECKBOX
    assert by_key["agree_terms"].text == "checked"
    assert by_key["newsletter"].label == GraphCellLabel.CHECKBOX
    assert by_key["newsletter"].text == "unchecked"
    assert by_key["mail_optin"].text == "checked"


def test_form_fields_empty_after_unload():
    backend = _get_backend(PyPdfiumDocumentBackend, FORM_PDF)
    backend.unload()
    assert backend.get_form_fields() == []


def test_malformed_form_environment_degrades_to_no_fields(monkeypatch, caplog):
    """A document whose form layer cannot be initialized yields no fields, no crash."""
    pdoc = pdfium.PdfDocument(FORM_PDF)
    try:

        def _raise(*args, **kwargs):
            raise pdfium.PdfiumError("form init failed")

        monkeypatch.setattr(pdoc, "init_forms", _raise)
        with caplog.at_level("WARNING", logger="docling.utils.form_utils"):
            assert extract_form_fields(pdoc) == []
        assert "form environment" in caplog.text
    finally:
        pdoc.close()


def test_non_pdfium_backend_reports_no_form_fields():
    """Backends without a PDFium handle (e.g. images) keep the empty default."""
    in_doc = InputDocument(
        path_or_stream=PLAIN_IMAGE,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
    )
    backend = in_doc._backend
    try:
        assert backend.get_form_fields() == []
    finally:
        backend.unload()


def test_pipeline_option_gates_form_extraction():
    def convert(extract_form_fields: bool):
        options = PdfPipelineOptions(extract_form_fields=extract_form_fields)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        return converter.convert(FORM_PDF).document

    enabled = convert(extract_form_fields=True)
    assert len(enabled.form_items) == 1
    assert {c.text for c in enabled.form_items[0].graph.cells} >= {
        "applicant_name",
        "Ada Lovelace",
        "checked",
        "unchecked",
    }

    disabled = convert(extract_form_fields=False)
    assert len(disabled.form_items) == 0
