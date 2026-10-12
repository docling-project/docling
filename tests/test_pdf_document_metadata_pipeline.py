# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""End-to-end: the PDF pipeline attaches /Lang and document properties when asked."""

from pathlib import Path

import pytest
from docling_core.types.doc.labels import HumanLanguageLabel

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.pdf_metadata import LANGUAGE_SOURCE_PDF_CATALOG

pytestmark = pytest.mark.ml_pdf_model

METADATA_PDF = Path("./tests/data/pdf/document_metadata.pdf")


def _convert(attach_document_metadata: bool):
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        attach_document_metadata=attach_document_metadata,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return converter.convert(METADATA_PDF).document


def test_pipeline_attaches_language_and_properties_when_enabled():
    doc = _convert(attach_document_metadata=True)

    meta = doc.body.meta
    assert meta is not None
    assert meta.language is not None
    assert meta.language.code == HumanLanguageLabel.EN
    assert meta.language.created_by == LANGUAGE_SOURCE_PDF_CATALOG
    custom = meta.get_custom_part()
    assert custom["pdf__lang"] == "en-US"
    assert custom["dc__title"] == "Fixture title"
    assert custom["pdf__producer"] == "docling-machine-test-fixture"
    # The page content is unaffected.
    assert "Document metadata fixture" in doc.export_to_markdown()


def test_pipeline_leaves_body_meta_unset_by_default():
    doc = _convert(attach_document_metadata=False)

    assert doc.body.meta is None
