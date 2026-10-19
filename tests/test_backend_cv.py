# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for .cv resume files.

The fixtures were built with the cvfile Python SDK (https://github.com/cvfile/cv)
from public facts about Marie Curie. Their visual PDF layer is a blank page, so
any text in the output comes from the embedded payload.
"""

from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import (
    CvFormatOption,
    DocumentConverter,
    PdfFormatOption,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

CV_SOURCES = Path("./tests/data/cv/sources")
MARKDOWN_CV = CV_SOURCES / "marie_curie.cv"
HTML_CV = CV_SOURCES / "marie_curie_html.cv"


@pytest.mark.parametrize("source", [MARKDOWN_CV, HTML_CV])
def test_e2e_cv_conversion_matches_groundtruth(source: Path):
    result = DocumentConverter(allowed_formats=[InputFormat.CV]).convert(source)
    groundtruth = source.parent.parent / "groundtruth" / source.name

    assert result.status is ConversionStatus.SUCCESS
    assert result.document.origin.mimetype == "application/vnd.cv+pdf"
    assert verify_document(
        result.document,
        str(groundtruth) + ".json",
        generate=GEN_TEST_DATA,
    ), "export to JSON"
    assert verify_export(
        result.document.export_to_markdown(),
        str(groundtruth) + ".md",
        generate=GEN_TEST_DATA,
    ), "export to Markdown"


@pytest.mark.parametrize(
    ("name", "allowed", "expected"),
    [
        ("marie_curie.cv", [InputFormat.CV, InputFormat.PDF], InputFormat.CV),
        ("marie_curie.pdf", [InputFormat.CV, InputFormat.PDF], InputFormat.CV),
        ("marie_curie.cv", [InputFormat.PDF], InputFormat.PDF),
        ("marie_curie.pdf", [InputFormat.PDF], InputFormat.PDF),
    ],
)
def test_cv_is_detected_only_when_the_cv_format_is_allowed(
    name: str, allowed: list[InputFormat], expected: InputFormat
):
    options = {InputFormat.CV: CvFormatOption(), InputFormat.PDF: PdfFormatOption()}
    stream = DocumentStream(name=name, stream=BytesIO(MARKDOWN_CV.read_bytes()))
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[stream])

    [in_doc] = conversion_input.docs({fmt: options[fmt] for fmt in allowed})

    assert in_doc.format is expected


@pytest.mark.parametrize("name", ["marie_curie.cv", "marie_curie.pdf"])
def test_cv_converts_as_pdf_without_cv_dependencies(name: str, monkeypatch):
    # A slim install without the format-cv extra keeps converting these files
    # as PDFs, as it did before the CV format existed.
    monkeypatch.setattr("docling.backend.cv_backend._CV_DEPENDENCIES_AVAILABLE", False)
    stream = DocumentStream(name=name, stream=BytesIO(MARKDOWN_CV.read_bytes()))
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[stream])

    [in_doc] = conversion_input.docs(
        {InputFormat.CV: CvFormatOption(), InputFormat.PDF: PdfFormatOption()}
    )

    assert in_doc.format is InputFormat.PDF


def test_pdf_without_cv_metadata_stays_pdf():
    source = Path("./tests/data/pdf/bookmark_sample.pdf")
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[source])

    [in_doc] = conversion_input.docs(
        {InputFormat.CV: CvFormatOption(), InputFormat.PDF: PdfFormatOption()}
    )

    assert in_doc.format is InputFormat.PDF


def test_cv_with_tampered_payload_fails():
    # Change one byte of the stored Markdown so it no longer matches the
    # sha-256 digest recorded in cv:integrity.
    data = MARKDOWN_CV.read_bytes().replace(b"Physicist", b"Physicisx", 1)
    result = DocumentConverter(allowed_formats=[InputFormat.CV]).convert(
        DocumentStream(name="tampered.cv", stream=BytesIO(data)),
        raises_on_error=False,
    )

    assert result.status is ConversionStatus.FAILURE
