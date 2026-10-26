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


@pytest.mark.parametrize(
    ("old", "new"),
    [
        # Unlink the XMP stream from the catalog. Its bytes, which declare the
        # .cv namespace, stay in the file, as they could in page content.
        pytest.param(
            b"/Metadata 9 0 R", b"/Metadatx 9 0 R", id="namespace-outside-catalog"
        ),
        # A PDF tool can drop the attachments and keep the XMP metadata.
        pytest.param(b"/AF [", b"/AX [", id="no-associated-files"),
        pytest.param(
            b"cv:primaryPayload>resume.md<",
            b"cv:primaryPayload>resume.mx<",
            id="primary-payload-not-embedded",
        ),
    ],
)
def test_pdf_without_a_cv_payload_stays_pdf(old: bytes, new: bytes):
    data = MARKDOWN_CV.read_bytes()
    assert old in data
    stream = DocumentStream(
        name="resume.pdf", stream=BytesIO(data.replace(old, new, 1))
    )
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[stream])

    [in_doc] = conversion_input.docs(
        {InputFormat.CV: CvFormatOption(), InputFormat.PDF: PdfFormatOption()}
    )

    assert in_doc.format is InputFormat.PDF


def _convert_cv(data: bytes):
    return DocumentConverter(allowed_formats=[InputFormat.CV]).convert(
        DocumentStream(name="edited.cv", stream=BytesIO(data)),
        raises_on_error=False,
    )


def _convert_edited_cv(old: bytes, new: bytes):
    # Each edit keeps the byte length, so the cross-reference table stays valid.
    assert len(old) == len(new)
    data = MARKDOWN_CV.read_bytes()
    assert old in data
    return _convert_cv(data.replace(old, new, 1))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        pytest.param(b"Physicist", b"Physicisx", id="payload-digest-mismatch"),
        pytest.param(b"/AF [", b"/AX [", id="no-associated-files"),
        pytest.param(
            b"cv:primaryPayload>resume.md</cv:primaryPayload>",
            b"cv:primaryPayloax>resume.md</cv:primaryPayloax>",
            id="no-primary-payload",
        ),
        pytest.param(
            b"cv:primaryPayload>resume.md<",
            b"cv:primaryPayload>resume.mx<",
            id="unknown-primary-payload",
        ),
        pytest.param(
            b"/Subtype /text#2Fmarkdown",
            b"/Subtype /text#2Fmarkdowx",
            id="unsupported-payload-type",
        ),
    ],
)
def test_invalid_cv_fails(old: bytes, new: bytes):
    assert _convert_edited_cv(old, new).status is ConversionStatus.FAILURE


def test_cv_payload_over_the_size_limit_fails(monkeypatch):
    # The fixture's payload declares 501 bytes in /Params /Size.
    monkeypatch.setattr("docling.backend.cv_backend._MAX_PAYLOAD_BYTES", 500)
    result = _convert_cv(MARKDOWN_CV.read_bytes())

    assert result.status is ConversionStatus.FAILURE


def test_cv_with_newer_major_version_converts(caplog):
    result = _convert_edited_cv(b"cv:version>1.0<", b"cv:version>2.0<")

    assert result.status is ConversionStatus.SUCCESS
    assert "newer than the supported version" in caplog.text
