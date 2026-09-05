# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""End-to-end tests for RTF conversion through the Word backend."""

from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.docx.drawingml.utils import get_libreoffice_cmd
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter

RTF_PATH = Path("tests/data/rtf/sources/legacy_sample.rtf")


@pytest.mark.skipif(
    get_libreoffice_cmd() is None,
    reason="LibreOffice not available",
)
@pytest.mark.parametrize("use_stream", [False, True], ids=["path", "stream"])
def test_rtf_conversion_preserves_content_and_origin(use_stream: bool):
    """RTF input is converted to DOCX and parsed without changing its origin."""
    source: Path | DocumentStream
    if use_stream:
        source = DocumentStream(
            name=RTF_PATH.name,
            stream=BytesIO(RTF_PATH.read_bytes()),
        )
    else:
        source = RTF_PATH

    result = DocumentConverter(allowed_formats=[InputFormat.RTF]).convert(source)
    document = result.document
    markdown = document.export_to_markdown()

    assert document.origin is not None
    assert document.origin.filename == RTF_PATH.name
    assert document.origin.mimetype == "application/rtf"
    assert "RTF support fixture" in markdown
    assert "Unicode text: café" in markdown
    assert "First cell" in markdown
    assert "Second cell" in markdown
    assert len(document.tables) == 1


def test_word_backend_advertises_rtf_support():
    assert InputFormat.RTF in MsWordDocumentBackend.supported_formats()
