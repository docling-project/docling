# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

FIXTURE = Path("tests/data/pdf/sources/malformed_table_orphan_text.pdf")

pytestmark = pytest.mark.ml_pdf_model


def test_pdf_table_orphan_text_is_preserved():
    """Text outside an incomplete TableFormer grid must remain in the document."""
    pipeline_options = PdfPipelineOptions(do_ocr=False)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    document = converter.convert(FIXTURE).document

    text_items = [text.text for text in document.texts]
    output_text = document.export_to_markdown()

    assert any("Comcast" in text for text in text_items)
    assert any("HWD" in text and "Water main" in text for text in text_items)
    assert "Project Owner" in output_text
    assert "WSDOT" in output_text
    assert "Project Description" in output_text
    assert "SR509 Completion Project" in output_text
    assert "Highway or Route" in output_text
    assert len(output_text.split()) > 100
