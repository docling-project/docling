from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TextFormattingOptions,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

# Lives apart from tests/test_pdf_formatting.py because CI selects whole modules by their
# module-level marker, and this case needs the OCR lane rather than the PDF-model one.
pytestmark = pytest.mark.ml_ocr


def test_scanned_pages_carry_no_formatting() -> None:
    # OCR produces cells without font information, so nothing is claimed about their styling.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = False
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    pipeline_options.text_formatting_options = TextFormattingOptions(enabled=True)

    input_document = InputDocument(
        path_or_stream=Path("tests/data/ocr/sources/ocr_test.pdf"),
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
    )
    result = StandardPdfPipeline(pipeline_options).execute(
        input_document, raises_on_error=True
    )

    assert result.document.texts
    assert all(t.formatting is None for t in result.document.texts)
