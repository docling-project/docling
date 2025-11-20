from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionAssets
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def test_conversion_result_json_roundtrip_string():
    pdf_doc = Path("./tests/data/pdf/redp5110_sampled.pdf")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = False
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_parsed_pages = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )
    conv_res = doc_converter.convert(pdf_doc)

    fpath: Path = Path("./test-conversion.zip")

    conv_res.save(filename=fpath)  # returns string when no filename is given
    # assert isinstance(json_str, str) and len(json_str) > 0

    loaded = ConversionAssets.load(filename=fpath)

    assert loaded.status == conv_res.status
    assert loaded.document.name == conv_res.document.name
