from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import ConversionResult, InputDocument

def test_conversion_result_json_roundtrip_string():
    pdf_doc = Path("./tests/data/pdf/redp5110_sampled.pdf")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
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
    
    json_str = conv_res.save_as_json("./test-conversion.json")  # returns string when no filename is given
    print(json_str)
    
    assert isinstance(json_str, str) and len(json_str) > 0

    loaded = ConversionResult.load_from_json("./test-conversion.json")

    assert loaded.status == conv.status
    assert loaded.input.valid is True
    assert loaded.input.file.name == conv.input.file.name
    assert loaded.document.name == conv.document.name


