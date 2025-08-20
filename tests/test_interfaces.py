from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_conversion_result_v2

GENERATE = GEN_TEST_DATA


def get_pdf_path():
    pdf_path = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")
    return pdf_path


@pytest.fixture
def converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    pipeline_options.generate_parsed_pages = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PdfFormatOption().backend,
            )
        }
    )

    return converter


def test_convert_path(converter: DocumentConverter):
    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    doc_result = converter.convert(pdf_path)
    verify_conversion_result_v2(
        input_path=pdf_path, doc_result=doc_result, generate=GENERATE
    )


def test_convert_stream(converter: DocumentConverter):
    pdf_path = get_pdf_path()
    print(f"converting {pdf_path}")

    buf = BytesIO(pdf_path.open("rb").read())
    stream = DocumentStream(name=pdf_path.name, stream=buf)

    doc_result = converter.convert(stream)
    verify_conversion_result_v2(
        input_path=pdf_path, doc_result=doc_result, generate=GENERATE
    )
