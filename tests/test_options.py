import os
from pathlib import Path

import pytest

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat, QualityGrade
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf/2206.01062.pdf")


def get_converters_with_table_options():
    for cell_matching in [True, False]:
        for mode in [TableFormerMode.FAST, TableFormerMode.ACCURATE]:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = cell_matching
            pipeline_options.table_structure_options.mode = mode

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            yield converter


def test_accelerator_options():
    # Check the default options
    ao = AcceleratorOptions()
    assert ao.num_threads == 4, "Wrong default num_threads"
    assert ao.device == AcceleratorDevice.AUTO, "Wrong default device"

    # Use API
    ao2 = AcceleratorOptions(num_threads=2, device=AcceleratorDevice.MPS)
    ao3 = AcceleratorOptions(num_threads=3, device=AcceleratorDevice.CUDA)
    assert ao2.num_threads == 2
    assert ao2.device == AcceleratorDevice.MPS
    assert ao3.num_threads == 3
    assert ao3.device == AcceleratorDevice.CUDA

    # Use envvars (regular + alternative) and default values
    os.environ["OMP_NUM_THREADS"] = "1"
    ao.__init__()
    assert ao.num_threads == 1
    assert ao.device == AcceleratorDevice.AUTO
    os.environ["DOCLING_DEVICE"] = "cpu"
    ao.__init__()
    assert ao.device == AcceleratorDevice.CPU
    assert ao.num_threads == 1

    # Use envvars and override in init
    os.environ["DOCLING_DEVICE"] = "cpu"
    ao4 = AcceleratorOptions(num_threads=5, device=AcceleratorDevice.MPS)
    assert ao4.num_threads == 5
    assert ao4.device == AcceleratorDevice.MPS

    # Use regular and alternative envvar
    os.environ["DOCLING_NUM_THREADS"] = "2"
    ao5 = AcceleratorOptions()
    assert ao5.num_threads == 2
    assert ao5.device == AcceleratorDevice.CPU

    # Use wrong values
    is_exception = False
    try:
        os.environ["DOCLING_DEVICE"] = "wrong"
        ao5.__init__()
    except Exception as ex:
        print(ex)
        is_exception = True
    assert is_exception

    # Use misformatted alternative envvar
    del os.environ["DOCLING_NUM_THREADS"]
    del os.environ["DOCLING_DEVICE"]
    os.environ["OMP_NUM_THREADS"] = "wrong"
    ao6 = AcceleratorOptions()
    assert ao6.num_threads == 4
    assert ao6.device == AcceleratorDevice.AUTO


def test_e2e_conversions(test_doc_path):
    for converter in get_converters_with_table_options():
        print(f"converting {test_doc_path}")

        doc_result: ConversionResult = converter.convert(test_doc_path)

        assert doc_result.status == ConversionStatus.SUCCESS


def test_page_range(test_doc_path):
    converter = DocumentConverter()
    doc_result: ConversionResult = converter.convert(test_doc_path, page_range=(9, 9))

    assert doc_result.status == ConversionStatus.SUCCESS
    assert doc_result.input.page_count == 9
    assert doc_result.document.num_pages() == 1

    doc_result: ConversionResult = converter.convert(
        test_doc_path, page_range=(10, 10), raises_on_error=False
    )
    assert doc_result.status == ConversionStatus.FAILURE


def test_ocr_coverage_threshold(test_doc_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.bitmap_area_threshold = 1.1

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    test_doc_path = Path("./tests/data_scanned/ocr_test.pdf")
    doc_result: ConversionResult = converter.convert(test_doc_path)

    # this should have generated no results, since we set a very high threshold
    assert len(doc_result.document.texts) == 0


def test_parser_backends(test_doc_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    for backend_t in [
        DoclingParseV4DocumentBackend,
        DoclingParseV2DocumentBackend,
        DoclingParseDocumentBackend,
        PyPdfiumDocumentBackend,
    ]:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=backend_t,
                )
            }
        )

        test_doc_path = Path("./tests/data/pdf/code_and_formula.pdf")
        doc_result: ConversionResult = converter.convert(test_doc_path)

        assert doc_result.status == ConversionStatus.SUCCESS


def test_confidence(test_doc_path):
    converter = DocumentConverter()
    doc_result: ConversionResult = converter.convert(test_doc_path, page_range=(6, 9))

    assert doc_result.confidence.mean_grade == QualityGrade.EXCELLENT
    assert doc_result.confidence.low_grade == QualityGrade.EXCELLENT
