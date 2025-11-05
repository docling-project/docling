from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
)
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf_password/2206.01062_pg3.pdf")


@dataclass
class TestOption:
    options: PdfFormatOption
    name: str


def converter_opts_gen() -> Iterable[TestOption]:
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
    )

    backend_options = PdfBackendOptions(password="1234")

    yield TestOption(
        options=PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend,
            backend_options=backend_options,
        ),
        name="PyPdfium",
    )

    yield TestOption(
        options=PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseV4DocumentBackend,
            backend_options=backend_options,
        ),
        name="DoclingParseV4",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("test_options", converter_opts_gen(), ids=lambda o: o.name)
def test_get_text_from_rect(test_doc_path: Path, test_options: TestOption):
    converter = DocumentConverter(
        format_options={InputFormat.PDF: test_options.options}
    )

    res = converter.convert(test_doc_path)
    assert res.status == ConversionStatus.SUCCESS
