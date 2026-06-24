from pathlib import Path

import pytest
from docling_core.types.doc.document import SectionHeaderItem

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import DocumentLimits
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model


@pytest.mark.parametrize(
    "pipeline_cls",
    [StandardPdfPipeline, LegacyStandardPdfPipeline],
)
def test_pdf_pipeline_assigns_heading_levels_from_existing_fixture(
    pipeline_cls,
) -> None:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.generate_parsed_pages = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)

    input_document = InputDocument(
        path_or_stream=Path("tests/data/pdf/regression/2203.01017v2.pdf"),
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
        limits=DocumentLimits(page_range=(1, 6)),
    )

    result = pipeline_cls(pipeline_options).execute(
        input_document, raises_on_error=True
    )
    headings = {
        item.text: item.level
        for item in result.document.texts
        if isinstance(item, SectionHeaderItem)
    }

    assert headings["1. Introduction"] == 1
    assert headings["4.1. Model architecture."] == 2
    assert headings["5.1. Implementation Details"] == 2
