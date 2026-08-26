from pathlib import Path

import pytest
from docling_core.types.doc.common.formatting import Formatting
from docling_core.types.doc.document import SectionHeaderItem

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TextFormattingOptions,
)
from docling.datamodel.settings import DocumentLimits
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model

PIPELINES = [StandardPdfPipeline, LegacyStandardPdfPipeline]


def _convert(
    pipeline_cls,
    source: str,
    *,
    enabled: bool = True,
    page_range: tuple[int, int] | None = None,
) -> ConversionResult:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    if enabled:
        pipeline_options.text_formatting_options = TextFormattingOptions(enabled=True)

    limits = DocumentLimits(page_range=page_range) if page_range else DocumentLimits()
    input_document = InputDocument(
        path_or_stream=Path(source),
        format=InputFormat.PDF,
        backend=DoclingParseDocumentBackend,
        limits=limits,
    )
    return pipeline_cls(pipeline_options).execute(input_document, raises_on_error=True)


@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_italic_body_text_is_recovered(pipeline_cls) -> None:
    result = _convert(
        pipeline_cls, "tests/data/pdf/sources/2203.01017v2.pdf", page_range=(1, 1)
    )
    texts = result.document.texts

    # The abstract is set in NimbusRomNo9L-ReguItal, the body around it is not.
    abstract = next(t for t in texts if t.text.startswith("Tables organize valuable"))
    assert abstract.formatting == Formatting(italic=True)
    assert all(
        t.formatting is None
        for t in texts
        if t.text.startswith("Tables are widely used")
    )
    # Heading markup already conveys prominence, so headings are never formatted.
    assert all(t.formatting is None for t in texts if isinstance(t, SectionHeaderItem))


@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_bold_text_inside_a_picture_is_recovered(pipeline_cls) -> None:
    result = _convert(pipeline_cls, "tests/data/pdf/sources/amt_handbook_sample.pdf")
    formatting = {t.text: t.formatting for t in result.document.texts}

    # Labels inside the figure, reached through the picture-child path.
    assert formatting["Tightened nut"] == Formatting(bold=True)
    assert formatting["Untightened nut"] == Formatting(bold=True)
    # A bold label in front of an italic caption: the item is not uniformly emphasized.
    assert formatting["Figure 7-26. Self-locking nuts."] is None


@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_formatting_is_not_recovered_unless_enabled(pipeline_cls) -> None:
    result = _convert(
        pipeline_cls, "tests/data/pdf/sources/amt_handbook_sample.pdf", enabled=False
    )

    assert all(t.formatting is None for t in result.document.texts)
