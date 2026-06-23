"""
Test unit for document extraction functionality.
"""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.document_extractor import DocumentExtractor

IS_CI = bool(os.getenv("CI"))


class ExampleTemplate(BaseModel):
    bill_no: str = Field(
        examples=["A123", "5414"]
    )  # provide some examples, but not the actual value of the test sample
    total: float = Field(
        default=10.0, examples=[20.0]
    )  # provide a default value and some examples


@pytest.fixture
def extractor() -> DocumentExtractor:
    """Create a document converter instance for testing."""

    return DocumentExtractor(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])


@pytest.fixture
def test_file_path() -> Path:
    """Get the path to the test QR bill image."""
    return Path(__file__).parent / "data_scanned" / "qr_bill_example.jpg"
    # return Path("tests/data/pdf/code_and_formula.pdf")


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_string_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using string template."""
    str_templ = '{"bill_no": "string", "total": "number"}'

    result = extractor.extract(test_file_path, template=str_templ)

    print(result.pages)

    assert result.status is not None
    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_dict_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using dictionary template."""
    dict_templ = {
        "bill_no": "string",
        "total": "number",
    }

    result = extractor.extract(test_file_path, template=dict_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_pydantic_instance_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using pydantic instance template."""
    pydantic_instance_templ = ExampleTemplate(bill_no="4321")

    result = extractor.extract(test_file_path, template=pydantic_instance_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_pydantic_class_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using pydantic class template."""
    pydantic_class_templ = ExampleTemplate

    result = extractor.extract(test_file_path, template=pydantic_class_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


def test_extraction_format_not_allowed_is_policy() -> None:
    """A disallowed input format yields a SKIPPED result with a POLICY error."""
    from docling.datamodel.base_models import ConversionStatus, FailureCategory

    # Allow only PDF, then feed the JPEG sample so the format is rejected.
    pdf_only = DocumentExtractor(allowed_formats=[InputFormat.PDF])
    img = Path(__file__).parent / "data_scanned" / "qr_bill_example.jpg"
    result = pdf_only.extract(
        img, template='{"bill_no": "string"}', raises_on_error=False
    )

    assert result.status == ConversionStatus.SKIPPED
    assert result.errors, "format-not-allowed must produce a non-empty errors list"
    assert result.errors[0].category == FailureCategory.POLICY


def test_threaded_model_stage_failure_records_inference_category() -> None:
    from types import SimpleNamespace

    from docling.datamodel.base_models import (
        DoclingComponentType,
        FailureCategory,
        Page,
    )
    from docling.pipeline.standard_pdf_pipeline import (
        ThreadedItem,
        ThreadedPipelineStage,
    )

    def _raise(_conv_res, _pages):
        raise RuntimeError("ocr failed")

    stage = ThreadedPipelineStage(
        name="ocr",
        model=_raise,
        batch_size=1,
        batch_timeout=0.0,
        queue_max_size=1,
    )
    item = ThreadedItem(
        payload=Page(page_no=1),
        run_id=1,
        page_no=1,
        conv_res=SimpleNamespace(),
    )

    result = stage._process_batch([item])

    assert len(result) == 1
    assert result[0].is_failed
    assert result[0].failure is not None
    assert result[0].failure.component_type == DoclingComponentType.MODEL
    assert result[0].failure.category == FailureCategory.INFERENCE_FAILURE
    assert result[0].failure.page_no == 1


def test_threaded_model_stage_preserves_existing_failed_item_category() -> None:
    from types import SimpleNamespace

    from docling.datamodel.base_models import (
        DoclingComponentType,
        ErrorItem,
        FailureCategory,
        Page,
    )
    from docling.pipeline.standard_pdf_pipeline import (
        ThreadedItem,
        ThreadedPipelineStage,
    )

    def _raise(_conv_res, _pages):
        raise RuntimeError("ocr failed")

    prior_failure = ErrorItem(
        component_type=DoclingComponentType.DOCUMENT_BACKEND,
        module_name="preprocess",
        error_message="Page 1 failed to parse.",
        category=FailureCategory.BACKEND_FAILURE,
        page_no=1,
    )
    stage = ThreadedPipelineStage(
        name="ocr",
        model=_raise,
        batch_size=2,
        batch_timeout=0.0,
        queue_max_size=1,
    )
    already_failed = ThreadedItem(
        payload=Page(page_no=1),
        run_id=1,
        page_no=1,
        conv_res=SimpleNamespace(),
        error=RuntimeError("Page 1 failed to parse."),
        failure=prior_failure,
        is_failed=True,
    )
    valid = ThreadedItem(
        payload=Page(page_no=2),
        run_id=1,
        page_no=2,
        conv_res=SimpleNamespace(),
    )

    result = stage._process_batch([already_failed, valid])

    assert result[0].failure == prior_failure
    assert result[0].error is already_failed.error
    assert result[1].failure is not None
    assert result[1].failure.category == FailureCategory.INFERENCE_FAILURE


def test_standard_pipeline_integrate_preserves_failed_page_category() -> None:
    from types import SimpleNamespace

    from docling.datamodel.base_models import (
        ConversionStatus,
        DoclingComponentType,
        ErrorItem,
        FailureCategory,
        Page,
    )
    from docling.pipeline.standard_pdf_pipeline import (
        ProcessingResult,
        StandardPdfPipeline,
    )

    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline.keep_images = False
    pipeline.keep_backend = False
    pipeline.pipeline_options = SimpleNamespace(generate_parsed_pages=False)

    failure = ErrorItem(
        component_type=DoclingComponentType.MODEL,
        module_name="ocr",
        error_message="ocr failed",
        category=FailureCategory.INFERENCE_FAILURE,
        page_no=1,
    )
    conv_res = SimpleNamespace(pages=[Page(page_no=1)], errors=[], status=None)

    pipeline._integrate_results(
        conv_res,
        ProcessingResult(
            failed_pages=[(1, RuntimeError("ocr failed"), failure)],
            total_expected=1,
        ),
    )

    assert conv_res.status == ConversionStatus.FAILURE
    assert conv_res.errors == [failure]


def test_extraction_vlm_pipeline_runtime_failure_is_unknown() -> None:
    from types import SimpleNamespace

    from docling.datamodel.base_models import FailureCategory
    from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)

    def _raise(_input_doc):
        raise RuntimeError("image extraction failed")

    pipeline._get_images_from_input = _raise
    ext_res = SimpleNamespace(
        input=SimpleNamespace(_backend=object()),
        pages=[],
        errors=[],
        status=None,
    )

    result = pipeline._extract_data(ext_res)

    assert result.errors
    assert result.errors[0].category == FailureCategory.UNKNOWN


def test_extraction_pipeline_failure_is_categorized() -> None:
    """A failing extraction pipeline records a PIPELINE/UNKNOWN ErrorItem."""
    from docling.datamodel.base_models import (
        ConversionStatus,
        DoclingComponentType,
        FailureCategory,
    )
    from docling.datamodel.extraction import ExtractionResult
    from docling.datamodel.pipeline_options import PipelineOptions
    from docling.pipeline.base_extraction_pipeline import BaseExtractionPipeline

    class _FailingPipeline(BaseExtractionPipeline):
        def _extract_data(self, ext_res, template=None):
            raise RuntimeError("boom")

        def _determine_status(self, ext_res):
            return ConversionStatus.SUCCESS

        @classmethod
        def get_default_options(cls):
            return PipelineOptions()

    # Build a minimal valid InputDocument from the sample image.
    img = Path(__file__).parent / "data_scanned" / "qr_bill_example.jpg"
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.datamodel.document import InputDocument

    input_doc = InputDocument(
        path_or_stream=img,
        format=InputFormat.IMAGE,
        backend=DoclingParseV4DocumentBackend,
        filename=img.name,
    )

    pipeline = _FailingPipeline(PipelineOptions())
    result: ExtractionResult = pipeline.execute(input_doc, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    assert result.errors
    err = result.errors[0]
    assert err.component_type == DoclingComponentType.PIPELINE
    assert err.category == FailureCategory.UNKNOWN
