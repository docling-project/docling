import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
    QualityGrade,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.image_classification_engine_options import (
    ApiKserveV2ImageClassificationEngineOptions,
)
from docling.datamodel.pipeline_options import (
    NemotronOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    ConversionError,
    DocumentConverter,
    PdfFormatOption,
)
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr.nemotron_ocr_model import NemotronOcrModel
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline


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
    ao4 = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.XPU)
    assert ao2.num_threads == 2
    assert ao2.device == AcceleratorDevice.MPS
    assert ao3.num_threads == 3
    assert ao3.device == AcceleratorDevice.CUDA
    assert ao4.num_threads == 4
    assert ao4.device == AcceleratorDevice.XPU

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
    ao5 = AcceleratorOptions(num_threads=5, device=AcceleratorDevice.MPS)
    assert ao5.num_threads == 5
    assert ao5.device == AcceleratorDevice.MPS

    # Use regular and alternative envvar
    os.environ["DOCLING_NUM_THREADS"] = "2"
    ao6 = AcceleratorOptions()
    assert ao6.num_threads == 2
    assert ao6.device == AcceleratorDevice.CPU

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
    ao7 = AcceleratorOptions()
    assert ao7.num_threads == 4
    assert ao7.device == AcceleratorDevice.AUTO


def test_kserve_v2_binary_data_deprecated_alias():
    options = ApiKserveV2ImageClassificationEngineOptions(
        url="localhost:8001",
        grpc_use_binary_data=False,
    )

    assert options.use_binary_data is False
    with pytest.deprecated_call(match="deprecated; use use_binary_data instead"):
        assert options.grpc_use_binary_data is False
    assert "grpc_use_binary_data" not in options.model_dump()

    options = ApiKserveV2ImageClassificationEngineOptions(
        url="localhost:8001",
        use_binary_data=True,
        grpc_use_binary_data=False,
    )
    assert options.use_binary_data is True


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


def test_document_timeout(test_doc_path):
    from docling.datamodel.base_models import FailureCategory

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(document_timeout=1)
            )
        }
    )
    result = converter.convert(test_doc_path)
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, (
        "Expected document timeout to result in PARTIAL_SUCCESS status"
    )
    # Verify timeout error is present
    assert result.has_timeout_errors(), "Expected timeout errors to be recorded"
    assert any(e.category == FailureCategory.TIMEOUT for e in result.errors), (
        "Expected at least one error with TIMEOUT category"
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(document_timeout=1),
                pipeline_cls=LegacyStandardPdfPipeline,
            )
        }
    )
    result = converter.convert(test_doc_path)
    assert result.status == ConversionStatus.PARTIAL_SUCCESS, (
        "Expected document timeout to result in PARTIAL_SUCCESS status"
    )
    # Verify timeout error is present for legacy pipeline too
    assert result.has_timeout_errors(), (
        "Expected timeout errors to be recorded in legacy pipeline"
    )


def test_invalid_input_over_max_file_size(test_doc_path):
    from docling.datamodel.base_models import FailureCategory

    converter = DocumentConverter()
    result = converter.convert(test_doc_path, raises_on_error=False, max_file_size=10)
    assert result.status == ConversionStatus.FAILURE
    assert result.errors, "over-max_file_size must produce a non-empty errors list"
    assert result.errors[0].category == FailureCategory.POLICY


def test_invalid_input_over_max_num_pages(test_doc_path):
    from docling.datamodel.base_models import FailureCategory

    converter = DocumentConverter()
    result = converter.convert(test_doc_path, raises_on_error=False, max_num_pages=1)
    assert result.status == ConversionStatus.FAILURE
    assert result.errors, "over-max_num_pages must produce a non-empty errors list"
    assert result.errors[0].category == FailureCategory.POLICY


def test_invalid_input_unreadable_source():
    from io import BytesIO

    from docling.datamodel.base_models import DocumentStream, FailureCategory

    # Bytes that do not parse as any allowed format -> backend rejects them.
    converter = DocumentConverter()
    stream = DocumentStream(name="broken.pdf", stream=BytesIO(b"%PDF-1.4 not really"))
    result = converter.convert(stream, raises_on_error=False)
    assert result.status == ConversionStatus.FAILURE
    assert result.errors, "unreadable source must produce a non-empty errors list"
    assert result.errors[0].category == FailureCategory.BACKEND_FAILURE


def test_invalid_input_unreadable_source_with_exception_surfaces_error_details():
    from io import BytesIO

    import docling.backend.docling_parse_backend as docling_parse_backend_module
    from docling.datamodel.base_models import DocumentStream

    converter = DocumentConverter()
    stream = DocumentStream(name="broken.pdf", stream=BytesIO(b"broken"))

    with (
        patch.object(
            docling_parse_backend_module.pdfium,
            "PdfDocument",
            side_effect=RuntimeError("bad trailer"),
        ),
        pytest.raises(
            ConversionError,
            match=r"Conversion failed for: broken\.pdf with status: failure\. Errors: docling-parse could not load document .*: bad trailer",
        ),
    ):
        converter.convert(stream, raises_on_error=True)


def test_invalid_input_unreachable_source():
    """A source that cannot be resolved is SOURCE_UNAVAILABLE, no network needed."""
    from docling.datamodel.base_models import FailureCategory

    # Drive the source-resolution failure synthetically rather than relying on
    # real DNS/network behavior for an "unreachable" URL.
    with patch(
        "docling.datamodel.document.resolve_source_to_stream",
        side_effect=OSError("connection refused"),
    ):
        converter = DocumentConverter()
        result = converter.convert("https://example.com/foo.pdf", raises_on_error=False)
    assert result.status == ConversionStatus.FAILURE
    assert result.errors, "unreachable source must produce a non-empty errors list"
    assert result.errors[0].category == FailureCategory.SOURCE_UNAVAILABLE


def test_convert_unsupported_format_with_exception_surfaces_error_details():
    from io import BytesIO

    with pytest.raises(
        ConversionError,
        match=r"Conversion failed for: input\.xyz with status: skipped\. Errors: File format not allowed: input\.xyz",
    ):
        DocumentConverter().convert(
            DocumentStream(name="input.xyz", stream=BytesIO(b"xyz")),
            raises_on_error=True,
        )


def test_backend_parse_error_is_backend_failure():
    """DocumentLoadError from backend init is the parse-failure signal."""
    from io import BytesIO

    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.base_models import FailureCategory
    from docling.datamodel.document import InputDocument, build_invalid_input_errors
    from docling.exceptions import DocumentLoadError

    class _ParseFailBackend(AbstractDocumentBackend):
        def __init__(self, *args, **kwargs):
            raise DocumentLoadError("these bytes are not a valid PDF")

        def is_valid(self) -> bool:
            return False

        @classmethod
        def supported_formats(cls):
            return {InputFormat.PDF}

        @classmethod
        def supports_pagination(cls) -> bool:
            return False

        def unload(self):
            pass

    doc = InputDocument(
        path_or_stream=BytesIO(b"anything"),
        format=InputFormat.PDF,
        backend=_ParseFailBackend,
        filename="x.pdf",
    )
    assert doc.valid is False
    error = build_invalid_input_errors(doc)[0]
    assert error.category == FailureCategory.BACKEND_FAILURE
    # The DocumentLoadError message is surfaced, not a generic placeholder.
    assert error.error_message == "these bytes are not a valid PDF"


def test_bare_runtime_error_from_backend_is_unknown():
    """A bare RuntimeError is an internal defect, not the bad-input signal.

    Only DocumentLoadError is treated as bad input. A bare RuntimeError is
    caught by __init__'s handler and recorded as UNKNOWN, not BACKEND_FAILURE.
    """
    from io import BytesIO

    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.base_models import FailureCategory
    from docling.datamodel.document import InputDocument, build_invalid_input_errors

    class _BareRuntimeErrorBackend(AbstractDocumentBackend):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("internal defect")

        def is_valid(self) -> bool:
            return False

        @classmethod
        def supported_formats(cls):
            return {InputFormat.PDF}

        @classmethod
        def supports_pagination(cls) -> bool:
            return False

        def unload(self):
            pass

    doc = InputDocument(
        path_or_stream=BytesIO(b"anything"),
        format=InputFormat.PDF,
        backend=_BareRuntimeErrorBackend,
        filename="x.pdf",
    )
    assert doc.valid is False
    assert build_invalid_input_errors(doc)[0].category == FailureCategory.UNKNOWN


def test_backend_value_error_propagates():
    """Backend ValueError is not rewritten into an invalid-input result."""
    from io import BytesIO

    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.document import InputDocument

    class _ValueErrorBackend(AbstractDocumentBackend):
        def __init__(self, *args, **kwargs):
            raise ValueError("backend limit exceeded")

        def is_valid(self) -> bool:
            return False

        @classmethod
        def supported_formats(cls):
            return {InputFormat.PDF}

        @classmethod
        def supports_pagination(cls) -> bool:
            return False

        def unload(self):
            pass

    with pytest.raises(ValueError, match="backend limit exceeded"):
        InputDocument(
            path_or_stream=BytesIO(b"anything"),
            format=InputFormat.PDF,
            backend=_ValueErrorBackend,
            filename="x.pdf",
        )


def test_internal_backend_error_is_not_masked_as_bad_input():
    """A non-RuntimeError init failure (e.g. missing dep) is not BACKEND_FAILURE.

    Such an exception is not the backends' bad-input signal, so it is left to
    propagate rather than being silently recorded as a parse failure.
    """
    from io import BytesIO

    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.document import InputDocument

    class _MissingDepBackend(AbstractDocumentBackend):
        def __init__(self, *args, **kwargs):
            raise ImportError("optional dependency 'foo' is not installed")

        def is_valid(self) -> bool:
            return False

        @classmethod
        def supported_formats(cls):
            return {InputFormat.PDF}

        @classmethod
        def supports_pagination(cls) -> bool:
            return False

        def unload(self):
            pass

    with pytest.raises(ImportError):
        InputDocument(
            path_or_stream=BytesIO(b"anything"),
            format=InputFormat.PDF,
            backend=_MissingDepBackend,
            filename="x.pdf",
        )


def test_page_error_carries_page_no(test_doc_path):
    """Page-scoped errors set page_no rather than prefixing the message."""
    from docling.datamodel.base_models import ErrorItem

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(document_timeout=1)
            )
        }
    )
    result = converter.convert(test_doc_path)
    # The timeout filters uninitialized pages, surfacing per-page parse errors.
    page_errors = [e for e in result.errors if e.page_no is not None]
    for err in page_errors:
        assert isinstance(err, ErrorItem)
        assert err.page_no >= 1
        assert not err.error_message.startswith("Page ")


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


def test_nemotron_ocr_backend_registration():
    factory = get_ocr_factory(allow_external_plugins=False)

    model = factory.create_instance(
        options=NemotronOcrOptions(),
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
    )

    assert isinstance(model, NemotronOcrModel)


def test_parser_backends(test_doc_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    for backend_t in [
        DoclingParseDocumentBackend,
        ThreadedDoclingParseDocumentBackend,
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


def test_pipeline_cache_after_initialize(test_doc_path):
    """Test that initialize_pipeline caches correctly and convert reuses the cache.

    Regression test for #3109: code_formula_options were mutated in-place during
    pipeline initialization, changing the options hash and causing a cache miss
    when convert() was called afterwards.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    converter.initialize_pipeline(InputFormat.PDF)
    assert len(converter._get_initialized_pipelines()) == 1

    converter.convert(test_doc_path)
    assert len(converter._get_initialized_pipelines()) == 1, (
        "Pipeline should be reused from cache, not re-initialized"
    )


def test_confidence(test_doc_path):
    converter = DocumentConverter()
    doc_result: ConversionResult = converter.convert(test_doc_path, page_range=(6, 9))

    assert doc_result.confidence.mean_grade == QualityGrade.EXCELLENT
    assert doc_result.confidence.low_grade in (
        QualityGrade.GOOD,
        QualityGrade.EXCELLENT,
    )


def test_pipeline_cache_with_chart_extraction():
    """Test that chart extraction doesn't cause pipeline cache invalidation.

    Verifies the fix for a bug where enabling chart extraction mutated shared
    pipeline_options, changing its hash and causing unnecessary re-initialization.
    """

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_chart_extraction = True

    with (
        patch(
            "docling.pipeline.base_pipeline.ChartExtractionModelGraniteVisionV4"
        ) as mock_chart,
        patch(
            "docling.pipeline.base_pipeline.DocumentPictureClassifier"
        ) as mock_classifier,
    ):
        mock_chart.return_value = Mock(enabled=True)
        mock_classifier.return_value = Mock(enabled=True)

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        converter.initialize_pipeline(InputFormat.PDF)
        assert len(converter._get_initialized_pipelines()) == 1

        converter._get_pipeline(InputFormat.PDF)
        assert len(converter._get_initialized_pipelines()) == 1, (
            "Pipeline should be reused from cache, not re-initialized"
        )
