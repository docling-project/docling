from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.abstract_backend import (
    AbstractDocumentBackend,
    PaginatedDocumentBackend,
)
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import PipelineOptions
from docling.document_converter import ConversionError, DocumentConverter, FormatOption
from docling.pipeline.base_pipeline import BasePipeline
from docling.pipeline.simple_pipeline import SimplePipeline

pytestmark = pytest.mark.cross_platform


def get_pdf_path():
    pdf_path = Path("./tests/data/pdf/2305.03393v1-pg9.pdf")
    return pdf_path


@pytest.fixture
def converter():
    converter = DocumentConverter()

    return converter


def test_convert_unsupported_doc_format_wout_exception(converter: DocumentConverter):
    result = converter.convert(
        DocumentStream(name="input.xyz", stream=BytesIO(b"xyz")), raises_on_error=False
    )
    assert result.status == ConversionStatus.SKIPPED


def test_convert_unsupported_doc_format_with_exception(converter: DocumentConverter):
    with pytest.raises(ConversionError):
        converter.convert(
            DocumentStream(name="input.xyz", stream=BytesIO(b"xyz")),
            raises_on_error=True,
        )


def test_convert_too_small_filesize_limit_wout_exception(converter: DocumentConverter):
    result = converter.convert(get_pdf_path(), max_file_size=1, raises_on_error=False)
    assert result.status == ConversionStatus.FAILURE


def test_convert_too_small_filesize_limit_with_exception(converter: DocumentConverter):
    with pytest.raises(ConversionError):
        converter.convert(get_pdf_path(), max_file_size=1, raises_on_error=True)


def test_convert_no_pipeline_wout_exception():
    converter = DocumentConverter()
    # Bypass the model validator by setting pipeline_options to None after construction.
    # This triggers the defensive "no pipeline" code path in _execute_pipeline.
    converter.format_to_options[InputFormat.MD].pipeline_options = None
    result = converter.convert(
        DocumentStream(name="test.md", stream=BytesIO(b"# Hello")),
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE


def test_convert_no_pipeline_with_exception():
    converter = DocumentConverter()
    converter.format_to_options[InputFormat.MD].pipeline_options = None
    with pytest.raises(ConversionError):
        converter.convert(
            DocumentStream(name="test.md", stream=BytesIO(b"# Hello")),
            raises_on_error=True,
        )


def test_convert_unloads_input_backend_when_pipeline_initialization_fails():
    backends = []

    class TrackingBackend(AbstractDocumentBackend):
        def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path):
            super().__init__(in_doc, path_or_stream)
            self.unload_calls = 0
            backends.append(self)

        def is_valid(self) -> bool:
            return True

        def unload(self):
            self.unload_calls += 1
            return super().unload()

        @classmethod
        def supports_pagination(cls) -> bool:
            return False

        @classmethod
        def supported_formats(cls) -> set[InputFormat]:
            return {InputFormat.MD}

    class FailingPipeline(BasePipeline):
        def __init__(self, pipeline_options: PipelineOptions):
            raise RuntimeError("pipeline initialization failed")

        def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
            return conv_res

        def _determine_status(self, conv_res: ConversionResult) -> ConversionStatus:
            return conv_res.status

        def _unload(self, conv_res: ConversionResult) -> ConversionResult:
            return conv_res

        @classmethod
        def get_default_options(cls) -> PipelineOptions:
            return PipelineOptions()

        @classmethod
        def is_backend_supported(cls, backend: AbstractDocumentBackend) -> bool:
            return True

    stream = BytesIO(b"# Hello")
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: FormatOption(
                pipeline_cls=FailingPipeline,
                backend=TrackingBackend,
            )
        },
    )

    with pytest.raises(RuntimeError, match="pipeline initialization failed"):
        converter.convert(
            DocumentStream(name="test.md", stream=stream),
            raises_on_error=True,
        )

    assert len(backends) == 1
    assert backends[0].unload_calls == 1
    assert stream.closed


def test_convert_unloads_input_backend_when_document_is_rejected_after_opening():
    backends = []

    class TrackingPaginatedBackend(PaginatedDocumentBackend):
        def __init__(self, in_doc: InputDocument, path_or_stream: BytesIO | Path):
            super().__init__(in_doc, path_or_stream)
            self.unload_calls = 0
            backends.append(self)

        def is_valid(self) -> bool:
            return True

        def page_count(self) -> int:
            return 2

        def unload(self):
            self.unload_calls += 1
            return super().unload()

        @classmethod
        def supports_pagination(cls) -> bool:
            return True

        @classmethod
        def supported_formats(cls) -> set[InputFormat]:
            return {InputFormat.MD}

    stream = BytesIO(b"# Hello")
    converter = DocumentConverter(
        allowed_formats=[InputFormat.MD],
        format_options={
            InputFormat.MD: FormatOption(
                pipeline_cls=SimplePipeline,
                backend=TrackingPaginatedBackend,
            )
        },
    )

    result = converter.convert(
        DocumentStream(name="test.md", stream=stream),
        raises_on_error=False,
        max_num_pages=1,
    )

    assert result.status == ConversionStatus.FAILURE
    assert len(backends) == 1
    assert backends[0].unload_calls == 1
    assert stream.closed
