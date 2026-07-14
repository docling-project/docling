from io import BytesIO
from pathlib import Path

import pytest
from requests import ConnectionError, Response, Session

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
    pdf_path = Path("./tests/data/pdf/sources/2305.03393v1-pg9.pdf")
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


def test_convert_remote_too_large_filesize_limit_wout_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        r = Response()
        r.status_code = 200
        r.headers["Content-Length"] = "10"
        # A real streamed response always has a closeable ``raw``; provide one so
        # resolve_source_to_stream can close the response on the size-limit abort.
        r.raw = BytesIO(b"")
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    result = converter.convert(
        "https://example.com/input.pdf",
        max_file_size=5,
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE
    assert result.input.file.name == "input.pdf"
    assert result.input.filesize == 10
    assert result.input.valid is False


def test_convert_remote_too_large_filesize_limit_with_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        r = Response()
        r.status_code = 200
        r.headers["Content-Length"] = "10"
        # A real streamed response always has a closeable ``raw``; provide one so
        # resolve_source_to_stream can close the response on the size-limit abort.
        r.raw = BytesIO(b"")
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    with pytest.raises(ConversionError):
        converter.convert(
            "https://example.com/input.pdf",
            max_file_size=5,
            raises_on_error=True,
        )


def test_convert_unreachable_url_wout_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        raise ConnectionError("Failed to establish a new connection")

    monkeypatch.setattr(Session, "get", mock_get)

    result = converter.convert(
        "https://example.com/input.pdf?token=secret",
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE
    # Query string is dropped so the filename/extension stay recoverable.
    assert result.input.file.name == "input.pdf"
    assert result.input.valid is False


def test_convert_unreachable_url_with_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        raise ConnectionError("Failed to establish a new connection")

    monkeypatch.setattr(Session, "get", mock_get)

    with pytest.raises(ConversionError):
        converter.convert(
            "https://example.com/input.pdf",
            raises_on_error=True,
        )


def test_convert_http_error_status_wout_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        r = Response()
        r.status_code = 404
        r.url = "https://example.com/input.pdf"
        r.raw = BytesIO(b"")
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    result = converter.convert(
        "https://example.com/input.pdf",
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE
    assert result.input.valid is False


def test_convert_all_unreachable_url_does_not_abort_batch(
    converter: DocumentConverter, monkeypatch
):
    """One unreachable source must not fail its sibling documents in the batch."""

    def mock_get(self, *args, **kwargs):
        raise ConnectionError("Failed to establish a new connection")

    monkeypatch.setattr(Session, "get", mock_get)

    sources = [
        DocumentStream(name="ok.md", stream=BytesIO(b"# Hello")),
        "https://example.com/missing.pdf",
    ]
    results = list(converter.convert_all(sources, raises_on_error=False))

    assert len(results) == 2
    statuses = {res.input.file.name: res.status for res in results}
    assert statuses["ok.md"] == ConversionStatus.SUCCESS
    assert statuses["missing.pdf"] == ConversionStatus.FAILURE


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


def test_convert_invalid_pdf_bytes_chains_original_exception(
    converter: DocumentConverter,
):
    """Regression test for #1920: ConversionError must preserve the underlying
    backend exception via ``__cause__`` so applications can classify failures
    (e.g. encrypted vs. corrupted PDFs) programmatically."""
    from docling.exceptions import DocumentLoadError

    stream = DocumentStream(name="invalid.pdf", stream=BytesIO(b"not a valid pdf file"))
    with pytest.raises(ConversionError) as exc_info:
        converter.convert(stream, raises_on_error=True)

    cause = exc_info.value.__cause__
    assert cause is not None, "ConversionError should chain the backend exception"
    assert isinstance(cause, DocumentLoadError)
    # The backend itself chains the parser error (e.g. PdfiumError), so the
    # root cause stays reachable for fine-grained classification.
    assert cause.__cause__ is not None


def test_convert_invalid_pdf_bytes_wout_exception_records_original_error(
    converter: DocumentConverter,
):
    """With raises_on_error=False the rejection still captures the original
    exception, keeping both error-handling modes equally informative."""
    stream = DocumentStream(name="invalid.pdf", stream=BytesIO(b"not a valid pdf file"))
    result = converter.convert(stream, raises_on_error=False)

    assert result.status == ConversionStatus.FAILURE
    rejection = result.input._rejection
    assert rejection is not None
    assert rejection.original_error is not None


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
