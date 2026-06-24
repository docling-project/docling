from io import BytesIO
from pathlib import Path

import pytest
from requests import ConnectionError, Response, Session

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import ConversionError, DocumentConverter

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
