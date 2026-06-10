from io import BytesIO
from pathlib import Path

import pytest
from requests import Response, Session

from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import ConversionError, DocumentConverter

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


def test_convert_remote_too_large_filesize_limit_wout_exception(
    converter: DocumentConverter, monkeypatch
):
    def mock_get(self, *args, **kwargs):
        r = Response()
        r.status_code = 200
        r.headers["Content-Length"] = "10"
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
        return r

    monkeypatch.setattr(Session, "get", mock_get)

    with pytest.raises(ConversionError):
        converter.convert(
            "https://example.com/input.pdf",
            max_file_size=5,
            raises_on_error=True,
        )


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
