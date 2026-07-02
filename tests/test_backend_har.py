import json
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.har_backend import HarDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.exceptions import DocumentLoadError

pytestmark = pytest.mark.cross_platform

_SOURCES = Path("tests/data/har/sources")


def _make_input(path_or_stream, filename="simple.har"):
    return InputDocument(
        path_or_stream=path_or_stream,
        format=InputFormat.HAR,
        backend=HarDocumentBackend,
        filename=filename,
    )


def _backend_from_path(name="simple.har"):
    p = _SOURCES / name
    return HarDocumentBackend(in_doc=_make_input(p), path_or_stream=p)


def _backend_from_stream(name="simple.har"):
    raw = (_SOURCES / name).read_bytes()
    stream = BytesIO(raw)
    return HarDocumentBackend(in_doc=_make_input(stream, name), path_or_stream=stream)


# --- format registration ---


def test_input_format_exists():
    assert InputFormat.HAR == "har"


def test_supported_formats():
    assert InputFormat.HAR in HarDocumentBackend.supported_formats()


def test_no_pagination():
    assert HarDocumentBackend.supports_pagination() is False


# --- backend loading ---


def test_valid_from_path():
    backend = _backend_from_path()
    assert backend.is_valid()


def test_valid_from_stream():
    backend = _backend_from_stream()
    assert backend.is_valid()


def test_invalid_json_raises():
    stream = BytesIO(b"not valid json {{{")
    with pytest.raises(DocumentLoadError):
        HarDocumentBackend(in_doc=_make_input(stream), path_or_stream=stream)


# --- conversion content ---


def test_convert_from_path():
    doc = _backend_from_path().convert()
    md = doc.export_to_markdown()

    assert "GET https://api.example.com/users" in md
    assert "200" in md
    assert "POST https://api.example.com/login" in md
    assert "401" in md


def test_convert_from_stream():
    doc = _backend_from_stream().convert()
    md = doc.export_to_markdown()
    assert "GET https://api.example.com/users" in md


def test_request_body_included():
    doc = _backend_from_path().convert()
    md = doc.export_to_markdown()
    assert "alice" in md


def test_response_body_included():
    doc = _backend_from_path().convert()
    md = doc.export_to_markdown()
    assert "bad credentials" in md


def test_empty_entries_returns_document():
    har = {"log": {"entries": []}}
    stream = BytesIO(json.dumps(har).encode())
    backend = HarDocumentBackend(in_doc=_make_input(stream), path_or_stream=stream)
    doc = backend.convert()
    assert doc is not None


def test_missing_log_key_returns_document():
    stream = BytesIO(b"{}")
    backend = HarDocumentBackend(in_doc=_make_input(stream), path_or_stream=stream)
    doc = backend.convert()
    assert doc is not None


def test_binary_response_body_excluded():
    har = {
        "log": {
            "entries": [
                {
                    "request": {
                        "method": "GET",
                        "url": "https://example.com/image.png",
                    },
                    "response": {
                        "status": 200,
                        "statusText": "OK",
                        "content": {
                            "mimeType": "image/png",
                            "text": "iVBORw0KGgo=",
                        },
                    },
                }
            ]
        }
    }
    stream = BytesIO(json.dumps(har).encode())
    backend = HarDocumentBackend(in_doc=_make_input(stream), path_or_stream=stream)
    doc = backend.convert()
    md = doc.export_to_markdown()
    assert "iVBORw0KGgo=" not in md


# --- document converter integration ---


def test_document_converter_accepts_har():
    pytest.importorskip("pypdfium2")
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(allowed_formats=[InputFormat.HAR])
    result = converter.convert(_SOURCES / "simple.har")
    md = result.document.export_to_markdown()
    assert "GET https://api.example.com/users" in md
