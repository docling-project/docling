import json
from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.datamodel.document import (
    ConversionResult,
    DoclingDocument,
    _DocumentConversionInput,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
pytestmark = pytest.mark.cross_platform


def get_converter() -> DocumentConverter:
    return DocumentConverter(allowed_formats=[InputFormat.BOXNOTE])


def _boxnote_stream(payload: dict, name: str = "test.boxnote") -> DocumentStream:
    raw = json.dumps(payload).encode("utf-8")
    return DocumentStream(name=name, stream=BytesIO(raw))


def test_e2e_boxnote_conversions():
    directory = Path("./tests/data/boxnote/sources/")
    boxnote_paths = sorted(directory.rglob("*.boxnote"))
    converter = get_converter()

    for boxnote_path in boxnote_paths:
        gt_path = boxnote_path.parent.parent / "groundtruth" / boxnote_path.name

        conv_result: ConversionResult = converter.convert(boxnote_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_boxnote_format_detection():
    payload = {"doc": {"type": "doc", "content": []}}
    stream = _boxnote_stream(payload)
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    assert dci._guess_format(stream) == InputFormat.BOXNOTE


def test_legacy_boxnote_reports_unsupported():
    legacy = {
        "atext": {"text": "Hello, World!\n", "attribs": "*0+d|1+1"},
        "pool": {"numToAttrib": {"0": ["author", "1"]}, "nextNum": 1},
    }
    conv_result = get_converter().convert(
        _boxnote_stream(legacy, name="legacy.boxnote"), raises_on_error=False
    )

    assert conv_result.status == ConversionStatus.FAILURE
    assert any("legacy" in err.error_message.lower() for err in conv_result.errors)


def test_empty_boxnote_fails_cleanly():
    conv_result = get_converter().convert(
        DocumentStream(name="empty.boxnote", stream=BytesIO(b"")),
        raises_on_error=False,
    )

    assert conv_result.status == ConversionStatus.FAILURE


@pytest.mark.parametrize("payload", [b"[1, 2, 3]", b"42", b"null", b'"hello"'])
def test_non_object_boxnote_fails_cleanly(payload: bytes):
    conv_result = get_converter().convert(
        DocumentStream(name="scalar.boxnote", stream=BytesIO(payload)),
        raises_on_error=False,
    )

    assert conv_result.status == ConversionStatus.FAILURE


def _linked_paragraph(href) -> dict:
    return {
        "doc": {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "marks": [{"type": "link", "attrs": {"href": href}}],
                            "text": "click",
                        }
                    ],
                }
            ],
        }
    }


def test_safe_hyperlink_is_kept():
    doc = (
        get_converter()
        .convert(_boxnote_stream(_linked_paragraph("https://x.com")))
        .document
    )

    assert str(doc.texts[0].hyperlink) == "https://x.com/"


@pytest.mark.parametrize(
    "href", ["javascript:alert(1)", "hello", "", "http://[::1", "//cdn.example.com/x"]
)
def test_unsafe_hyperlink_is_dropped(href: str):
    doc = get_converter().convert(_boxnote_stream(_linked_paragraph(href))).document

    assert doc.texts[0].hyperlink is None


@pytest.mark.parametrize("href", [42, True, ["http://x.com"], {"href": "http://x.com"}])
def test_non_string_hyperlink_is_dropped(href):
    doc = get_converter().convert(_boxnote_stream(_linked_paragraph(href))).document

    assert doc.texts[0].hyperlink is None
