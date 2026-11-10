# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


def _page(text: str, head: str = "") -> str:
    head = f"<head>{head}</head>" if head else ""
    return f"<html>{head}<body><h1>Doc</h1><p>{text}</p></body></html>"


def _convert_to_texts(source: bytes) -> list[str]:
    stream = DocumentStream(name="page.html", stream=BytesIO(source))
    doc = DocumentConverter().convert(stream).document
    return [item.text for item in doc.texts]


def _convert_to_text(source: bytes) -> str:
    """Convert the page and return the text of its paragraph."""
    return _convert_to_texts(source)[-1]


@pytest.mark.parametrize(
    "text",
    [
        "Zażółć gęślą jaźń",
        "é é é",
        "Qualcomm® AI Engine for Qualcomm® Hexagon™ NPU",
        "Größe und Maße",
        "Où est le café ?",
    ],
)
def test_utf8_page_without_declared_encoding(text: str):
    """A UTF-8 page that declares no encoding keeps its text.

    BeautifulSoup asks a statistical detector when a page declares no encoding.
    On short text that detector reads UTF-8 as another encoding, for example
    "é é é" as "矇 矇 矇" and "Hexagon™" as "Hexagonâ\x84ą".
    """
    source = _page(text).encode()

    assert _convert_to_text(source) == text


@pytest.mark.parametrize(
    ("charset", "text"),
    [
        ("windows-1251", "Привет, мир"),
        ("windows-1252", "café au lait"),
        ("shift_jis", "日本語のテキスト"),
        # UTF-8 bytes of "été" that the page declares as windows-1252
        ("windows-1252", "Ã©tÃ©"),
    ],
)
def test_declared_encoding_wins_over_utf8_guess(charset: str, text: str):
    """A declared encoding still decides how the bytes are read."""
    source = _page(text, f'<meta charset="{charset}">').encode(charset)

    assert _convert_to_text(source) == text


def test_utf8_page_with_byte_order_mark():
    """The byte-order mark is dropped, not read as text."""
    text = "Zażółć gęślą jaźń"
    source = b"\xef\xbb\xbf" + _page(text).encode()

    assert _convert_to_texts(source) == ["Doc", text]


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-16", "utf-32"])
def test_utf16_and_utf32_page_with_ascii_text(encoding: str):
    """ASCII text in UTF-16 or UTF-32 is valid UTF-8 with NUL bytes.

    It must stay with the detector and the byte-order mark, and not be read as
    UTF-8.
    """
    source = _page("plain text").encode(encoding)

    assert _convert_to_texts(source) == ["Doc", "plain text"]
