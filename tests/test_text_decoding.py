# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for :func:`docling.utils.text_decoding.decode_text`."""

import codecs
from io import BytesIO

import pytest

from docling.exceptions import DocumentLoadError
from docling.utils.text_decoding import decode_text


def test_explicit_utf8_drops_leading_bom_like_autodetect() -> None:
    # A UTF-8 file that starts with a byte-order mark must decode identically
    # whether the encoding is auto-detected or explicitly requested as "utf-8".
    # See https://github.com/docling-project/docling/issues/4846
    content = codecs.BOM_UTF8 + b"# Title\nHello"

    auto = decode_text(BytesIO(content), encoding=None)
    explicit = decode_text(BytesIO(content), encoding="utf-8")

    assert not auto.startswith("\ufeff")
    assert not explicit.startswith("\ufeff")
    assert auto == explicit


@pytest.mark.parametrize(
    ("encoding", "bom"),
    [
        ("utf-8", codecs.BOM_UTF8),
        ("utf8", codecs.BOM_UTF8),
        ("UTF-8", codecs.BOM_UTF8),
        ("utf_8", codecs.BOM_UTF8),
        ("UTF8", codecs.BOM_UTF8),
        ("u8", codecs.BOM_UTF8),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("utf-32-le", codecs.BOM_UTF32_LE),
        ("utf-32-be", codecs.BOM_UTF32_BE),
    ],
)
def test_every_unicode_encoding_drops_leading_bom(encoding: str, bom: bytes) -> None:
    # A leading byte-order mark is never content, so every Unicode codec —
    # under any alias Python accepts, and with either endianness spelled out
    # explicitly — must decode it away, exactly as the auto-detect path does.
    # See https://github.com/docling-project/docling/issues/4846
    content = bom + "# Title\nHello".encode(encoding)

    assert decode_text(BytesIO(content), encoding=encoding) == "# Title\nHello"


def test_plain_utf8_without_bom_is_unchanged() -> None:
    # Files without a BOM must not be affected by the BOM handling.
    content = b"# Title\nHello"

    assert decode_text(BytesIO(content), encoding="utf-8") == decode_text(
        BytesIO(content), encoding=None
    )


def test_explicit_utf8_sig_keeps_behaviour() -> None:
    content = codecs.BOM_UTF8 + b"# Title\nHello"
    assert not decode_text(BytesIO(content), encoding="utf-8-sig").startswith("\ufeff")


def test_other_encodings_still_read_the_mark_as_content() -> None:
    # Only a Unicode codec decodes the mark to U+FEFF. A single-byte codec
    # such as cp1252 decodes those bytes as the characters they always were,
    # so a file that really is cp1252 keeps its bytes.
    content = codecs.BOM_UTF8 + b"# Title\nHello"

    assert decode_text(BytesIO(content), encoding="cp1252") == "ï»¿# Title\nHello"


def test_unknown_encoding_is_reported() -> None:
    with pytest.raises(DocumentLoadError):
        decode_text(BytesIO(b"# Title\nHello"), encoding="not-a-codec")
