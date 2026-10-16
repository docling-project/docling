# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for :func:`docling.utils.text_decoding.decode_text`."""

import codecs
from io import BytesIO

import pytest

from docling.exceptions import DocumentLoadError
from docling.utils.text_decoding import decode_text, drop_utf8_bom, is_utf8

BOM = codecs.BOM_UTF8


def test_explicit_utf8_drops_leading_bom_like_autodetect() -> None:
    # A UTF-8 file that starts with a byte-order mark must decode identically
    # whether the encoding is auto-detected or explicitly requested as "utf-8".
    # See https://github.com/docling-project/docling/issues/4846
    content = BOM + b"# Title\nHello"

    auto = decode_text(BytesIO(content), encoding=None)
    explicit = decode_text(BytesIO(content), encoding="utf-8")

    assert not auto.startswith("﻿")
    assert not explicit.startswith("﻿")
    assert auto == explicit


@pytest.mark.parametrize("encoding", ["utf-8", "utf8", "UTF-8", "utf_8", "UTF8", "u8"])
def test_every_utf8_alias_drops_leading_bom(encoding: str) -> None:
    # The codec registry folds "-" to "_" and ignores case, so every alias it
    # accepts names the same UTF-8 codec and must drop the mark the way "utf-8"
    # does. See https://github.com/docling-project/docling/issues/4846
    content = BOM + b"# Title\nHello"

    assert decode_text(BytesIO(content), encoding=encoding) == "# Title\nHello"


def test_plain_utf8_without_bom_is_unchanged() -> None:
    # Files without a BOM must not be affected by the BOM handling.
    content = b"# Title\nHello"

    assert decode_text(BytesIO(content), encoding="utf-8") == decode_text(
        BytesIO(content), encoding=None
    )


def test_explicit_utf8_sig_keeps_behaviour() -> None:
    content = BOM + b"# Title\nHello"
    assert not decode_text(BytesIO(content), encoding="utf-8-sig").startswith("﻿")


def test_other_encodings_still_read_the_mark_as_content() -> None:
    # Only a UTF-8 request drops the mark. Any other codec decodes those bytes
    # the way it always did, so a file that really is cp1252 keeps its bytes.
    content = BOM + b"# Title\nHello"

    assert decode_text(BytesIO(content), encoding="cp1252") == "ï»¿# Title\nHello"


def test_unknown_encoding_is_reported() -> None:
    with pytest.raises(DocumentLoadError):
        decode_text(BytesIO(b"# Title\nHello"), encoding="not-a-codec")


def test_is_utf8_matches_the_codec_registry() -> None:
    for alias in ("utf-8", "utf8", "UTF-8", "utf_8", "UTF8", "u8"):
        assert is_utf8(alias)
    # "utf-8-sig" drops the mark on its own, so it needs no upgrade; everything
    # else is either a different codec or no codec at all.
    for other in ("utf-8-sig", "utf-16", "cp1252", "not-a-codec"):
        assert not is_utf8(other)


def test_drop_utf8_bom_strips_the_mark_only() -> None:
    assert drop_utf8_bom(BOM + b"data") == b"data"
    assert drop_utf8_bom(b"data") == b"data"
    assert drop_utf8_bom(codecs.BOM_UTF16_LE + b"data") == codecs.BOM_UTF16_LE + b"data"
