# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Encoding handling shared by the Markdown, CSV and AsciiDoc backends.

Each of them reads a whole user-supplied file as text, and each decodes the
path and the stream separately, so every case here is checked on both routes.
"""

from io import BytesIO

import pytest

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling.utils.text_decoding import decode_text

ACCENTED = "résumé naïve café"

FORMATS = [
    pytest.param(InputFormat.MD, "md", f"# Rapport\n\n{ACCENTED}\n", id="md"),
    pytest.param(InputFormat.CSV, "csv", f"note\n{ACCENTED}\n", id="csv"),
    pytest.param(
        InputFormat.ASCIIDOC, "adoc", f"= Rapport\n\n{ACCENTED}\n", id="asciidoc"
    ),
]


def _export_both_routes(fmt, suffix, raw, tmp_path) -> tuple[str, str]:
    """Convert the same bytes as a stream and as a file, returning both exports."""
    converter = DocumentConverter(allowed_formats=[fmt])

    stream_doc = converter.convert(
        DocumentStream(name=f"doc.{suffix}", stream=BytesIO(raw)),
        raises_on_error=True,
    ).document

    path = tmp_path / f"doc.{suffix}"
    path.write_bytes(raw)
    file_doc = converter.convert(path, raises_on_error=True).document

    return stream_doc.export_to_markdown(), file_doc.export_to_markdown()


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
@pytest.mark.parametrize("encoding", ["cp1252", "latin-1", "utf-16"])
def test_text_that_is_not_utf8_still_converts(fmt, suffix, text, encoding, tmp_path):
    """A file in a non-UTF-8 encoding converts instead of failing to load.

    Decoding was strict UTF-8, so a single accented byte from any of these
    encodings aborted the document.
    """
    for export in _export_both_routes(fmt, suffix, text.encode(encoding), tmp_path):
        assert ACCENTED in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_utf16_is_decoded_from_its_mark_not_guessed(fmt, suffix, text, tmp_path):
    """UTF-16 is decoded as UTF-16, not as a single-byte codec.

    latin-1 maps all 256 byte values, so it cannot fail and would turn the mark
    and the NUL padding into text rather than raising.
    """
    for export in _export_both_routes(fmt, suffix, text.encode("utf-16"), tmp_path):
        assert ACCENTED in export
        assert "ÿþ" not in export
        assert "\x00" not in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_utf32_is_not_decoded_as_utf16(fmt, suffix, text, tmp_path):
    """The UTF-32 LE mark opens with the UTF-16 LE mark, so order matters.

    Matching UTF-16 first consumes two of the four bytes and reads the rest as
    UTF-16, which yields NUL-separated text instead of the document.
    """
    for export in _export_both_routes(fmt, suffix, text.encode("utf-32"), tmp_path):
        assert ACCENTED in export
        assert "\x00" not in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_utf8_input_is_unaffected(fmt, suffix, text, encoding, tmp_path):
    """UTF-8, with or without a mark, decodes exactly as it did before."""
    for export in _export_both_routes(fmt, suffix, text.encode(encoding), tmp_path):
        assert ACCENTED in export
        assert "﻿" not in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_bytes_undefined_in_cp1252_fall_through_to_latin1(fmt, suffix, text, tmp_path):
    """cp1252 leaves five byte values undefined, so it can still raise.

    latin-1 maps them, and reaching it is the whole reason it sits at the end
    of the chain rather than cp1252.
    """
    raw = text.encode("latin-1").replace(b"caf", b"caf\x81")

    for export in _export_both_routes(fmt, suffix, raw, tmp_path):
        assert "caf\x81é" in export


@pytest.mark.parametrize("raw_endings", ["\r\n", "\r", "\n"])
def test_file_route_still_matches_text_mode_open(raw_endings, tmp_path):
    """Reading a path went through open() in text mode, which translates line
    endings; reading bytes does not, so the file route has to keep doing it."""
    path = tmp_path / "doc.md"
    path.write_bytes(f"# T{raw_endings}{raw_endings}{ACCENTED}{raw_endings}".encode())

    with open(path, encoding="utf-8-sig") as handle:
        assert decode_text(path) == handle.read()
