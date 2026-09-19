# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Encoding handling shared by the Markdown, CSV and AsciiDoc backends.

Each of them reads a whole user-supplied file as text, and each decodes the
path and the stream separately, so every case here is checked on both routes.
"""

import logging
from io import BytesIO

import pytest

from docling.datamodel.backend_options import (
    AsciiDocBackendOptions,
    CsvBackendOptions,
    MarkdownBackendOptions,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import (
    AsciiDocFormatOption,
    CsvFormatOption,
    DocumentConverter,
    MarkdownFormatOption,
)
from docling.exceptions import ConversionError, DocumentLoadError
from docling.utils.text_decoding import decode_text

ACCENTED = "résumé naïve café"
CYRILLIC = "отчёт проверка"

FORMATS = [
    pytest.param(InputFormat.MD, "md", f"# Rapport\n\n{ACCENTED}\n", id="md"),
    pytest.param(InputFormat.CSV, "csv", f"note\n{ACCENTED}\n", id="csv"),
    pytest.param(
        InputFormat.ASCIIDOC, "adoc", f"= Rapport\n\n{ACCENTED}\n", id="asciidoc"
    ),
]

OPTIONS_BY_FORMAT = {
    InputFormat.MD: (MarkdownFormatOption, MarkdownBackendOptions),
    InputFormat.CSV: (CsvFormatOption, CsvBackendOptions),
    InputFormat.ASCIIDOC: (AsciiDocFormatOption, AsciiDocBackendOptions),
}


def _converter(fmt, encoding=None) -> DocumentConverter:
    if encoding is None:
        return DocumentConverter(allowed_formats=[fmt])

    format_option_cls, options_cls = OPTIONS_BY_FORMAT[fmt]
    return DocumentConverter(
        allowed_formats=[fmt],
        format_options={
            fmt: format_option_cls(backend_options=options_cls(encoding=encoding))
        },
    )


def _export_both_routes(fmt, suffix, raw, tmp_path, encoding=None) -> tuple[str, str]:
    """Convert the same bytes as a stream and as a file, returning both exports."""
    converter = _converter(fmt, encoding)

    stream_doc = converter.convert(
        DocumentStream(name=f"doc.{suffix}", stream=BytesIO(raw)),
        raises_on_error=True,
    ).document

    path = tmp_path / f"doc.{suffix}"
    path.write_bytes(raw)
    file_doc = converter.convert(path, raises_on_error=True).document

    return stream_doc.export_to_markdown(), file_doc.export_to_markdown()


def _both_routes_raise(fmt, suffix, raw, tmp_path) -> list[BaseException]:
    """Convert the same bytes both ways, asserting each fails, and return the causes.

    The converter reports a bad input as ConversionError and keeps the backend's
    own exception on __cause__, so that is where the message has to be readable.
    """
    converter = _converter(fmt)

    causes = []
    with pytest.raises(ConversionError) as stream_error:
        converter.convert(
            DocumentStream(name=f"doc.{suffix}", stream=BytesIO(raw)),
            raises_on_error=True,
        )
    causes.append(stream_error.value.__cause__)

    path = tmp_path / f"doc.{suffix}"
    path.write_bytes(raw)
    with pytest.raises(ConversionError) as file_error:
        converter.convert(path, raises_on_error=True)
    causes.append(file_error.value.__cause__)

    for cause in causes:
        assert isinstance(cause, DocumentLoadError)
    return causes


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
@pytest.mark.parametrize("encoding", ["cp1252", "latin-1", "utf-16"])
def test_text_that_is_not_utf8_still_converts(fmt, suffix, text, encoding, tmp_path):
    """A file in a non-UTF-8 encoding converts instead of failing to load.

    Decoding was strict UTF-8, so a single accented byte from any of these
    encodings aborted the document. These three are covered without guessing:
    UTF-16 states itself with a mark, and the accented range of latin-1 is
    shared with cp1252 byte for byte.
    """
    for export in _export_both_routes(fmt, suffix, text.encode(encoding), tmp_path):
        assert ACCENTED in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_utf16_is_decoded_from_its_mark_not_guessed(fmt, suffix, text, tmp_path):
    """UTF-16 is decoded as UTF-16, not as a single-byte codec.

    Its NUL padding is inside cp1252's repertoire, so without the mark being
    read first the fallback would turn the whole file into text rather than
    raising.
    """
    for export in _export_both_routes(fmt, suffix, text.encode("utf-16"), tmp_path):
        assert ACCENTED in export
        assert "ÿþ" not in export
        assert "\x00" not in export


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("content", ["plain ASCII", ACCENTED, "ABCDĀ"])
def test_bomless_utf16_requires_encoding(
    fmt, suffix, text, encoding, content, tmp_path
):
    raw = text.replace(ACCENTED, content).encode(encoding)

    for error in _both_routes_raise(fmt, suffix, raw, tmp_path):
        assert "UTF-16" in str(error)
        assert "`encoding`" in str(error)
        assert "utf-16-le" in str(error)
        assert "utf-16-be" in str(error)

    for export in _export_both_routes(fmt, suffix, raw, tmp_path, encoding=encoding):
        assert content in export
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
def test_damaged_utf8_raises_instead_of_being_re_decoded(fmt, suffix, text, tmp_path):
    """UTF-8 with one stray byte is reported, not re-decoded into mojibake.

    cp1252 maps that byte, so without the ratio gate the file would convert and
    every accented character in it would silently become two wrong ones.
    """
    raw = text.encode("utf-8").replace(b"caf", b"caf\xff")

    for error in _both_routes_raise(fmt, suffix, raw, tmp_path):
        assert "damaged UTF-8" in str(error)
        assert "`encoding`" in str(error)


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_bytes_undefined_in_cp1252_raise(fmt, suffix, text, tmp_path):
    """cp1252 leaves five byte values undefined, and nothing follows it.

    Reaching the end of the chain is an error naming the option that fixes it,
    rather than a further guess that cannot fail.
    """
    raw = text.encode("latin-1").replace(b"caf", b"caf\x81")

    for error in _both_routes_raise(fmt, suffix, raw, tmp_path):
        assert "neither UTF-8 nor cp1252" in str(error)
        assert "`encoding`" in str(error)


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_cp1252_fallback_warns(fmt, suffix, text, tmp_path, caplog):
    """Guessing is announced, because the guess can be wrong without failing."""
    with caplog.at_level(logging.WARNING, logger="docling.utils.text_decoding"):
        _export_both_routes(fmt, suffix, text.encode("cp1252"), tmp_path)

    assert any("decoded it as cp1252" in record.message for record in caplog.records)


@pytest.mark.parametrize("fmt, suffix, text", FORMATS)
def test_requested_encoding_is_used_instead_of_guessing(fmt, suffix, text, tmp_path):
    """An encoding the chain cannot reach is read correctly when it is declared.

    KOI8-R is inside cp1252's repertoire, so the guess would produce mojibake
    with no error; this is the only way to get the text itself.
    """
    cyrillic_text = text.replace(ACCENTED, CYRILLIC)
    raw = cyrillic_text.encode("koi8-r")

    for export in _export_both_routes(fmt, suffix, raw, tmp_path, encoding="koi8-r"):
        assert CYRILLIC in export


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
@pytest.mark.parametrize("text", ["", "Report", ACCENTED])
def test_text_without_nul_is_unaffected(encoding, text, tmp_path):
    raw = text.encode(encoding)
    path = tmp_path / "doc.md"
    path.write_bytes(raw)

    assert decode_text(BytesIO(raw)) == text
    assert decode_text(path) == text


@pytest.mark.parametrize("raw", [b"\x00", b"Report\x00 caf\xc3\xa9", b"caf\xe9\x00"])
def test_undeclared_nul_bytes_require_encoding(raw, tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(raw)

    for source in (BytesIO(raw), path):
        with pytest.raises(DocumentLoadError, match=r"NUL bytes.*`encoding`"):
            decode_text(source)


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "utf-8-sig"])
def test_declared_encoding_takes_precedence_over_nul_bytes(encoding, tmp_path):
    text = "R\x00e\x00p\x00o\x00r\x00t\x00"
    raw = text.encode(encoding)
    path = tmp_path / "doc.md"
    path.write_bytes(raw)

    assert decode_text(BytesIO(raw), encoding=encoding) == text
    assert decode_text(path, encoding=encoding) == text


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_bom_takes_precedence_over_nul_bytes(encoding, tmp_path):
    text = "R\x00e\x00p\x00o\x00r\x00t\x00"
    raw = text.encode(encoding)
    path = tmp_path / "doc.md"
    path.write_bytes(raw)

    assert decode_text(BytesIO(raw)) == text
    assert decode_text(path) == text


def test_requested_encoding_that_cannot_decode_says_so(tmp_path):
    """A wrong declared encoding is an error naming it, not a silent fallback."""
    path = tmp_path / "doc.md"
    path.write_bytes("# Rapport\n".encode("utf-16"))

    with pytest.raises(DocumentLoadError) as error:
        decode_text(path, "utf-8")

    assert "'utf-8'" in str(error.value)


@pytest.mark.parametrize("raw_endings", ["\r\n", "\r", "\n"])
def test_file_route_still_matches_text_mode_open(raw_endings, tmp_path):
    """Reading a path went through open() in text mode, which translates line
    endings; reading bytes does not, so the file route has to keep doing it."""
    path = tmp_path / "doc.md"
    path.write_bytes(f"# T{raw_endings}{raw_endings}{ACCENTED}{raw_endings}".encode())

    with open(path, encoding="utf-8-sig") as handle:
        assert decode_text(path) == handle.read()
