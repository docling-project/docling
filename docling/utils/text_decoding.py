# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Decoding of plain-text documents that carry no declared encoding."""

import codecs
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling.exceptions import DocumentLoadError

_log = logging.getLogger(__name__)

# A byte-order mark states the encoding, so it settles the question before
# anything is guessed. UTF-32 is tested first because the UTF-32 LE mark starts
# with the UTF-16 LE mark, and the reverse order reads UTF-32 as UTF-16.
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# The one encoding guessed after UTF-8. It covers the Windows-codepage files
# this fallback exists for and nothing follows it, so bytes it cannot map are
# an error rather than another guess.
_FALLBACK_ENCODING: str = "cp1252"

# Above this share of well-formed UTF-8 among the non-ASCII bytes, the input
# reads as UTF-8 with damage rather than as a different encoding. Measured on
# document-length samples: UTF-8 with one stray byte scores 0.99 to 1.00,
# cp1252, latin-1 and KOI8-R score 0.00, Shift-JIS and GBK 0.26 to 0.29.
_UTF8_DAMAGE_THRESHOLD: float = 0.5

_SET_ENCODING_HINT = (
    "Set the `encoding` backend option (for example "
    '`MarkdownBackendOptions(encoding="shift_jis")`) to decode it explicitly.'
)


def _utf8_sequence_length(raw: bytes, start: int) -> int:
    """Length of the well-formed UTF-8 sequence at ``start``, or 0 if there is none."""
    lead = raw[start]
    if 0xC2 <= lead <= 0xDF:
        length = 2
    elif 0xE0 <= lead <= 0xEF:
        length = 3
    elif 0xF0 <= lead <= 0xF4:
        length = 4
    else:
        return 0

    chunk = raw[start : start + length]
    if len(chunk) != length:
        return 0
    try:
        # Python's own validator, so overlong forms and surrogates are rejected
        # without restating the rules here.
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return 0
    return length


def _utf8_high_byte_coverage(raw: bytes) -> float:
    """Share of the non-ASCII bytes that belong to well-formed UTF-8 sequences.

    A file that is UTF-8 apart from a few damaged bytes scores near 1, because
    everything else still validates. A file in a single-byte legacy encoding
    scores near 0, because its high bytes form no sequences at all. That
    separates damage, which must be reported, from an encoding we can guess.
    """
    high = 0
    covered = 0
    index = 0
    while index < len(raw):
        if raw[index] < 0x80:
            index += 1
            continue
        length = _utf8_sequence_length(raw, index)
        if length:
            high += length
            covered += length
            index += length
        else:
            high += 1
            index += 1
    return covered / high if high else 0.0


def _decode_bytes(raw: bytes, encoding: Optional[str] = None) -> str:
    if encoding is not None:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError) as error:
            raise DocumentLoadError(
                f"Could not decode the document with the requested encoding "
                f"{encoding!r}: {error}"
            ) from error

    for bom, bom_encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw.decode(bom_encoding)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as utf8_error:
        coverage = _utf8_high_byte_coverage(raw)
        if coverage >= _UTF8_DAMAGE_THRESHOLD:
            raise DocumentLoadError(
                f"The document is not valid UTF-8, but {coverage:.0%} of its "
                f"non-ASCII bytes still form well-formed UTF-8 sequences, so it "
                f"reads as damaged UTF-8 rather than as another encoding. "
                f"Decoding it as {_FALLBACK_ENCODING} would turn the damage into "
                f"text that looks fine and is not. {_SET_ENCODING_HINT}"
            ) from utf8_error

        try:
            text = raw.decode(_FALLBACK_ENCODING)
        except UnicodeDecodeError as fallback_error:
            raise DocumentLoadError(
                f"The document is neither UTF-8 nor {_FALLBACK_ENCODING}, and its "
                f"encoding is not declared, so it cannot be decoded reliably. "
                f"{_SET_ENCODING_HINT}"
            ) from fallback_error

        _log.warning(
            "Document is not UTF-8; decoded it as %s. Text in any other "
            "single-byte or multi-byte encoding will be wrong. %s",
            _FALLBACK_ENCODING,
            _SET_ENCODING_HINT,
        )
        return text


def decode_text(
    path_or_stream: Union[BytesIO, Path], encoding: Optional[str] = None
) -> str:
    """Read a plain-text document and decode it.

    With ``encoding`` set, the content is decoded with it and nothing is
    guessed. Without it, a byte-order mark is honoured, then UTF-8 is tried,
    then cp1252; input that is none of those raises ``DocumentLoadError``
    rather than being re-decoded into mojibake.

    Line endings are translated as text-mode ``open()`` does, so a document
    converts the same way whether it arrives as a path or as a stream.
    """
    raw = (
        path_or_stream.getvalue()
        if isinstance(path_or_stream, BytesIO)
        else path_or_stream.read_bytes()
    )
    text = _decode_bytes(raw, encoding)
    return text.replace("\r\n", "\n").replace("\r", "\n")
