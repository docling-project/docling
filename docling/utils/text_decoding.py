# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Decoding of plain-text documents that carry no declared encoding."""

import codecs
from io import BytesIO
from pathlib import Path
from typing import Union

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

# Tried in order when there is no mark. latin-1 maps all 256 byte values and so
# never raises, which is what makes it a usable last resort and why nothing may
# follow it.
_FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8", "cp1252", "latin-1")


def _decode_bytes(raw: bytes) -> str:
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw.decode(encoding)

    *strict, last_resort = _FALLBACK_ENCODINGS
    for encoding in strict:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode(last_resort)


def decode_text(path_or_stream: Union[BytesIO, Path]) -> str:
    """Read a plain-text document and decode it.

    A byte-order mark is honoured. Without one the content is decoded as UTF-8,
    falling back to cp1252 and then latin-1.
    """
    if isinstance(path_or_stream, BytesIO):
        return _decode_bytes(path_or_stream.getvalue())

    # Paths were read through open() in text mode, which translates line
    # endings. Keep that, so only the set of accepted encodings changes.
    text = _decode_bytes(path_or_stream.read_bytes())
    return text.replace("\r\n", "\n").replace("\r", "\n")
