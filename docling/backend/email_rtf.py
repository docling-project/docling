# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Read the HTML body that Outlook keeps encapsulated in a compressed RTF body.

Outlook can store the original HTML of a message only inside ``PidTagRtfCompressed``.
This module decompresses that property ([MS-OXRTFCP]) and extracts the HTML from the
``\\fromhtml1`` RTF ([MS-OXRTFEX]).
"""

from __future__ import annotations

import codecs
import re
import struct

# [MS-OXRTFCP] 2.1.2.1: the dictionary starts with this RTF text.
_LZFU_PREBUF = (
    b"{\\rtf1\\ansi\\mac\\deff0\\deftab720{\\fonttbl;}{\\f0\\fnil \\froman \\fswiss "
    b"\\fmodern \\fscript \\fdecor MS Sans SerifSymbolArialTimes New RomanCourier"
    b"{\\colortbl\\red0\\green0\\blue0\r\n\\par \\pard\\plain\\f0\\fs20\\b\\i\\u\\tab\\tx"
)
_LZFU_COMPRESSED = 0x75465A4C  # "LZFu"
_LZFU_UNCOMPRESSED = 0x414C454D  # "MELA"
_LZFU_DICT_SIZE = 4096

# Windows code page of each RTF \fcharset value.
_FCHARSET_CODEPAGES = {
    77: 10000,
    128: 932,
    129: 949,
    130: 1361,
    134: 936,
    136: 950,
    161: 1253,
    162: 1254,
    163: 1258,
    177: 1255,
    178: 1256,
    186: 1257,
    204: 1251,
    222: 874,
    238: 1250,
}

# Groups that hold no message text, unless marked with \* (which is always skipped).
_SKIPPED_DESTINATIONS = {
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "pict",
    "header",
    "footer",
    "listtable",
    "listoverridetable",
    "revtbl",
    "rsidtbl",
    "generator",
    "xmlnstbl",
}

_TOKEN = re.compile(
    rb"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?"  # control word with optional parameter
    rb"|\\'([0-9a-fA-F]{2})"  # hex-escaped byte
    rb"|\\(.)"  # control symbol
    rb"|([{}])"  # group delimiters
    rb"|([^\\{}\r\n]+)"  # text
    rb"|[\r\n]+",  # line breaks have no meaning in RTF
    re.DOTALL,
)


def decompress_rtf(data: bytes) -> bytes:
    """Decompress a ``PidTagRtfCompressed`` value ([MS-OXRTFCP])."""
    if len(data) < 16:
        raise ValueError("Compressed RTF is shorter than its header")
    comp_size, raw_size, comp_type, _crc = struct.unpack_from("<IIII", data)
    body = data[16 : 4 + comp_size]
    if comp_type == _LZFU_UNCOMPRESSED:
        return body[:raw_size]
    if comp_type != _LZFU_COMPRESSED:
        raise ValueError(f"Unknown compressed RTF type 0x{comp_type:08x}")

    dictionary = bytearray(_LZFU_DICT_SIZE)
    dictionary[: len(_LZFU_PREBUF)] = _LZFU_PREBUF
    write = len(_LZFU_PREBUF)
    out = bytearray()
    pos = 0
    while pos < len(body):
        control = body[pos]
        pos += 1
        for bit in range(8):
            if pos >= len(body):
                break
            if control & (1 << bit):
                if pos + 1 >= len(body):
                    raise ValueError("Compressed RTF ends inside a reference")
                (reference,) = struct.unpack_from(">H", body, pos)
                pos += 2
                offset, length = reference >> 4, (reference & 0xF) + 2
                if offset == write:
                    return bytes(out[:raw_size])
                for i in range(length):
                    byte = dictionary[(offset + i) % _LZFU_DICT_SIZE]
                    out.append(byte)
                    dictionary[write] = byte
                    write = (write + 1) % _LZFU_DICT_SIZE
            else:
                byte = body[pos]
                pos += 1
                out.append(byte)
                dictionary[write] = byte
                write = (write + 1) % _LZFU_DICT_SIZE
    raise ValueError("Compressed RTF ends before its end marker")


def _codec(codepage: int) -> str:
    name = f"cp{codepage}"
    try:
        codecs.lookup(name)
    except LookupError:
        return "cp1252"
    return name


class _State:
    __slots__ = ("codec", "skip", "suppressed", "uc")

    def __init__(self, codec: str) -> None:
        self.codec = codec
        self.suppressed = False
        self.skip = False
        self.uc = 1

    def copy(self) -> _State:
        state = _State(self.codec)
        state.suppressed = self.suppressed
        state.skip = self.skip
        state.uc = self.uc
        return state


_SYMBOL_TEXT = {b"~": "\xa0", b"_": "-", b"-": ""}


class _HtmlExtractor:
    """Walk the RTF tokens and keep the parts that are original HTML."""

    def __init__(self, default_codec: str, font_codecs: dict[int, str]) -> None:
        self.default_codec = default_codec
        self.font_codecs = font_codecs
        self.out: list[str] = []
        self.pending = (
            bytearray()
        )  # \'hh bytes, decoded together for multibyte code pages
        self.stack: list[_State] = []
        self.state = _State(default_codec)
        self.htmltag_depth: int | None = None
        self.group_start = False  # the next token is the first one in a new group
        self.skip_chars = 0  # characters to drop after a \uN

    def flush(self) -> None:
        if self.pending:
            self.out.append(self.pending.decode(self.state.codec, errors="replace"))
            self.pending.clear()

    def emit(self, text: str) -> None:
        if text:
            self.flush()
            self.out.append(text)

    def emitting(self) -> bool:
        return not self.state.skip and (
            self.htmltag_depth is not None or not self.state.suppressed
        )

    def consume_skip(self, count: int = 1) -> int:
        """Drop up to ``count`` characters that follow a \\uN; return how many."""
        dropped = min(self.skip_chars, count)
        self.skip_chars -= dropped
        return dropped

    def open_group(self) -> None:
        self.flush()
        self.stack.append(self.state)
        self.state = self.state.copy()
        self.group_start = True

    def close_group(self) -> None:
        self.flush()
        if self.htmltag_depth == len(self.stack):
            self.htmltag_depth = None
        if self.stack:
            self.state = self.stack.pop()

    def symbol(self, symbol: bytes, first_in_group: bool) -> None:
        if symbol == b"*":
            if first_in_group:
                # Read the destination word: only \*\htmltag holds HTML.
                self.group_start = True
                self.state.skip = True
            return
        if not self.consume_skip() and self.emitting():
            self.emit(_SYMBOL_TEXT.get(symbol, symbol.decode("latin-1")))

    def word(self, name: str, param: bytes | None, first_in_group: bool) -> None:
        if first_in_group and name.startswith("htmltag") and self.state.skip:
            self.state.skip = False
            self.htmltag_depth = len(self.stack)
        elif first_in_group and name in _SKIPPED_DESTINATIONS:
            self.state.skip = True
        elif name == "htmlrtf":
            self.flush()
            self.state.suppressed = param != b"0"
        elif name == "uc":
            self.state.uc = int(param) if param else 1
        elif name == "u":
            code = int(param or 0)
            if self.emitting():
                self.emit(chr(code + 0x10000 if code < 0 else code))
            self.skip_chars = self.state.uc
        elif name == "f" and param is not None:
            self.flush()
            self.state.codec = self.font_codecs.get(int(param), self.default_codec)
        elif self.emitting():
            self.emit({"par": "\r\n", "line": "\r\n", "tab": "\t"}.get(name, ""))

    def hex_byte(self, value: bytes) -> None:
        if not self.consume_skip() and self.emitting():
            self.pending.append(int(value, 16))

    def text(self, data: bytes) -> None:
        data = data[self.consume_skip(len(data)) :]
        if data and self.emitting():
            self.emit(data.decode(self.state.codec, errors="replace"))

    def feed(self, match: re.Match[bytes]) -> None:
        word, param, hex_byte, symbol, brace, text = match.groups()
        first_in_group, self.group_start = self.group_start, False
        if brace == b"{":
            self.open_group()
        elif brace == b"}":
            self.close_group()
        elif symbol is not None:
            self.symbol(symbol, first_in_group)
        elif word is not None:
            self.word(word.decode("ascii"), param, first_in_group)
        elif hex_byte is not None:
            self.hex_byte(hex_byte)
        elif text is not None:
            self.text(text)


def html_from_rtf(rtf: bytes) -> str | None:
    """Return the HTML encapsulated in an RTF body, or ``None`` if it holds none.

    Text inside ``{\\*\\htmltag ...}`` groups and text outside ``\\htmlrtf`` regions is
    original HTML; everything else is RTF added by Outlook and is dropped.
    """
    if b"\\fromhtml1" not in rtf[:1024]:
        return None

    ansi = re.search(rb"\\ansicpg(\d+)", rtf[:1024])
    default_codec = _codec(int(ansi.group(1))) if ansi else "cp1252"
    font_codecs: dict[int, str] = {}
    for match in re.finditer(rb"\\f(\d+)[^;{}]*?\\fcharset(\d+)", rtf):
        codepage = _FCHARSET_CODEPAGES.get(int(match.group(2)))
        if codepage is not None:
            font_codecs[int(match.group(1))] = _codec(codepage)

    extractor = _HtmlExtractor(default_codec, font_codecs)
    for match in _TOKEN.finditer(rtf):
        extractor.feed(match)
    extractor.flush()
    html = "".join(extractor.out).strip()
    return html or None
