# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import struct

import pytest

from docling.backend.email_rtf import decompress_rtf, html_from_rtf


def test_decompress_rtf_spec_example():
    # [MS-OXRTFCP] 3.1.1, example 1.
    compressed = bytes.fromhex(
        "2d0000002b0000004c5a4675f1c5c7a703000a00726370673132354232"
        "0af32068656c090020627705b06c647d0a800fa0"
    )

    assert decompress_rtf(compressed) == (
        b"{\\rtf1\\ansi\\ansicpg1252\\pard hello world}\r\n"
    )


def test_decompress_rtf_uncompressed_body():
    rtf = b"{\\rtf1 plain}"
    data = struct.pack("<IIII", len(rtf) + 12, len(rtf), 0x414C454D, 0) + rtf

    assert decompress_rtf(data) == rtf


def test_decompress_rtf_rejects_unknown_type():
    with pytest.raises(ValueError):
        decompress_rtf(struct.pack("<IIII", 12, 0, 0x12345678, 0))


@pytest.mark.parametrize(
    ("end", "message"),
    [
        (10, "shorter than its header"),
        (21, "ends before its end marker"),
        (-1, "ends inside a reference"),
    ],
    ids=["header", "literals", "reference"],
)
def test_decompress_rtf_rejects_truncated_data(end, message):
    compressed = bytes.fromhex(
        "2d0000002b0000004c5a4675f1c5c7a703000a00726370673132354232"
        "0af32068656c090020627705b06c647d0a800fa0"
    )

    with pytest.raises(ValueError, match=message):
        decompress_rtf(compressed[:end])


def test_html_from_rtf_keeps_tags_and_text_and_drops_rtf_only_parts():
    rtf = (
        b"{\\rtf1\\ansi\\ansicpg1252\\fromhtml1 \\deff0{\\fonttbl{\\f0\\fswiss Arial;}}\n"
        b"{\\*\\htmltag19 <html>}{\\*\\htmltag50 <body>}\n"
        b"{\\*\\htmltag64 <p>}\\htmlrtf {\\htmlrtf0 Caf\\'e9 & co\n"
        b"{\\*\\htmltag72 </p>}\\htmlrtf }\\htmlrtf0 \n"
        b"\\htmlrtf\\par\\htmlrtf0\n"
        b"{\\*\\htmltag64 <pre>}line one\\par line\\tab two{\\*\\htmltag72 </pre>}\n"
        b"{\\*\\htmltag58 </body>}{\\*\\htmltag27 </html>}}"
    )

    assert html_from_rtf(rtf) == (
        "<html><body><p>Café & co</p><pre>line one\r\nline\ttwo</pre></body></html>"
    )


def test_html_from_rtf_decodes_unicode_and_font_code_pages():
    rtf = (
        b"{\\rtf1\\ansi\\ansicpg1251\\fromhtml1 \\deff0"
        b"{\\fonttbl{\\f0\\fswiss\\fcharset204 Arial;}{\\f1\\fnil\\fcharset128 MS Gothic;}}\n"
        b"{\\*\\htmltag64 <p>}\\'cf\\'f0\\'e8\\'e2\\'e5\\'f2 \\uc1\\u8364?{\\f1 \\'82\\'a0}"
        b"{\\*\\htmltag72 </p>}}"
    )

    assert html_from_rtf(rtf) == "<p>Привет €あ</p>"


def test_html_from_rtf_returns_none_for_plain_rtf():
    assert html_from_rtf(b"{\\rtf1\\ansi\\ansicpg1252 hello\\par}") is None


def test_html_from_rtf_decodes_control_symbols():
    rtf = (
        b"{\\rtf1\\ansi\\ansicpg1252\\fromhtml1 "
        b"{\\*\\htmltag64 <p>}a\\{b\\}c\\\\d\\~e\\-f\\_g{\\*\\htmltag72 </p>}}"
    )

    assert html_from_rtf(rtf) == "<p>a{b}c\\d\xa0ef-g</p>"


def test_html_from_rtf_falls_back_to_cp1252_for_an_unknown_code_page():
    rtf = b"{\\rtf1\\ansi\\ansicpg99999\\fromhtml1 {\\*\\htmltag64 <p>}caf\\'e9{\\*\\htmltag72 </p>}}"

    assert html_from_rtf(rtf) == "<p>café</p>"


def test_html_from_rtf_drops_unicode_in_rtf_only_text_and_ignores_extra_braces():
    rtf = (
        b"{\\rtf1\\ansi\\fromhtml1 {\\*\\htmltag64 <p>}"
        b"\\htmlrtf \\u8364?\\~\\htmlrtf0 ok{\\b \\*!}{\\*\\htmltag72 </p>}}}}"
    )

    assert html_from_rtf(rtf) == "<p>ok!</p>"
