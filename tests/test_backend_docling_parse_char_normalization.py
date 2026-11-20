# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""docling-parse folds dashes and minus signs unless the backend option disables it.

The fixture sets an en dash (U+2013), an em dash (U+2014), a minus sign
(U+2212), a horizontal bar (U+2015), a figure dash (U+2012) and a hyphen
(U+2010) in Helvetica, mapped by a ToUnicode CMap. The hyphen folds in both
modes. The fixture is built as raw PDF bytes, so the test needs no models and
no extra dependency.
"""

from pathlib import Path

import pytest

from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import NativePdfPipelineOptions
from docling.document_converter import DocumentConverter, NativePdfFormatOption

CONTENT = (
    b"BT /F1 14 Tf 72 760 Td (pages 3\x015) Tj ET\n"
    b"BT /F1 14 Tf 72 720 Td (Product \x02 Nutshell) Tj ET\n"
    b"BT /F1 14 Tf 72 680 Td (Balance \x0312) Tj ET\n"
    b"BT /F1 14 Tf 72 640 Td (Rating \x04\x04) Tj ET\n"
    b"BT /F1 14 Tf 72 600 Td (Figure 1\x052) Tj ET\n"
    b"BT /F1 14 Tf 72 560 Td (co\x06operate) Tj ET\n"
)
TO_UNICODE = (
    b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
    b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
    b"/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
    b"1 begincodespacerange <00> <FF> endcodespacerange\n"
    b"6 beginbfchar <01> <2013> <02> <2014> <03> <2212> <04> <2015> "
    b"<05> <2012> <06> <2010> endbfchar\n"
    b"1 beginbfrange <20> <7E> <0020> endbfrange\n"
    b"endcmap CMapName currentdict /CMap defineresource pop end end"
)


def _build_pdf() -> bytes:
    """A one-page PDF with six lines in Helvetica and a ToUnicode CMap."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(CONTENT) + CONTENT + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding "
        b"/Differences [1 /endash /emdash /minus /horizontalbar /figuredash "
        b"/hyphen] >> /ToUnicode 6 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(TO_UNICODE) + TO_UNICODE + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    startxref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        startxref,
    )
    return bytes(out)


def _convert_texts(
    tmp_path: Path, backend_options: PdfBackendOptions | None
) -> list[str]:
    pdf_path = tmp_path / "dashes.pdf"
    pdf_path.write_bytes(_build_pdf())
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: NativePdfFormatOption(
                pipeline_options=NativePdfPipelineOptions(generate_page_images=False),
                backend_options=backend_options,
            )
        }
    )
    result = converter.convert(pdf_path)
    assert result.status == ConversionStatus.SUCCESS
    return sorted(item.text for item in result.document.texts)


@pytest.mark.parametrize(
    ("backend_options", "expected"),
    [
        pytest.param(
            None,
            [
                "Balance -12",
                "Figure 1-2",
                "Product - Nutshell",
                "Rating --",
                "co-operate",
                "pages 3-5",
            ],
            id="default-folds",
        ),
        pytest.param(
            PdfBackendOptions(do_char_normalization=False),
            [
                "Balance \u221212",
                "Figure 1\u20122",
                "Product \u2014 Nutshell",
                "Rating \u2015\u2015",
                "co-operate",
                "pages 3\u20135",
            ],
            id="disabled-keeps",
        ),
    ],
)
def test_char_normalization_option_controls_dash_folding(
    tmp_path: Path,
    backend_options: PdfBackendOptions | None,
    expected: list[str],
) -> None:
    assert _convert_texts(tmp_path, backend_options) == expected
