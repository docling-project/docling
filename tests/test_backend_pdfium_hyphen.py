# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The pypdfium2 backend restores the hyphen PDFium reports as a control code.

PDFium has no character for a hyphen that ends a line: FPDFText_GetUnicode
returns STX (U+0002) and FPDFText_GetText returns U+FFFE. The glyph on the page
is the hyphen, so the text of a cell must show it. Without the restore the
control code reaches the export, where XML 1.0 forbids it.

The fixture is built as raw PDF bytes, so the test needs no models and no extra
dependency.
"""

from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

FONT_SIZE = 24
# The first line ends in a hyphen: PDFium treats it as a line-final hyphen and
# reports the control code instead of the glyph.
LINES = (
    (b"Completion is 30 June; the long-", 700),
    (b"stop date is 30 September.", 676),
)
EXPECTED_LINES = ["Completion is 30 June; the long-", "stop date is 30 September."]


def _build_pdf() -> bytes:
    """A one-page PDF with the two lines in Helvetica."""
    content = b"".join(
        b"BT /F1 %d Tf 72 %d Td (%s) Tj ET\n" % (FONT_SIZE, baseline, text)
        for text, baseline in LINES
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
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


def test_text_cells_restore_the_hyphen_of_a_line_final_glyph(tmp_path: Path) -> None:
    pdf_path = tmp_path / "line_final_hyphen.pdf"
    pdf_path.write_bytes(_build_pdf())
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    page_backend = in_doc._backend.load_page(0)

    cells = list(page_backend.get_text_cells())

    # The line that ends in the hyphen must carry the hyphen, not U+0002.
    assert [cell.text.strip() for cell in cells] == EXPECTED_LINES
    assert cells[0].orig.strip() == EXPECTED_LINES[0]
    # The table code path reads the text of a rect straight from PDFium.
    text_in_rect = page_backend.get_text_in_rect(cells[0].rect.to_bounding_box())
    assert text_in_rect == EXPECTED_LINES[0]
