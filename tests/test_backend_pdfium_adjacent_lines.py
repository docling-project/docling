# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Text cells of the pypdfium2 backend keep the letters of adjacent lines out.

PDFium selects every character whose box touches a query rect. With solid
leading (line spacing equal to the font size), the descenders of one line reach
into the box of the next line, and the accents on its capitals reach back up.
The fixture sets two Polish lines in this way: "p" and "y" hang below the first
line, and the acute accents of "Ó" rise above the second.

The fixture is built as raw PDF bytes, so the test needs no models and no extra
dependency.
"""

from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

FONT_SIZE = 24
# Text (WinAnsi, \323 is "Ó") and baseline of each line, FONT_SIZE apart.
LINES = ((b"Odpady wystawiamy", 700), (b"ODBI\\323R ODPAD\\323W", 676))


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


def test_text_cells_do_not_take_letters_of_adjacent_lines(tmp_path: Path) -> None:
    pdf_path = tmp_path / "solid_leading.pdf"
    pdf_path.write_bytes(_build_pdf())
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    page_backend = in_doc._backend.load_page(0)

    cells = list(page_backend.get_text_cells())

    # The fixture must overlap the two lines, as tightly set text does.
    first_line, second_line = (cell.rect.to_bounding_box() for cell in cells)
    assert first_line.b > second_line.t
    assert [cell.text.strip() for cell in cells] == [
        "Odpady wystawiamy",
        "ODBIÓR ODPADÓW",
    ]
