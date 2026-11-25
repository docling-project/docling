# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A punctuation run of the pypdfium2 backend joins its word without a space.

PDFium gives a full stop set as its own text run a tight glyph box on the
baseline, well below the top of the line. The fixture sets an italic phrase
followed by a roman stop, as legal text does with Latin phrases.

The fixture is built as raw PDF bytes, so the test needs no models and no extra
dependency.
"""

from pathlib import Path

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

CONTENT = (
    b"BT /F2 11 Tf 72 700 Td (The clause applies mutatis mutandis) Tj "
    b"/F1 11 Tf (.) Tj ( The price is payable on completion.) Tj ET"
)


# A table whose row-group label is set rotated, spanning two body rows: the
# rows' cells lie inside the label's vertical extent.
ROTATED_LABEL = (
    b"BT /F1 12 Tf 0 1 -1 0 80 566 Tm (EUROPE) Tj ET "
    b"BT /F1 10 Tf 90 605 Td (France) Tj ET BT /F1 10 Tf 200 605 Td (12.5) Tj ET "
    b"BT /F1 10 Tf 90 589 Td (Spain) Tj ET BT /F1 10 Tf 200 589 Td (9.8) Tj ET"
)


def _build_pdf(content: bytes = CONTENT) -> bytes:
    """A one-page PDF with the given content in Times-Roman (F1) and Times-Italic (F2)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Italic "
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


def test_punctuation_run_joins_its_word(tmp_path: Path) -> None:
    pdf_path = tmp_path / "punct.pdf"
    pdf_path.write_bytes(_build_pdf())
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    page_backend = in_doc._backend.load_page(0)

    cells = list(page_backend.get_text_cells())

    assert [cell.text.strip() for cell in cells] == [
        "The clause applies mutatis mutandis. The price is payable on completion."
    ]


def _cell_texts(pdf_path: Path) -> list[tuple[str, float]]:
    """Text and top of each cell on the first page."""
    in_doc = InputDocument(
        path_or_stream=pdf_path,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    page_backend = in_doc._backend.load_page(0)
    return [
        (cell.text.strip(), cell.rect.to_bounding_box().t)
        for cell in page_backend.get_text_cells()
    ]


def test_rows_beside_a_rotated_label_stay_separate(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rotated_label.pdf"
    pdf_path.write_bytes(_build_pdf(ROTATED_LABEL))

    assert [text for text, _ in _cell_texts(pdf_path)] == [
        "EUROPE",
        "France",
        "12.5",
        "Spain",
        "9.8",
    ]


def test_right_to_left_answers_across_a_gap_stay_separate() -> None:
    """The Yes/No answers beside the first question keep one cell each.

    The words are set right to left, so the second run lies left of the
    first, with a check box between them.
    """
    pdf_path = Path("tests/data/pdf/sources/right_to_left_03.pdf")

    first_answers = [text for text, top in _cell_texts(pdf_path) if 230 < top < 260]

    assert "بلی" in first_answers
    assert "خير" in first_answers
