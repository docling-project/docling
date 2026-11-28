# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Punctuation runs of the pypdfium2 backend join their row without a space.

PDFium gives a full stop set as its own text run a tight glyph box on the
baseline, well below the top of the line. The fixture sets an italic phrase
followed by a roman stop, as legal text does with Latin phrases.

A comma run additionally hangs below the baseline: on a row without
descenders its top edge sits inside the row while its bottom edge dips below
the row's, so the containment rule alone would split amounts like
"$250,500,000" across cells.

The tests build small PDFs or use existing fixtures. They need no models or
extra dependencies.
"""

from pathlib import Path

import pytest

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

CONTENT = (
    b"BT /F2 11 Tf 72 700 Td (The clause applies mutatis mutandis) Tj "
    b"/F1 11 Tf (.) Tj ( The price is payable on completion.) Tj ET"
)

CONTENT_COMMA = (
    b"BT /F1 11 Tf 72 700 Td (the sum of $250) Tj (,) Tj (500,000 and 32.5%) Tj ET\n"
    b"BT /F1 11 Tf 72 680 Td (ACME HOLDINGS PLC) Tj (,) Tj ( as Parent) Tj ET"
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


def test_descending_punctuation_run_joins_its_row(tmp_path: Path) -> None:
    """A comma hanging below a descenderless row keeps the line in one cell."""
    pdf_path = tmp_path / "descend.pdf"
    pdf_path.write_bytes(_build_pdf(CONTENT_COMMA))

    assert [text for text, _ in _cell_texts(pdf_path)] == [
        "the sum of $250,500,000 and 32.5%",
        "ACME HOLDINGS PLC, as Parent",
    ]


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        pytest.param(
            b"(,) Tj (NEXT) Tj ET",
            ["ACME,NEXT"],
            id="same-row",
        ),
        pytest.param(
            b"ET BT /F1 11 Tf 200 700 Td (,) Tj (NEXT) Tj ET",
            ["ACME", ",", "NEXT"],
            id="gap-before-comma",
        ),
        pytest.param(
            b"(,) Tj ET BT /F1 11 Tf 200 700 Td (NEXT) Tj ( ROW) Tj ET",
            ["ACME,", "NEXT ROW"],
            id="gap-after-comma",
        ),
        pytest.param(
            b"(,) Tj ET BT /F1 11 Tf 107 696 Td (NEXT) Tj ( ROW) Tj ET",
            ["ACME,", "NEXT ROW"],
            id="close-lower-row",
        ),
        pytest.param(
            b"/F1 4 Tf -2 Ts (2) Tj 0 Ts /F1 11 Tf (NEXT) Tj ET",
            ["ACME", "2", "NEXT"],
            id="descending-digit",
        ),
        pytest.param(
            b"-2 Ts (.) Tj ET",
            ["ACME", "."],
            id="no-vertical-overlap",
        ),
        pytest.param(
            b"/F1 20 Tf -1.6 Ts (,) Tj ET",
            ["ACME", ","],
            id="descent-exceeds-tolerance",
        ),
    ],
)
def test_descending_punctuation_row_boundaries(
    tmp_path: Path, runs: bytes, expected: list[str]
) -> None:
    pdf_path = tmp_path / "row_boundaries.pdf"
    pdf_path.write_bytes(_build_pdf(b"BT /F1 11 Tf 72 700 Td (ACME) Tj " + runs))

    assert [text for text, _ in _cell_texts(pdf_path)] == expected


def test_joined_comma_keeps_adjacent_superscript_separate() -> None:
    """Joining the comma does not copy a nearby superscript into its cell."""
    pdf_path = Path("tests/data/latex/sources/2412.19437/figures/basic_arch.pdf")
    texts = [text for text, _ in _cell_texts(pdf_path)]
    v = "\U0001d42f" * 2  # Mathematical bold small v.
    t = "\U0001d461" * 2  # Mathematical italic small t.
    i = "\U0001d456" * 2  # Mathematical italic small i.
    superscript = "\U0001d436" * 2  # Mathematical italic capital C.
    prefix = "{" + v
    index = next(n for n, text in enumerate(texts) if text.startswith(prefix))
    assert texts[index : index + 2] == [f"{prefix}{t},{i}", superscript]
