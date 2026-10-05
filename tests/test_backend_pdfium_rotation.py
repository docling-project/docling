# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Text-cell geometry for pages carrying a ``/Rotate`` entry.

PDFium reports text coordinates in the page's unrotated (MediaBox) frame and
ignores ``/Rotate``, while ``PdfPage.get_size()`` and the rendered page bitmap
are already in the rotated *display* frame. These tests pin down that the
pypdfium2 backend hands back a single, consistent frame.

The fixtures are built as raw PDF bytes so the tests need no models, no GPU and
no extra dependency: the same three words are drawn at the same *displayed*
position four times, once with a plain landscape MediaBox and once for each of
``/Rotate 90``, ``180`` and ``270``. All four documents look identical to a
reader, so the text cells the backend extracts from them must match too.
"""

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

# The page every variant *displays*: US Letter, landscape.
DISPLAY_WIDTH = 792.0
DISPLAY_HEIGHT = 612.0

FONT_SIZE = 24
WORDS = ("ALPHA", "BRAVO", "CHARLIE")
# Left edge of each word and the shared baseline, in the display frame
# (bottom-left origin). The gaps are small enough that the backend's horizontal
# merging collapses the three words into a single cell.
WORD_X = (50.0, 130.0, 215.0)
BASELINE_Y = 512.0

# Rects come straight from the same glyph metrics in every variant, so the only
# difference is floating-point noise from the rotation arithmetic.
TOLERANCE = 0.5


def _media_box(rotation: int) -> tuple[float, float]:
    """Size of the MediaBox needed to display as ``DISPLAY_WIDTH x DISPLAY_HEIGHT``."""
    if rotation in (90, 270):
        return DISPLAY_HEIGHT, DISPLAY_WIDTH
    return DISPLAY_WIDTH, DISPLAY_HEIGHT


def _text_matrix(rotation: int, x: float) -> tuple[float, ...]:
    """Text matrix placing a word at ``(x, BASELINE_Y)`` of the *displayed* page.

    ``/Rotate`` turns the page clockwise for display, so the content stream has
    to counter-rotate for the text to come out upright and in the same place.
    """
    if rotation == 90:
        return (0.0, 1.0, -1.0, 0.0, DISPLAY_HEIGHT - BASELINE_Y, x)
    elif rotation == 180:
        return (-1.0, 0.0, 0.0, -1.0, DISPLAY_WIDTH - x, DISPLAY_HEIGHT - BASELINE_Y)
    elif rotation == 270:
        return (0.0, -1.0, 1.0, 0.0, BASELINE_Y, DISPLAY_WIDTH - x)
    return (1.0, 0.0, 0.0, 1.0, x, BASELINE_Y)


def _build_pdf(rotation: int) -> bytes:
    """A one-page PDF with the three words, stored with the given ``/Rotate``."""
    media_width, media_height = _media_box(rotation)
    blocks = [
        "BT /F1 {size} Tf {matrix} Tm ({word}) Tj ET".format(
            size=FONT_SIZE,
            matrix=" ".join(f"{v:g}" for v in _text_matrix(rotation, x)),
            word=word,
        )
        for x, word in zip(WORD_X, WORDS, strict=True)
    ]
    content = ("\n".join(blocks) + "\n").encode("ascii")

    rotate_entry = f" /Rotate {rotation}" if rotation else ""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {media_width:g} {media_height:g}]{rotate_entry} "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    startxref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def _write_pdf(tmp_path: Path, rotation: int) -> Path:
    path = tmp_path / f"rotate_{rotation}.pdf"
    path.write_bytes(_build_pdf(rotation))
    return path


def _page_backend(tmp_path: Path, rotation: int) -> PyPdfiumPageBackend:
    in_doc = InputDocument(
        path_or_stream=_write_pdf(tmp_path, rotation),
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )
    return in_doc._backend.load_page(0)


def _union(cells: list[TextCell]) -> tuple[float, float, float, float]:
    """``(l, t, r, b)`` covering every cell, in top-left origin."""
    boxes = [cell.rect.to_bounding_box() for cell in cells]
    assert boxes, "no text cells were extracted"
    assert all(box.coord_origin == CoordOrigin.TOPLEFT for box in boxes)
    return (
        min(box.l for box in boxes),
        min(box.t for box in boxes),
        max(box.r for box in boxes),
        max(box.b for box in boxes),
    )


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_text_cells_match_unrotated_twin(tmp_path: Path, rotation: int) -> None:
    """A rotated page and its flat twin display alike, so their cells must agree."""
    flat = list(_page_backend(tmp_path, 0).get_text_cells())
    rotated = list(_page_backend(tmp_path, rotation).get_text_cells())

    assert [cell.text for cell in rotated] == [cell.text for cell in flat]
    assert _union(rotated) == pytest.approx(_union(flat), abs=TOLERANCE)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_text_cells_lie_inside_the_page(tmp_path: Path, rotation: int) -> None:
    """Cells must sit within the page as ``get_size()`` reports it."""
    page_backend = _page_backend(tmp_path, rotation)
    size = page_backend.get_size()
    assert (size.width, size.height) == (DISPLAY_WIDTH, DISPLAY_HEIGHT)

    for cell in page_backend.get_text_cells():
        box = cell.rect.to_bounding_box()
        assert 0 <= box.l <= box.r <= size.width
        assert 0 <= box.t <= box.b <= size.height


def test_unrotated_page_geometry_is_unchanged(tmp_path: Path) -> None:
    """Without ``/Rotate`` the cells stay exactly what PDFium reports."""
    path = _write_pdf(tmp_path, 0)

    pdf = pdfium.PdfDocument(path)
    try:
        text_page = pdf[0].get_textpage()
        rects = [text_page.get_rect(i) for i in range(text_page.count_rects())]
    finally:
        pdf.close()
    assert rects

    expected = (
        min(rect[0] for rect in rects),
        DISPLAY_HEIGHT - max(rect[3] for rect in rects),
        max(rect[2] for rect in rects),
        DISPLAY_HEIGHT - min(rect[1] for rect in rects),
    )

    in_doc = InputDocument(
        path_or_stream=path, format=InputFormat.PDF, backend=PyPdfiumDocumentBackend
    )
    cells = list(in_doc._backend.load_page(0).get_text_cells())
    assert _union(cells) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_get_text_in_rect_accepts_display_coordinates(
    tmp_path: Path, rotation: int
) -> None:
    """Re-extraction must undo the rotation before querying PDFium again."""
    page_backend = _page_backend(tmp_path, rotation)
    left, top, right, bottom = _union(list(page_backend.get_text_cells()))

    text = page_backend.get_text_in_rect(
        BoundingBox(l=left - 2, t=top - 2, r=right + 2, b=bottom + 2)
    )

    for word in WORDS:
        assert word in text


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_merged_cell_text_survives_rotation(tmp_path: Path, rotation: int) -> None:
    """Merging re-reads the text through PDFium, which needs the unrotated frame."""
    cells = list(_page_backend(tmp_path, rotation).get_text_cells())

    # The three words sit on one row with sub-word gaps, so they merge into one
    # cell whose text is re-extracted from the merged bounding box.
    assert len(cells) == 1
    for word in WORDS:
        assert word in cells[0].text
