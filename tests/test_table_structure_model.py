# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.models.stages.table_structure.table_structure_model import (
    _words_and_ocr_cells,
)


def _cell(bbox: BoundingBox, text: str, from_ocr: bool) -> TextCell:
    return TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=from_ocr,
    )


HEADER_BOX = BoundingBox(l=0, t=0, r=50, b=10)
VALUE_BOX = BoundingBox(l=60, t=20, r=90, b=30)


def test_table_tokens_keep_ocr_lines_next_to_native_words() -> None:
    # A scanned form with one filled-in native value: the OCR header must still
    # reach the table, and a native word under OCR text must not be doubled.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    filled_value = _cell(VALUE_BOX, "5000", from_ocr=False)
    hidden_word = _cell(BoundingBox(l=5, t=2, r=20, b=8), "Week", from_ocr=False)

    tokens = _words_and_ocr_cells(
        word_cells=[filled_value, hidden_word],
        cluster_cells=[header, native_line],
    )

    assert [c.text for c in tokens] == ["5000", "Week Begin"]


def test_table_tokens_prefer_ocr_words_over_their_line() -> None:
    # An OCR engine with word output (e.g. a third-party plugin) also writes
    # OCR word cells. They replace their OCR line. The OCR word on the native
    # value is dropped: the OCR merge kept the native text there.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    words = [
        _cell(BoundingBox(l=0, t=0, r=22, b=10), "Week", from_ocr=True),
        _cell(BoundingBox(l=26, t=0, r=50, b=10), "Begin", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=False),
    ]

    tokens = _words_and_ocr_cells(word_cells=words, cluster_cells=[header, native_line])

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week", True),
        ("Begin", True),
    ]


def test_table_tokens_keep_native_lines_without_word_cells() -> None:
    # Backends such as PyPdfiumDocumentBackend emit textline_cells only
    # (word_cells is empty). When OCR keeps even one cell in the table
    # cluster, all surviving native lines must still reach TableFormer.
    col_header = _cell(BoundingBox(l=10, t=10, r=80, b=22), "Function", from_ocr=False)
    row_value = _cell(BoundingBox(l=10, t=30, r=80, b=42), "Govern", from_ocr=False)
    ocr_noise = _cell(BoundingBox(l=120, t=30, r=150, b=42), "| =|", from_ocr=True)

    tokens = _words_and_ocr_cells(
        word_cells=[],
        cluster_cells=[col_header, row_value, ocr_noise],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("Function", False),
        ("Govern", False),
        ("| =|", True),
    ]


def test_table_tokens_drop_ocr_words_on_native_lines_without_native_words() -> None:
    # When a line-only PDF backend is paired with a word-level OCR plugin,
    # OCR words overlapping a kept native line must be dropped, while OCR
    # words on an OCR line replace that OCR line.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    ocr_words = [
        _cell(BoundingBox(l=0, t=0, r=22, b=10), "Week", from_ocr=True),
        _cell(BoundingBox(l=26, t=0, r=50, b=10), "Begin", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=True),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=ocr_words,
        cluster_cells=[header, native_line],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week", True),
        ("Begin", True),
    ]


def test_table_tokens_keep_uncovered_native_lines_alongside_native_words() -> None:
    # Native lines covered by finer native words use those words, while any
    # surviving native line without covering native words is retained.
    row1 = _cell(BoundingBox(l=10, t=10, r=90, b=22), "Row 1", from_ocr=False)
    row2 = _cell(BoundingBox(l=10, t=30, r=90, b=42), "Row 2", from_ocr=False)
    ocr_note = _cell(BoundingBox(l=10, t=50, r=90, b=62), "Note", from_ocr=True)
    words = [
        _cell(BoundingBox(l=10, t=10, r=45, b=22), "Row", from_ocr=False),
        _cell(BoundingBox(l=55, t=10, r=90, b=22), "1", from_ocr=False),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=words,
        cluster_cells=[row1, row2, ocr_note],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("Row", False),
        ("1", False),
        ("Row 2", False),
        ("Note", True),
    ]


def test_table_tokens_ignore_blank_word_cells() -> None:
    # A whitespace-only word cell must not mask a non-empty line cell.
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    ocr_line = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    blank_words = [
        _cell(VALUE_BOX, "   ", from_ocr=False),
        _cell(HEADER_BOX, " ", from_ocr=True),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=blank_words,
        cluster_cells=[native_line, ocr_line],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week Begin", True),
    ]
