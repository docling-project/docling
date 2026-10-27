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
