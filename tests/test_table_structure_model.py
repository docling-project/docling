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


def test_table_tokens_keep_ocr_text_next_to_native_words() -> None:
    # A scanned form with one filled-in native value: the OCR header must still
    # reach the table, and a native word under OCR text must not be doubled.
    header = _cell(BoundingBox(l=0, t=0, r=50, b=10), "Week Begin", from_ocr=True)
    native_line = _cell(BoundingBox(l=60, t=20, r=90, b=30), "5000", from_ocr=False)
    filled_value = _cell(BoundingBox(l=60, t=20, r=90, b=30), "5000", from_ocr=False)
    hidden_word = _cell(BoundingBox(l=5, t=2, r=20, b=8), "Week", from_ocr=False)

    tokens = _words_and_ocr_cells(
        word_cells=[filled_value, hidden_word],
        cluster_cells=[header, native_line],
    )

    assert [c.text for c in tokens] == ["5000", "Week Begin"]
