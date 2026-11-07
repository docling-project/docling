# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A text line joined across columns must still yield the words inside one cell.

docling-parse 7.21+ can merge the words of several table columns into one
text line. ``get_text_in_rect`` keeps a line only when more than half of it
lies in the rectangle, so each column comes back empty. See #5136.
"""

from docling_core.types.doc import BoundingBox

from docling.backend.docling_parse_backend import ThreadedDoclingParsePageBackend

_LINE = BoundingBox(l=40.0, t=92.82, r=250.5, b=102.07)
_LINE_TEXT = "An. gambiae s.l. An. funestus Culex spp."
_WORDS = (
    ("An.", BoundingBox(l=40.0, t=92.82, r=55.0, b=102.07)),
    ("gambiae", BoundingBox(l=57.8, t=92.82, r=96.1, b=102.07)),
    ("s.l.", BoundingBox(l=98.9, t=92.82, r=111.7, b=102.07)),
    ("An.", BoundingBox(l=129.7, t=92.82, r=144.7, b=102.07)),
    ("funestus", BoundingBox(l=147.5, t=92.82, r=185.3, b=102.07)),
    ("Culex", BoundingBox(l=203.3, t=92.82, r=228.9, b=102.07)),
    ("spp.", BoundingBox(l=231.6, t=92.82, r=250.5, b=102.07)),
)
_COLUMNS = (
    "An. gambiae s.l.",
    "An. funestus",
    "Culex spp.",
)


class _Rect:
    def __init__(self, bbox: BoundingBox) -> None:
        self._bbox = bbox

    def to_bounding_box(self) -> BoundingBox:
        return self._bbox


class _Cell:
    def __init__(self, text: str, bbox: BoundingBox) -> None:
        self.text = text
        self.orig = text
        self.rect = _Rect(bbox)

    def to_top_left_origin(self, _page_height: float) -> "_Cell":
        return self


class _Dimension:
    height = 200.0


class _Page:
    def __init__(self) -> None:
        self.dimension = _Dimension()
        self.textline_cells = [_Cell(_LINE_TEXT, _LINE)]
        self.char_cells: list[_Cell] = []
        self.word_cells = [_Cell(text, bbox) for text, bbox in _WORDS]


class _Result:
    success = True
    page_number = 1

    def get_page(self) -> _Page:
        return _Page()


def _column_box(text: str) -> BoundingBox:
    wanted = text.split()
    texts = [word for word, _bbox in _WORDS]
    start = 0
    while texts[start : start + len(wanted)] != wanted:
        start += 1
    chosen = _WORDS[start : start + len(wanted)]
    return BoundingBox(
        l=min(bbox.l for _word, bbox in chosen) - 1,
        t=_LINE.t - 1,
        r=max(bbox.r for _word, bbox in chosen) + 1,
        b=_LINE.b + 1,
    )


def test_column_rect_keeps_the_words_of_a_line_that_spans_it() -> None:
    backend = ThreadedDoclingParsePageBackend(_Result())

    # More than half of the joined line is outside every column.
    for phrase in _COLUMNS:
        box = _column_box(phrase)
        assert _LINE.intersection_over_self(box) <= 0.5
        assert backend.get_text_in_rect(box) == phrase

    # A rectangle that holds the line still returns the line, unchanged.
    whole = BoundingBox(l=_LINE.l - 1, t=_LINE.t - 1, r=_LINE.r + 1, b=_LINE.b + 1)
    assert backend.get_text_in_rect(whole) == _LINE_TEXT

    # Most of the line is inside, so the cell still receives the whole line.
    mostly = BoundingBox(
        l=_LINE.l,
        t=_LINE.t,
        r=_LINE.l + 0.6 * _LINE.width,
        b=_LINE.b,
    )
    assert _LINE.intersection_over_self(mostly) > 0.5
    assert backend.get_text_in_rect(mostly) == _LINE_TEXT
