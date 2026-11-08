# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for right-to-left page support in the rule-based reading order."""

import copy

from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel

from docling.models.postprocessing.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
    SeparatorElement,
    detect_text_direction,
)

_PAGE = Size(width=500.0, height=500.0)
_HEBREW = "טקסט עברי כאן " * 6
_ARABIC = "نص عربي هنا " * 6
_ENGLISH = "Some English text here " * 6


def _element(
    cid: int,
    left: float,
    right: float,
    text: str,
    page_no: int = 1,
    label: DocItemLabel = DocItemLabel.TEXT,
    bottom: float = 100.0,
    top: float = 400.0,
) -> PageElement:
    return PageElement(
        cid=cid,
        ref=RefItem(cref=f"#/{page_no}/{cid}"),
        text=text,
        page_no=page_no,
        page_size=_PAGE,
        label=label,
        l=left,
        r=right,
        b=bottom,
        t=top,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


def _two_columns(text: str, page_no: int = 1) -> list[PageElement]:
    # Right column holds the first block in RTL reading order.
    return [
        _element(10 * page_no, 300.0, 480.0, text + " right", page_no),
        _element(10 * page_no + 1, 20.0, 200.0, text + " left", page_no),
    ]


def test_detect_text_direction_hebrew() -> None:
    assert detect_text_direction([_HEBREW, _HEBREW]) == "rtl"


def test_detect_text_direction_arabic() -> None:
    assert detect_text_direction([_ARABIC]) == "rtl"


def test_detect_text_direction_english() -> None:
    assert detect_text_direction([_ENGLISH, _ENGLISH]) == "ltr"


def test_detect_text_direction_mixed_majority_rules() -> None:
    # Hebrew-majority page with one English line: rtl.
    assert detect_text_direction([_HEBREW * 3, _ENGLISH]) == "rtl"
    # English-majority page with a single Hebrew term: ltr.
    assert detect_text_direction([_ENGLISH * 3, _HEBREW[:10]]) == "ltr"


def test_detect_text_direction_numbers_only_defaults_ltr() -> None:
    assert detect_text_direction(["2026-11-08", "3.14 (42)"]) == "ltr"


def test_default_keeps_left_column_first_even_for_hebrew() -> None:
    # Opt-in: without page_direction the historical LTR order is preserved.
    right, left = _two_columns(_HEBREW)
    ordered = ReadingOrderPredictor().predict_reading_order([right, left])
    assert [e.cid for e in ordered] == [left.cid, right.cid]


def test_auto_orders_rtl_page_right_column_first() -> None:
    right, left = _two_columns(_HEBREW)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="auto"
    )
    assert [e.cid for e in ordered] == [right.cid, left.cid]


def test_auto_orders_arabic_page_right_column_first() -> None:
    right, left = _two_columns(_ARABIC)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="auto"
    )
    assert [e.cid for e in ordered] == [right.cid, left.cid]


def test_auto_keeps_ltr_page_left_column_first() -> None:
    right, left = _two_columns(_ENGLISH)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="auto"
    )
    assert [e.cid for e in ordered] == [left.cid, right.cid]


def test_forced_rtl_flips_english_page() -> None:
    right, left = _two_columns(_ENGLISH)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="rtl"
    )
    assert [e.cid for e in ordered] == [right.cid, left.cid]


def test_forced_ltr_keeps_hebrew_page() -> None:
    right, left = _two_columns(_HEBREW)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="ltr"
    )
    assert [e.cid for e in ordered] == [left.cid, right.cid]


def test_only_rtl_pages_are_mirrored() -> None:
    rtl_right, rtl_left = _two_columns(_HEBREW, page_no=1)
    ltr_right, ltr_left = _two_columns(_ENGLISH, page_no=2)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [rtl_right, rtl_left, ltr_right, ltr_left], page_direction="auto"
    )
    assert [e.cid for e in ordered] == [
        rtl_right.cid,
        rtl_left.cid,
        ltr_left.cid,
        ltr_right.cid,
    ]


def test_rtl_output_carries_original_coordinates() -> None:
    right, left = _two_columns(_HEBREW)
    ordered = ReadingOrderPredictor().predict_reading_order(
        [right, left], page_direction="auto"
    )
    assert ordered[0] is right
    assert ordered[1] is left
    assert ordered[0].l == 300.0
    assert ordered[0].r == 480.0


def test_inputs_are_not_mutated() -> None:
    right, left = _two_columns(_HEBREW)
    before = [copy.deepcopy(right), copy.deepcopy(left)]
    ReadingOrderPredictor().predict_reading_order([right, left], page_direction="auto")
    assert (right.l, right.r, right.b, right.t) == (
        before[0].l,
        before[0].r,
        before[0].b,
        before[0].t,
    )
    assert (left.l, left.r, left.b, left.t) == (
        before[1].l,
        before[1].r,
        before[1].b,
        before[1].t,
    )


def test_headers_and_footers_follow_page_direction() -> None:
    header_right = _element(
        1,
        300.0,
        480.0,
        _HEBREW,
        label=DocItemLabel.PAGE_HEADER,
        bottom=450.0,
        top=490.0,
    )
    header_left = _element(
        2, 20.0, 200.0, _HEBREW, label=DocItemLabel.PAGE_HEADER, bottom=450.0, top=490.0
    )
    body = _element(3, 20.0, 480.0, _HEBREW, bottom=100.0, top=400.0)
    footer = _element(
        4, 300.0, 480.0, _HEBREW, label=DocItemLabel.PAGE_FOOTER, bottom=10.0, top=50.0
    )
    ordered = ReadingOrderPredictor().predict_reading_order(
        [header_left, header_right, body, footer], page_direction="auto"
    )
    # Headers flip to RTL: right header first. Body and footer keep their slots.
    assert [e.cid for e in ordered] == [2, 1, 3, 4] or [e.cid for e in ordered] == [
        1,
        2,
        3,
        4,
    ]
    # RTL means the right header (cid 1) precedes the left header (cid 2):
    cids = [e.cid for e in ordered]
    assert cids.index(1) < cids.index(2)


def test_rtl_mirrors_vertical_separators() -> None:
    # A vertical rule between the columns must mirror too, so it splits the
    # columns in RTL order instead of prepending the left one.
    right, left = _two_columns(_HEBREW)
    separator = SeparatorElement(
        cid=99,
        page_no=1,
        page_size=_PAGE,
        orientation="vertical",
        l=250.0,
        r=250.0,
        b=50.0,
        t=450.0,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    ordered = ReadingOrderPredictor().predict_reading_order(
        [left, right], page_separators=[separator], page_direction="auto"
    )
    assert [e.cid for e in ordered] == [right.cid, left.cid]


def test_ltr_separators_unchanged_when_opted_out() -> None:
    right, left = _two_columns(_HEBREW)
    separator = SeparatorElement(
        cid=99,
        page_no=1,
        page_size=_PAGE,
        orientation="vertical",
        l=250.0,
        r=250.0,
        b=50.0,
        t=450.0,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    ordered = ReadingOrderPredictor().predict_reading_order(
        [left, right], page_separators=[separator]
    )
    assert [e.cid for e in ordered] == [left.cid, right.cid]
