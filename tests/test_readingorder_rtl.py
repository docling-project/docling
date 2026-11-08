# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for right-to-left page support in the rule-based reading order."""

import copy
from pathlib import PurePath

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel

from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    ContainerElement,
    InputFormat,
    Page,
    TextElement,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.models.postprocessing.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
    SeparatorElement,
    detect_text_direction,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
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


def _three_column_grid(text: str, cid_order: str):
    # Three columns x two rows. cid_order "ltr" assigns consecutive cids
    # left-to-right per row (typical LTR extraction), "rtl" right-to-left
    # (typical RTL extraction): the predictor links only consecutive cids.
    rows = {"1": (300.0, 450.0), "2": (100.0, 250.0)}
    cols = {"L": (20.0, 200.0), "M": (210.0, 390.0), "R": (400.0, 480.0)}
    sequence = ["L", "M", "R"] if cid_order == "ltr" else ["R", "M", "L"]
    elements = {}
    cid = 0
    for row in ["1", "2"]:
        for col in sequence:
            cid += 1
            left, right = cols[col]
            b, t = rows[row]
            elements[f"{col}{row}"] = _element(
                cid, left, right, f"{text} {col}{row}", bottom=b, top=t
            )
    return elements


def _vertical_separator(x: float) -> SeparatorElement:
    return SeparatorElement(
        cid=99,
        page_no=1,
        page_size=_PAGE,
        orientation="vertical",
        l=x,
        r=x,
        b=50.0,
        t=460.0,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


def test_rtl_mirrors_vertical_separators() -> None:
    # A vertical rule between the columns must mirror along with the
    # elements. The rule sits between the L and M columns; after mirroring it
    # must sit between the mirrored L and M columns. A centred rule between
    # two columns cannot detect a missing mirror: two-column gaps map onto
    # themselves.
    grid = _three_column_grid(_HEBREW, cid_order="rtl")
    separator = _vertical_separator(205.0)  # between L and M
    ordered = ReadingOrderPredictor().predict_reading_order(
        list(grid.values()), page_separators=[separator], page_direction="auto"
    )
    # R-M flow row by row, then the L column split off by the mirrored rule.
    # With an unmirrored rule the flow runs through L: R1 M1 L1 R2 M2 L2.
    assert [e.cid for e in ordered] == [
        grid["R1"].cid,
        grid["M1"].cid,
        grid["R2"].cid,
        grid["M2"].cid,
        grid["L1"].cid,
        grid["L2"].cid,
    ]


def test_ltr_vertical_separator_control() -> None:
    # Same grid on an LTR page: the unmirrored rule splits L from the M-R flow.
    grid = _three_column_grid(_ENGLISH, cid_order="ltr")
    separator = _vertical_separator(205.0)
    ordered = ReadingOrderPredictor().predict_reading_order(
        list(grid.values()), page_separators=[separator]
    )
    assert [e.cid for e in ordered] == [
        grid["L1"].cid,
        grid["L2"].cid,
        grid["M1"].cid,
        grid["R1"].cid,
        grid["M2"].cid,
        grid["R2"].cid,
    ]


def test_forced_rtl_merges_latin_split_word() -> None:
    # A Latin word split across the columns of an RTL page merges in reading
    # order: the first fragment sits in the right column, the second in the
    # left one.
    first = _element(10, 300.0, 480.0, "trans-")
    second = _element(11, 20.0, 200.0, "lation")
    merges = ReadingOrderPredictor().predict_merges(
        [first, second], page_direction="rtl"
    )
    assert merges == {first.cid: [second.cid]}


def test_ltr_merges_unchanged_with_direction_option() -> None:
    first = _element(10, 20.0, 200.0, "trans-")
    second = _element(11, 300.0, 480.0, "lation")
    predictor = ReadingOrderPredictor()
    expected = predictor.predict_merges([first, second])
    assert expected == {first.cid: [second.cid]}
    assert predictor.predict_merges([first, second], page_direction="ltr") == expected
    assert predictor.predict_merges([first, second], page_direction="auto") == expected


def test_rtl_does_not_merge_hebrew_texts() -> None:
    # The merge regexes only accept Latin fragments, so RTL scripts never
    # merge, mirrored or not.
    right, left = _two_columns(_HEBREW)
    merges = ReadingOrderPredictor().predict_merges([right, left], page_direction="rtl")
    assert merges == {}


def test_auto_merges_latin_split_word_on_hebrew_page() -> None:
    # Auto-detect marks the Hebrew-majority page RTL; an embedded Latin split
    # word still merges in reading order.
    first = _element(10, 300.0, 480.0, _HEBREW + " trans-")
    second = _element(11, 20.0, 200.0, "lation")
    merges = ReadingOrderPredictor().predict_merges(
        [first, second], page_direction="auto"
    )
    assert merges == {first.cid: [second.cid]}


def test_merges_do_not_mutate_input_geometry() -> None:
    first = _element(10, 300.0, 480.0, "trans-")
    second = _element(11, 20.0, 200.0, "lation")
    before = (first.l, first.r, second.l, second.r)
    ReadingOrderPredictor().predict_merges([first, second], page_direction="rtl")
    assert (first.l, first.r, second.l, second.r) == before


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


# --- Model-level: one direction decision per page -------------------------


def _cluster(cid: int, label: DocItemLabel, bbox: tuple) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
    )


def _text_element(cid: int, bbox: tuple, text: str) -> TextElement:
    return TextElement(
        label=DocItemLabel.TEXT,
        id=cid,
        text=text,
        page_no=1,
        cluster=_cluster(cid, DocItemLabel.TEXT, bbox),
    )


def _conversion_result(elements: list) -> ConversionResult:
    input_doc = InputDocument.model_construct(
        file=PurePath("input.pdf"),
        document_hash="0" * 64,
        valid=True,
        format=InputFormat.PDF,
    )
    return ConversionResult(
        input=input_doc,
        pages=[Page(page_no=1, size=Size(width=500, height=500))],
        assembled=AssembledUnit(elements=elements, body=elements),
    )


def _mixed_page(container_child_texts: tuple[str, str]) -> ConversionResult:
    # Hebrew top-level columns (right is read first) and an English FORM
    # container with two side-by-side children below.
    hebrew_right = _text_element(2, (300, 40, 480, 240), _HEBREW + " aleph.")
    hebrew_left = _text_element(3, (20, 40, 200, 240), _HEBREW + " bet.")
    eng_right = _text_element(4, (300, 320, 480, 420), container_child_texts[0])
    eng_left = _text_element(5, (20, 320, 200, 420), container_child_texts[1])
    container_cluster = _cluster(1, DocItemLabel.FORM, (0, 300, 500, 480))
    container_cluster.children = [eng_right.cluster, eng_left.cluster]
    container = ContainerElement(
        label=DocItemLabel.FORM, id=1, page_no=1, cluster=container_cluster
    )
    return _conversion_result(
        [container, hebrew_right, hebrew_left, eng_right, eng_left]
    )


def test_auto_decides_once_per_page_across_groups() -> None:
    # Hebrew-majority page: the English container children follow the page
    # direction (right child first) instead of their own English-only group.
    conv_res = _mixed_page(("Right side 1.", "Left side 2."))
    doc = ReadingOrderModel(ReadingOrderOptions(page_direction="auto"))(conv_res)
    texts = [item.text for item in doc.texts]
    assert texts[0].endswith("aleph.")
    assert texts[1].endswith("bet.")
    assert texts[2:] == ["Right side 1.", "Left side 2."]


def test_auto_ltr_page_keeps_hebrew_group_ltr() -> None:
    # English-majority page: the Hebrew container children follow the page
    # direction (left child first).
    eng_right = _text_element(2, (300, 40, 480, 240), _ENGLISH + " right.")
    eng_left = _text_element(3, (20, 40, 200, 240), _ENGLISH + " left.")
    heb_right = _text_element(4, (300, 320, 480, 420), _HEBREW + " aleph.")
    heb_left = _text_element(5, (20, 320, 200, 420), _HEBREW + " bet.")
    container_cluster = _cluster(1, DocItemLabel.FORM, (0, 300, 500, 480))
    container_cluster.children = [heb_right.cluster, heb_left.cluster]
    container = ContainerElement(
        label=DocItemLabel.FORM, id=1, page_no=1, cluster=container_cluster
    )
    conv_res = _conversion_result([container, eng_right, eng_left, heb_right, heb_left])
    doc = ReadingOrderModel(ReadingOrderOptions(page_direction="auto"))(conv_res)
    texts = [item.text for item in doc.texts]
    assert texts[0].endswith("left.")
    assert texts[1].endswith("right.")
    assert texts[2].endswith("bet.")
    assert texts[3].endswith("aleph.")
