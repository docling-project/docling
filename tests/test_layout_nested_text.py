# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Overlapping text proposals must not interleave physical paragraph lines."""

import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)

from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor


def _box(left, top, right, bottom):
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def _cell(index, text, box):
    return TextCell(
        index=index,
        text=text,
        orig=text,
        rect=BoundingRectangle.from_bounding_box(box),
        from_ocr=False,
    )


def _cluster(index, box, confidence, label=DocItemLabel.TEXT):
    return Cluster(id=index, bbox=box, confidence=confidence, label=label)


def _postprocess(cells, clusters, page_no=1):
    full = _box(0, 0, 600, 800)
    geometry = PdfPageGeometry(
        angle=0,
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        rect=BoundingRectangle.from_bounding_box(full),
        art_bbox=full,
        bleed_bbox=full,
        crop_bbox=full,
        media_bbox=full,
        trim_bbox=full,
    )
    page = Page(
        page_no=page_no,
        size=Size(width=600, height=800),
        parsed_page=SegmentedPdfPage(
            dimension=geometry, textline_cells=cells, char_cells=[], word_cells=[]
        ),
    )
    return LayoutPostprocessor(
        page, clusters, LayoutPostprocessorOptions()
    ).postprocess()


@pytest.mark.parametrize("column_offset", [0, 300])
@pytest.mark.parametrize("paragraph_gap", [4, 20])
@pytest.mark.parametrize("page_no", [1, 2])
def test_broad_text_proposal_does_not_take_paragraph_lines(
    column_offset, paragraph_gap, page_no
):
    # A lower-confidence prediction covers both paragraphs. Two lines extend
    # one point past their paragraph predictions, so max-coverage assignment
    # alone incorrectly moves those lines into the broad prediction.
    left = column_offset + 10
    right = column_offset + 200
    second_top = 40 + paragraph_gap
    rows = [
        ("Prior first", 10, right - 5),
        ("prior continuation", 25, right + 1),
        ("Current first", second_top, right - 5),
        ("current middle", second_top + 15, right + 1),
        ("current last.", second_top + 30, right - 5),
    ]
    cells = [
        _cell(i, text, _box(left, top, end, top + 10))
        for i, (text, top, end) in enumerate(rows)
    ]
    clusters = [
        _cluster(0, _box(left - 1, 9, right, 36), 0.9),
        _cluster(1, _box(left - 1, second_top - 1, right, second_top + 41), 0.85),
        _cluster(2, _box(left - 1, 9, right + 2, second_top + 41), 0.5),
    ]

    result = _postprocess(cells, clusters, page_no)

    assert [[cell.text for cell in c.cells] for c in result] == [
        ["Prior first", "prior continuation"],
        ["Current first", "current middle", "current last."],
    ]


def test_nested_predictions_restore_separated_paragraphs():
    # A confident parent spans a paragraph gap. The smaller predictions
    # corroborate both paragraphs, including a slightly clipped last line.
    cells = [
        _cell(0, "Earlier paragraph.", _box(10, 10, 195, 20)),
        _cell(1, "Current first", _box(10, 40, 195, 50)),
        _cell(2, "current middle", _box(10, 55, 201, 65)),
        _cell(3, "current last.", _box(10, 70, 195, 80)),
    ]
    clusters = [
        _cluster(0, _box(9, 9, 200, 78), 0.91),
        _cluster(1, _box(9, 9, 198, 21), 0.57),
        _cluster(2, _box(9, 39, 202, 81), 0.52),
    ]

    result = _postprocess(cells, clusters)

    assert [[cell.text for cell in c.cells] for c in result] == [
        ["Earlier paragraph."],
        ["Current first", "current middle", "current last."],
    ]


@pytest.mark.parametrize("line_gap", [4, 20])
def test_single_line_predictions_do_not_fragment_a_compact_text_block(line_gap):
    # Publication metadata is a compact four-line block. A prediction for
    # each slightly clipped line must not erase its grouping without paragraph
    # whitespace, even though its parent is a better fit for the complete lines.
    texts = ["Published April 2", "Publisher", "Editor", "www.example.org"]
    tops = [10 + index * (10 + line_gap) for index in range(len(texts))]
    cells = [
        _cell(index, text, _box(10, top, 150, top + 10))
        for index, (text, top) in enumerate(zip(texts, tops))
    ]
    clusters = [
        _cluster(index, _box(9, top - 1, 149, top + 11), 0.9)
        for index, top in enumerate(tops)
    ]
    clusters.append(_cluster(4, _box(9, 9, 151, tops[-1] + 11), 0.6))

    result = _postprocess(cells, clusters)

    expected = [texts] if line_gap < 10 else [[text] for text in texts]
    assert [[cell.text for cell in cluster.cells] for cluster in result] == expected


def test_text_proposal_with_unique_continuation_is_retained():
    cells = [
        _cell(0, "Opening", _box(10, 10, 90, 20)),
        _cell(1, "Continuation one", _box(10, 40, 90, 50)),
        _cell(2, "Continuation two.", _box(10, 60, 90, 70)),
    ]
    clusters = [
        _cluster(0, _box(0, 0, 100, 100), 0.6),
        _cluster(1, _box(0, 0, 100, 25), 0.9),
    ]

    result = _postprocess(cells, clusters)

    assert len(result) == 1
    assert [cell.text for cell in result[0].cells] == [c.text for c in cells]


def test_better_contained_text_does_not_move_to_partial_prediction():
    cells = [_cell(0, "One complete line.", _box(0, 10, 100, 20))]
    clusters = [
        _cluster(0, _box(0, 0, 100, 30), 0.5),
        _cluster(1, _box(40, 0, 100, 30), 0.95),
    ]

    result = _postprocess(cells, clusters)

    assert len(result) == 1
    assert result[0].id == 0
    assert result[0].cells == cells


def test_overlapping_text_does_not_demote_heading():
    cells = [_cell(0, "Heading", _box(10, 10, 90, 20))]
    clusters = [
        _cluster(0, _box(0, 0, 100, 30), 0.5, DocItemLabel.SECTION_HEADER),
        _cluster(1, _box(1, 1, 101, 31), 0.95),
    ]

    result = _postprocess(cells, clusters)

    assert len(result) == 1
    assert result[0].label == DocItemLabel.SECTION_HEADER
    assert result[0].cells == cells


def test_suppressing_text_does_not_expose_a_competing_heading():
    cells = [_cell(0, "Body sentence.", _box(0, 10, 100, 20))]
    clusters = [
        _cluster(0, _box(0, 0, 100, 30), 0.5),
        _cluster(1, _box(15, 0, 100, 30), 0.95),
        _cluster(2, _box(10, 0, 100, 30), 0.7, DocItemLabel.SECTION_HEADER),
    ]

    result = _postprocess(cells, clusters)

    assert len(result) == 1
    assert result[0].label == DocItemLabel.TEXT
    assert result[0].cells == cells


@pytest.mark.parametrize("line_gap", [2, 5, 9])
def test_nested_line_fragments_do_not_split_one_paragraph(line_gap):
    cells = [
        _cell(
            i,
            f"Line {i}.",
            _box(10, 10 + i * (10 + line_gap), 190, 20 + i * (10 + line_gap)),
        )
        for i in range(4)
    ]
    clusters = [
        _cluster(0, _box(9, 9, 201, 21 + 3 * (10 + line_gap)), 0.95),
        _cluster(1, _box(9, 9, 201, 21 + (10 + line_gap)), 0.6),
        _cluster(
            2, _box(9, 9 + 2 * (10 + line_gap), 201, 21 + 3 * (10 + line_gap)), 0.6
        ),
    ]
    result = _postprocess(cells, clusters)
    assert [[cell.text for cell in c.cells] for c in result] == [
        [c.text for c in cells]
    ]


def test_paragraph_gap_without_complete_predictions_does_not_invent_a_split():
    cells = [
        _cell(0, "First paragraph.", _box(10, 10, 190, 20)),
        _cell(1, "Second opening", _box(10, 40, 190, 50)),
        _cell(2, "unique continuation.", _box(10, 55, 190, 65)),
    ]
    clusters = [
        _cluster(0, _box(9, 9, 201, 66), 0.95),
        _cluster(1, _box(9, 9, 201, 21), 0.6),
        _cluster(2, _box(9, 39, 201, 51), 0.6),
    ]
    result = _postprocess(cells, clusters)
    assert [[cell.text for cell in c.cells] for c in result] == [
        [c.text for c in cells]
    ]


@pytest.mark.parametrize(
    "hint_confidence,expected_split",
    [(None, False), (0.1, False), (0.35, True), (0.9, True)],
)
def test_large_blank_line_requires_a_paragraph_prediction(
    hint_confidence, expected_split
):
    rows = [
        (10, "First opening"),
        (25, "first ending."),
        (55, "Second opening"),
        (70, "second ending."),
    ]
    cells = [
        _cell(i, text, _box(10, top, 190, top + 10))
        for i, (top, text) in enumerate(rows)
    ]
    clusters = [_cluster(0, _box(9, 9, 201, 81), 0.96)]
    if hint_confidence is not None:
        # Inference can retain only a weak proposal for one of the paragraphs.
        clusters.append(_cluster(1, _box(9, 9, 201, 36), hint_confidence))
    result = _postprocess(cells, clusters)
    actual = [[cell.text for cell in c.cells] for c in result]
    assert actual == (
        [[c.text for c in cells[:2]], [c.text for c in cells[2:]]]
        if expected_split
        else [[c.text for c in cells]]
    )


@pytest.mark.parametrize("overlap_lines", [False, True])
def test_paragraph_hints_do_not_split_columns_or_overlapping_lines(overlap_lines):
    rows = [
        (10, 10),
        (10, 25),
        (10 if overlap_lines else 310, 55),
        (10 if overlap_lines else 310, 70),
    ]
    cells = [
        _cell(i, f"Line {i}.", _box(left, top, left + 180, top + 10))
        for i, (left, top) in enumerate(rows)
    ]
    if overlap_lines:
        cells.append(_cell(4, "Another column.", _box(310, 10, 490, 20)))
    clusters = [
        _cluster(0, _box(9, 9, 501, 81), 0.96),
        _cluster(1, _box(9, 9, 501, 36), 0.35),
    ]
    result = _postprocess(cells, clusters)
    assert len(result) == 1
    assert {cell.index for cell in result[0].cells} == {cell.index for cell in cells}
