# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for reading the lines of a printed table of contents from parsed PDF words."""

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)

from docling.utils.printed_toc import TocLine, page_label_value, read_toc_lines

_REGION = BoundingBox(l=50, t=100, r=550, b=700, coord_origin=CoordOrigin.TOPLEFT)


def _box(left, top, right, bottom):
    return BoundingBox(
        l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT
    )


def _page(words) -> SegmentedPdfPage:
    """A page from ``(text, left, top, right)`` words, each 10 points high."""
    full = _box(0, 0, 600, 800)
    cells = [
        TextCell(
            index=i,
            rect=BoundingRectangle.from_bounding_box(_box(left, top, right, top + 10)),
            text=text,
            orig=text,
            from_ocr=False,
        )
        for i, (text, left, top, right) in enumerate(words)
    ]
    return SegmentedPdfPage(
        dimension=PdfPageGeometry(
            angle=0,
            boundary_type=PdfPageBoundaryType.CROP_BOX,
            rect=BoundingRectangle.from_bounding_box(full),
            art_bbox=full,
            bleed_bbox=full,
            crop_bbox=full,
            media_bbox=full,
            trim_bbox=full,
        ),
        char_cells=[],
        word_cells=cells,
        textline_cells=[],
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("12", ("arabic", 12)),
        ("xvii", ("roman", 17)),
        ("XIV", ("roman", 14)),
        ("iv", ("roman", 4)),
        ("Scope", None),
        ("", None),
    ],
)
def test_page_label_value(label, expected):
    assert page_label_value(label) == expected


def test_leaders_and_page_labels_are_split_from_the_entry():
    page = _page(
        [
            ("Foreword", 70, 110, 120),
            (".", 130, 110, 133),
            (".", 140, 110, 143),
            ("vi", 530, 110, 540),
            ("2.1", 70, 130, 85),
            ("General.........", 95, 130, 200),
            ("...12", 525, 130, 540),
        ]
    )

    lines = read_toc_lines(_REGION, page)

    assert [(line.text, line.page_label) for line in lines] == [
        ("Foreword", "vi"),
        ("2.1 General", "12"),
    ]


def test_a_right_aligned_label_needs_a_wide_gap_without_leaders():
    page = _page(
        [
            ("Chapter", 70, 110, 110),
            ("16", 114, 110, 124),
            ("Conclusion", 130, 110, 190),
            ("873", 520, 110, 540),
            ("Part", 70, 130, 95),
            ("II", 99, 130, 108),
        ]
    )

    lines = read_toc_lines(_REGION, page)

    assert lines == [
        TocLine(text="Chapter 16 Conclusion", page_label="873", indent=0, height=10),
        TocLine(text="Part II", page_label=None, indent=0, height=10),
    ]


def test_indent_is_relative_to_the_region_and_outside_words_are_ignored():
    page = _page(
        [
            ("Contents", 70, 60, 120),  # above the region
            ("Intro", 70, 110, 100),
            ("1", 530, 110, 536),
            ("Background", 90, 130, 150),
            ("2", 530, 130, 536),
        ]
    )

    lines = read_toc_lines(_REGION, page)

    assert [(line.text, line.indent) for line in lines] == [
        ("Intro", 0),
        ("Background", 20),
    ]
