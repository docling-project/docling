# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Replacing a broken PDF text layer with OCR (OcrOptions.replace_broken_text_layer).

A page whose text layer is broken scores 0 in its parse score; with the option
on, the OCR stage reads every page graded poor (below 0.5) in full and keeps only
its OCR text. Other pages, and pages without a text layer, are left as they were.
"""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import ConfidenceReport, Page, Size
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel, _empty_segmented_page
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)

_BROKEN = "&>66*;B \x1b8?.;7*7,. \x182;.,=8;< \x17869.7<*=287#;898<*5<"
_CLEAN = "We have audited the accompanying financial statements of the village."


def _cell(text: str, *, from_ocr: bool, y: float) -> TextCell:
    return TextCell(
        rect=BoundingRectangle(
            r_x0=10,
            r_y0=y,
            r_x1=300,
            r_y1=y,
            r_x2=300,
            r_y2=y + 12,
            r_x3=10,
            r_y3=y + 12,
        ),
        text=text,
        orig=text,
        from_ocr=from_ocr,
        confidence=0.9 if from_ocr else 1.0,
    )


def _page(*pdf_texts: str, parse_score: float = math.nan) -> Page:
    page = Page(page_no=1)
    page.size = Size(width=600.0, height=800.0)
    page.parsed_page = _empty_segmented_page(page)
    page.parsed_page.textline_cells = [
        _cell(text, from_ocr=False, y=700 - 20 * i) for i, text in enumerate(pdf_texts)
    ]
    page.parsed_page.has_lines = bool(pdf_texts)
    page._parse_score = parse_score
    return page


def _ocr_model(mode: OcrMode, replace: bool, **methods: Mock):
    options = SimpleNamespace(mode=mode, replace_broken_text_layer=replace)
    return SimpleNamespace(options=options, **methods)


@pytest.mark.parametrize(
    ("texts", "expected"), [((_BROKEN,) * 4, 0.0), ((_CLEAN,) * 4, 1.0)]
)
def test_parse_score_reflects_a_broken_text_layer(
    texts: tuple[str, ...], expected: float
) -> None:
    page = _page()
    segmented = _page(*texts).parsed_page
    page._backend = SimpleNamespace(
        get_segmented_page=lambda: segmented, get_bitmap_rects=lambda: iter(())
    )  # type: ignore[assignment]
    conv_res = SimpleNamespace(confidence=ConfidenceReport())
    model = PagePreprocessingModel(PagePreprocessingOptions(images_scale=None))

    model._parse_page_cells(conv_res, page)  # type: ignore[arg-type]

    assert conv_res.confidence.pages[1].parse_score == expected
    assert page._parse_score == expected


def test_page_below_threshold_is_ocrd_in_full() -> None:
    page = _page(_BROKEN, parse_score=0.0)
    find_rects = Mock(return_value=[])
    model = _ocr_model(
        OcrMode.DEFAULT, True, _find_pdf_aware_layout_ocr_rects=find_rects
    )

    (rect,) = BaseOcrModel.get_ocr_rects(model, page)  # type: ignore[arg-type]

    assert (rect.l, rect.t, rect.r, rect.b) == (0, 0, 600.0, 800.0)
    find_rects.assert_not_called()


@pytest.mark.parametrize(
    ("replace", "parse_score"),
    [
        (True, 1.0),  # good text layer
        (True, math.nan),  # no text layer: left to the usual OCR
        (False, 0.0),  # option off: behaviour unchanged
    ],
)
def test_other_pages_keep_their_ocr_regions(replace: bool, parse_score: float) -> None:
    page = _page(_CLEAN, parse_score=parse_score)
    find_rects = Mock(return_value=[])
    model = _ocr_model(
        OcrMode.DEFAULT, replace, _find_pdf_aware_layout_ocr_rects=find_rects
    )

    assert BaseOcrModel.get_ocr_rects(model, page) == []  # type: ignore[arg-type]
    find_rects.assert_called_once_with(page)


def test_ocr_text_replaces_the_pdf_text_below_threshold() -> None:
    page = _page(_BROKEN, parse_score=0.0)
    ocr_cell = _cell("Proxy Statement", from_ocr=True, y=700)
    merge = Mock()
    model = _ocr_model(OcrMode.DEFAULT, True, _merge_ocr_and_pdf_cells=merge)
    conv_res = SimpleNamespace(confidence=ConfidenceReport())

    BaseOcrModel.post_process_cells(model, [ocr_cell], page, conv_res)  # type: ignore[arg-type]

    assert page.cells == [ocr_cell]
    merge.assert_not_called()


def test_pdf_text_is_kept_above_threshold() -> None:
    page = _page(_CLEAN, parse_score=1.0)
    pdf_cell = page.cells[0]
    ocr_cell = _cell("figure label", from_ocr=True, y=100)
    merge = Mock(return_value=[pdf_cell, ocr_cell])
    model = _ocr_model(OcrMode.DEFAULT, True, _merge_ocr_and_pdf_cells=merge)
    conv_res = SimpleNamespace(confidence=ConfidenceReport())

    BaseOcrModel.post_process_cells(model, [ocr_cell], page, conv_res)  # type: ignore[arg-type]

    assert page.cells == [pdf_cell, ocr_cell]
    merge.assert_called_once()


def test_option_is_off_by_default() -> None:
    assert OcrOptions(lang=[]).replace_broken_text_layer is False
    assert OcrOptions(lang=[], replace_broken_text_layer=True).replace_broken_text_layer
