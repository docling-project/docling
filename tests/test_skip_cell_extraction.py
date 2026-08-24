"""Regression tests for skipping native cell extraction (issue #4058).

In full-page OCR mode the native text cells are discarded during OCR
post-processing, so the segmented-page decode is skipped entirely. A page
processed that way keeps ``parsed_page=None`` until OCR runs; OCR
post-processing and layout post-processing must handle that instead of
crashing.
"""

from types import SimpleNamespace

import pytest
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import (
    BoundingBox,
    Cluster,
    ConfidenceReport,
    LayoutPrediction,
    Page,
    Size,
)
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    LayoutPostprocessorOptions,
    OcrMode,
    PdfPipelineOptions,
)
from docling.models.base_ocr_model import BaseOcrModel, _empty_segmented_page
from docling.models.stages.layout.layout_postprocessing_model import (
    LayoutPostprocessingModel,
)
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    resolve_skip_cell_extraction,
)


def _page() -> Page:
    page = Page(page_no=1)
    page.size = Size(width=600.0, height=800.0)
    return page


def _ocr_cell(text: str, confidence: float) -> TextCell:
    return TextCell(
        rect=BoundingRectangle(
            r_x0=10, r_y0=30, r_x1=110, r_y1=30, r_x2=110, r_y2=10, r_x3=10, r_y3=10
        ),
        text=text,
        orig=text,
        from_ocr=True,
        confidence=confidence,
    )


def test_skip_derived_from_full_page_ocr_mode() -> None:
    # Full-page OCR discards native cells anyway -> the decode is skipped.
    options = PdfPipelineOptions(
        do_ocr=True, ocr_options=EasyOcrOptions(mode=OcrMode.FULL_PAGE)
    )
    assert resolve_skip_cell_extraction(options) is True


def test_no_skip_without_ocr() -> None:
    # Skipping without OCR would silently produce text-free output.
    options = PdfPipelineOptions(
        do_ocr=False, ocr_options=EasyOcrOptions(mode=OcrMode.FULL_PAGE)
    )
    assert resolve_skip_cell_extraction(options) is False


def test_no_skip_in_native_cell_dependent_modes() -> None:
    # The other OCR modes merge OCR output with native cells -> keep the decode.
    for mode in (OcrMode.LAYOUT_REGIONS, OcrMode.PDF_AWARE_LAYOUT_REGIONS):
        options = PdfPipelineOptions(do_ocr=True, ocr_options=EasyOcrOptions(mode=mode))
        assert resolve_skip_cell_extraction(options) is False
    # Defaults (auto OCR mode) keep the decode too.
    assert resolve_skip_cell_extraction(PdfPipelineOptions()) is False


def test_empty_segmented_page_matches_page_geometry() -> None:
    seg = _empty_segmented_page(_page())
    assert seg.dimension.crop_bbox.r == 600.0
    assert seg.dimension.crop_bbox.b == 800.0
    assert seg.char_cells == []
    assert seg.word_cells == []
    assert seg.textline_cells == []


def test_post_process_cells_tolerates_missing_parsed_page() -> None:
    # parsed_page is None when cell extraction was skipped; OCR output must
    # land in a fresh SegmentedPdfPage instead of tripping an assert.
    page = _page()
    assert page.parsed_page is None
    assert page.cells == []

    cell = _ocr_cell("part 42-A", confidence=0.9)
    ocr_model = SimpleNamespace(options=SimpleNamespace(mode=OcrMode.FULL_PAGE))
    conv_res = SimpleNamespace(confidence=ConfidenceReport())

    BaseOcrModel.post_process_cells(ocr_model, [cell], page, conv_res)

    assert page.parsed_page is not None
    assert page.parsed_page.textline_cells == [cell]
    assert page.parsed_page.has_lines
    assert page.parsed_page.word_cells == []
    assert conv_res.confidence.pages[1].ocr_score == pytest.approx(0.9)


def test_layout_write_back_guarded_without_parsed_page() -> None:
    # With cell assignment ENABLED and parsed_page None (skipped extraction),
    # the postprocessor previously asserted; it must now pass through.
    page = _page()
    page._backend = SimpleNamespace(is_valid=lambda: True)  # type: ignore[assignment]
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(
                id=0,
                label=DocItemLabel.TEXT,
                bbox=BoundingBox(l=10, t=10, r=200, b=100),
                confidence=0.8,
            )
        ]
    )

    model = LayoutPostprocessingModel(
        options=LayoutPostprocessorOptions(
            run_postprocessor=True,
            keep_empty_clusters=True,
            skip_cell_assignment=False,
        )
    )
    conv_res = SimpleNamespace(confidence=ConfidenceReport(), timings={})

    out_pages = list(model(conv_res, [page]))

    assert len(out_pages) == 1
    assert out_pages[0].predictions.layout.clusters
    assert page.parsed_page is None  # nothing to write back, and no crash
