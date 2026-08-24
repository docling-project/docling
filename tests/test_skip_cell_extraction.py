"""Regression tests for skip_cell_extraction (issue #4058).

A page whose native cell extraction was skipped keeps ``parsed_page=None``;
OCR post-processing and layout post-processing must handle that instead of
crashing, and the flag must be reachable from ``PdfPipelineOptions``.
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
    LayoutPostprocessorOptions,
    OcrMode,
    PdfPipelineOptions,
)
from docling.models.base_ocr_model import BaseOcrModel, _empty_segmented_page
from docling.models.stages.layout.layout_postprocessing_model import (
    LayoutPostprocessingModel,
)
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
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


def test_pdf_pipeline_options_expose_skip_cell_extraction() -> None:
    assert PdfPipelineOptions().skip_cell_extraction is False
    assert PdfPipelineOptions(skip_cell_extraction=True).skip_cell_extraction is True
    # The preprocessing stage accepts the plumbed value.
    options = PagePreprocessingOptions(images_scale=None, skip_cell_extraction=True)
    assert PagePreprocessingModel(options=options).options.skip_cell_extraction


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
