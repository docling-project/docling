# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Behavioral tests for the standalone LayoutPostprocessingModel stage."""

from types import SimpleNamespace

import numpy as np
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
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.models.base_ocr_model import _empty_segmented_page
from docling.models.stages.layout.layout_postprocessing_model import (
    LayoutPostprocessingModel,
)


class _StubBackend:
    """Minimal stand-in for a PdfPageBackend (only validity is queried here)."""

    def __init__(self, valid: bool = True) -> None:
        self._valid = valid

    def is_valid(self) -> bool:
        return self._valid


def _conv_res() -> SimpleNamespace:
    # A full ConversionResult requires a real InputDocument; the stage only
    # touches `.confidence` (a real ConfidenceReport) and `.timings`.
    return SimpleNamespace(confidence=ConfidenceReport(), timings={})


def _cluster(cid: int, confidence: float, bbox: tuple) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        confidence=confidence,
    )


def _page_with_raw_clusters(clusters: list[Cluster], valid: bool = True) -> Page:
    page = Page(page_no=1)
    page.size = Size(width=600.0, height=800.0)
    page._backend = _StubBackend(valid=valid)  # type: ignore[assignment]
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    return page


def test_pass_through_when_postprocessor_disabled() -> None:
    # TableCrops scenario: clusters already carry cells; the stage must not
    # mutate them and only computes the layout_score.
    raw = LayoutPrediction(clusters=[_cluster(0, 1.0, (0, 0, 600, 800))])
    page = Page(page_no=1)
    page.size = Size(width=600.0, height=800.0)
    page._backend = _StubBackend()  # type: ignore[assignment]
    page.predictions.layout = raw

    model = LayoutPostprocessingModel(
        options=LayoutPostprocessorOptions(run_postprocessor=False)
    )
    conv_res = _conv_res()

    out_pages = list(model(conv_res, [page]))

    assert len(out_pages) == 1
    # Raw prediction object is preserved untouched.
    assert out_pages[0].predictions.layout is raw
    assert conv_res.confidence.pages[1].layout_score == pytest.approx(1.0)


def test_runs_postprocessor_and_scores_processed_clusters() -> None:
    # Two well-separated text clusters; with keep_empty_clusters they survive
    # post-processing and the score is their mean confidence. Cell assignment
    # is skipped so the test does not need a parsed_page.
    clusters = [
        _cluster(0, 0.6, (10, 10, 200, 100)),
        _cluster(1, 0.8, (10, 400, 200, 500)),
    ]
    page = _page_with_raw_clusters(clusters)

    model = LayoutPostprocessingModel(
        options=LayoutPostprocessorOptions(
            run_postprocessor=True,
            keep_empty_clusters=True,
            skip_cell_assignment=True,
        )
    )
    conv_res = _conv_res()

    out_pages = list(model(conv_res, [page]))

    assert len(out_pages) == 1
    produced = out_pages[0].predictions.layout.clusters
    assert produced  # postprocessor kept the clusters
    # The score is computed over exactly the clusters the stage emits.
    expected = float(np.mean([c.confidence for c in produced]))
    assert conv_res.confidence.pages[1].layout_score == pytest.approx(expected)
    assert conv_res.confidence.pages[1].layout_score == pytest.approx(0.7)


def test_orphan_text_confidence_does_not_inflate_layout_score() -> None:
    # The layout model finds a small footer, but its OCR cell is wider than
    # the overlap threshold allows. Post-processing drops the model cluster
    # and creates an orphan TEXT cluster with OCR confidence instead.
    page = _page_with_raw_clusters(
        [
            Cluster(
                id=0,
                label=DocItemLabel.PAGE_FOOTER,
                bbox=BoundingBox(l=295, t=803, r=301, b=813),
                confidence=0.76,
            )
        ]
    )
    page.size = Size(width=595, height=842)
    page.parsed_page = _empty_segmented_page(page)
    page.parsed_page.textline_cells = [
        TextCell(
            index=0,
            text="4",
            orig="4",
            from_ocr=True,
            confidence=1.0,
            rect=BoundingRectangle.from_bounding_box(
                BoundingBox(l=289, t=798, r=311, b=817)
            ),
        )
    ]

    conv_res = _conv_res()
    [prediction] = LayoutPostprocessingModel(
        options=LayoutPostprocessorOptions()
    ).postprocess_layout(conv_res, [page])

    assert len(prediction.clusters) == 1
    assert prediction.clusters[0].label == DocItemLabel.TEXT
    assert prediction.clusters[0].confidence == pytest.approx(1.0)
    # No model cluster survived, so OCR confidence must not produce an
    # excellent layout score.
    assert conv_res.confidence.pages[1].layout_score == pytest.approx(0.0)


def test_invalid_page_passes_through_without_scoring() -> None:
    page = _page_with_raw_clusters([_cluster(0, 0.9, (0, 0, 100, 100))], valid=False)
    model = LayoutPostprocessingModel(
        options=LayoutPostprocessorOptions(run_postprocessor=True)
    )
    conv_res = _conv_res()

    out_pages = list(model(conv_res, [page]))

    assert len(out_pages) == 1
    # No score written for invalid pages -> stays at the NaN default.
    assert np.isnan(conv_res.confidence.pages[1].layout_score)
