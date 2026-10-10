# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OCR rect deduplication must not hand the same page area to the OCR engine twice."""

from collections.abc import Iterable

from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel

TL = CoordOrigin.TOPLEFT

# Three clusters that frame a fourth one on three sides (a "C" around a block), on a
# 300x300pt page. The frame and the block are not connected, but the box that
# encloses the frame also encloses the block.
FRAME = [
    BoundingBox(l=20, t=20, r=280, b=40, coord_origin=TL),
    BoundingBox(l=20, t=20, r=40, b=280, coord_origin=TL),
    BoundingBox(l=20, t=260, r=280, b=280, coord_origin=TL),
]
ENCLOSED_BLOCK = BoundingBox(l=80, t=80, r=240, b=220, coord_origin=TL)


class _OcrRectsOnlyModel(BaseOcrModel):
    """Minimal concrete `BaseOcrModel`: only the rect selection is under test."""

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        raise NotImplementedError

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return OcrOptions


def _ocr_rects(clusters: list[BoundingBox]) -> list[BoundingBox]:
    model = _OcrRectsOnlyModel(
        enabled=True,
        artifacts_path=None,
        options=OcrOptions(kind="test", lang=["en"], mode=OcrMode.LAYOUT_REGIONS),
        accelerator_options=AcceleratorOptions(),
    )
    page = Page(page_no=0)
    page.size = Size(width=300, height=300)
    page.predictions.layout = LayoutPrediction(
        clusters=[
            Cluster(id=i, label=DocItemLabel.TEXT, bbox=bbox)
            for i, bbox in enumerate(clusters)
        ]
    )
    return model.get_ocr_rects(page)


def test_enclosed_cluster_is_not_ocred_twice():
    rects = _ocr_rects([*FRAME, ENCLOSED_BLOCK])

    assert len(rects) == 1
    (outer,) = rects
    assert outer.l <= ENCLOSED_BLOCK.l and outer.t <= ENCLOSED_BLOCK.t
    assert outer.r >= ENCLOSED_BLOCK.r and outer.b >= ENCLOSED_BLOCK.b


def test_partially_overlapping_rects_are_kept():
    # Two L-shaped frames whose enclosing boxes overlap without either one containing
    # the other: dropping one would leave part of the page without OCR.
    top_left = [
        BoundingBox(l=20, t=20, r=200, b=40, coord_origin=TL),
        BoundingBox(l=20, t=20, r=40, b=200, coord_origin=TL),
    ]
    bottom_right = [
        BoundingBox(l=100, t=260, r=280, b=280, coord_origin=TL),
        BoundingBox(l=260, t=100, r=280, b=280, coord_origin=TL),
    ]

    rects = _ocr_rects([*top_left, *bottom_right])

    assert len(rects) == 2
