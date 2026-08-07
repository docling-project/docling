from collections.abc import Iterable
from typing import Type

from docling_core.types.doc import CoordOrigin, DocItemLabel

from docling.datamodel.base_models import (
    BoundingBox,
    Cluster,
    LayoutPrediction,
    Page,
    Size,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel


class _DummyOcrModel(BaseOcrModel):
    """Minimal concrete subclass, only to satisfy ABC instantiation."""

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        yield from page_batch

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return OcrOptions


class _FakeRect:
    def __init__(self, bbox: BoundingBox) -> None:
        self._bbox = bbox

    def to_bounding_box(self) -> BoundingBox:
        return self._bbox


class _FakeTextCell:
    def __init__(self, bbox: BoundingBox) -> None:
        self.rect = _FakeRect(bbox)
        self.text = "x"


class _FakeBackend:
    def __init__(
        self, text_cells: list[_FakeTextCell], bitmap_rects: list[BoundingBox]
    ) -> None:
        self._text_cells = text_cells
        self._bitmap_rects = bitmap_rects

    def is_valid(self) -> bool:
        return True

    def get_text_cells(self) -> list[_FakeTextCell]:
        return self._text_cells

    def get_bitmap_rects(self) -> list[BoundingBox]:
        return self._bitmap_rects


def _cluster(cid: int, bbox: tuple, confidence: float = 0.8) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(
            l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT
        ),
        confidence=confidence,
    )


def _bbox(coords: tuple) -> BoundingBox:
    left, top, right, bottom = coords
    return BoundingBox(
        l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT
    )


def _make_model() -> _DummyOcrModel:
    model = object.__new__(_DummyOcrModel)
    # Bypass `_deduplicate_rects`'s dilation/rasterization: this test targets the
    # per-cluster inclusion decision in `_find_pdf_aware_layout_ocr_rects`, not the
    # separate blob-merging step, so an identity passthrough keeps the returned
    # rects directly comparable to the input clusters.
    model._deduplicate_rects = lambda size, rects, dilation_size=0: (0.0, list(rects))  # type: ignore[method-assign]
    return model


def _make_page(clusters: list[Cluster], backend: _FakeBackend) -> Page:
    page = Page(page_no=1)
    page.size = Size(width=600, height=800)
    page.predictions.layout = LayoutPrediction(clusters=clusters)
    page._backend = backend  # type: ignore[assignment]
    return page


def test_cluster_mostly_covered_by_native_text_is_excluded() -> None:
    # A cluster whose native PDF text spans (almost) its whole area needs no OCR.
    cluster = _cluster(1, (10, 10, 190, 30))
    text_cell = _FakeTextCell(_bbox((10, 10, 190, 30)))
    backend = _FakeBackend(text_cells=[text_cell], bitmap_rects=[])
    page = _make_page([cluster], backend)

    model = _make_model()
    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)

    assert ocr_rects == []


def test_cluster_partially_covered_by_native_text_is_still_ocrd() -> None:
    # Regression test for the "silent complete drop" / "missing mid-paragraph
    # clause" bugs: a list item whose bullet is native PDF text but whose
    # remaining label is a separately rasterized/stamped image (no native text of
    # its own) must still be sent to OCR for the uncovered portion, instead of
    # being excluded wholesale just because *some* native text overlaps it.
    cluster = _cluster(1, (10, 60, 190, 80))
    bullet_cell = _FakeTextCell(_bbox((10, 60, 30, 80)))  # covers ~11% of the area
    backend = _FakeBackend(text_cells=[bullet_cell], bitmap_rects=[])
    page = _make_page([cluster], backend)

    model = _make_model()
    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)

    assert ocr_rects == [cluster.bbox]


def test_cluster_without_any_native_text_is_ocrd() -> None:
    cluster = _cluster(1, (10, 110, 190, 130))
    backend = _FakeBackend(text_cells=[], bitmap_rects=[])
    page = _make_page([cluster], backend)

    model = _make_model()
    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)

    assert ocr_rects == [cluster.bbox]


def test_cluster_overlapping_bitmap_is_ocrd_even_with_full_text_coverage() -> None:
    cluster = _cluster(1, (10, 160, 190, 180))
    text_cell = _FakeTextCell(_bbox((10, 160, 190, 180)))
    bitmap = _bbox((10, 160, 190, 180))
    backend = _FakeBackend(text_cells=[text_cell], bitmap_rects=[bitmap])
    page = _make_page([cluster], backend)

    model = _make_model()
    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)

    assert ocr_rects == [cluster.bbox]
