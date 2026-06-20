from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from docling_core.types.doc import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions
from docling.models.base_ocr_model import BaseOcrModel

if TYPE_CHECKING:
    from docling.backend.pdf_backend import PdfPageBackend


class _BitmapBackend:
    def __init__(self, bitmap_rects: list[BoundingBox]):
        self.bitmap_rects = bitmap_rects

    def get_bitmap_rects(self) -> list[BoundingBox]:
        return self.bitmap_rects


class _OcrModel(BaseOcrModel):
    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return OcrOptions

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        return page_batch


def _cell(text: str, *, index: int) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle(
            r_x0=0,
            r_y0=index * 10,
            r_x1=100,
            r_y1=index * 10,
            r_x2=100,
            r_y2=index * 10 + 5,
            r_x3=0,
            r_y3=index * 10 + 5,
        ),
        text=text,
        orig=text,
        from_ocr=False,
    )


def _page_geometry() -> PdfPageGeometry:
    bbox = BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)
    return PdfPageGeometry(
        angle=0,
        rect=BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=100,
            r_y1=0,
            r_x2=100,
            r_y2=100,
            r_x3=0,
            r_y3=100,
        ),
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        art_bbox=bbox,
        bleed_bbox=bbox,
        crop_bbox=bbox,
        media_bbox=bbox,
        trim_bbox=bbox,
    )


def _page(texts: list[str]) -> Page:
    page = Page(page_no=1, size=Size(width=100, height=100))
    page.parsed_page = SegmentedPdfPage(
        dimension=_page_geometry(),
        char_cells=[],
        word_cells=[],
        textline_cells=[_cell(text, index=i) for i, text in enumerate(texts)],
    )
    page._backend = cast(
        "PdfPageBackend",
        _BitmapBackend(
            [BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)]
        ),
    )
    return page


def _model(*, skip_text_layer_pages: bool, force_full_page_ocr: bool = False):
    return _OcrModel(
        enabled=True,
        artifacts_path=None,
        options=OcrOptions(
            lang=[],
            force_full_page_ocr=force_full_page_ocr,
            skip_text_layer_pages=skip_text_layer_pages,
        ),
        accelerator_options=AcceleratorOptions(),
    )


def test_text_layer_skip_is_opt_in():
    page = _page(
        ["This page has enough native text.", "More text.", "Still more text."]
    )

    assert _model(skip_text_layer_pages=False).get_ocr_rects(page) != []


def test_text_layer_skip_drops_ocr_rects_for_text_rich_page():
    page = _page(
        [
            "This page has a usable text layer with enough text to avoid OCR.",
            "Docling should keep the programmatic text instead of OCRing it.",
            "The bitmap resources on this page do not require OCR.",
        ]
    )

    assert _model(skip_text_layer_pages=True).get_ocr_rects(page) == []


def test_text_layer_skip_keeps_ocr_rects_for_sparse_text_page():
    page = _page(["Footer"])

    assert _model(skip_text_layer_pages=True).get_ocr_rects(page) != []


def test_force_full_page_ocr_overrides_text_layer_skip():
    page = _page(
        [
            "This page has a usable text layer with enough text to avoid OCR.",
            "Docling should keep the programmatic text instead of OCRing it.",
            "The bitmap resources on this page do not require OCR.",
        ]
    )

    assert (
        _model(skip_text_layer_pages=True, force_full_page_ocr=True).get_ocr_rects(page)
        != []
    )
