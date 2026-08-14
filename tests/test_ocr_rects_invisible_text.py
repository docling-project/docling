"""OCR rect selection must not treat an invisible text layer as programmatic text."""

from collections.abc import Iterable
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.labels import DocItemLabel

from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    ThreadedDoclingParseDocumentBackend,
)
from docling.backend.pdf_backend import PdfPageBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel

FIXTURE = Path("./tests/data/pdf/invisible_text_layer.pdf")

# The two text lines of the fixture, in top-left page coordinates.
VISIBLE_LINE = BoundingBox(l=60, t=70, r=400, b=110, coord_origin=CoordOrigin.TOPLEFT)
INVISIBLE_LINE = BoundingBox(
    l=60, t=470, r=400, b=510, coord_origin=CoordOrigin.TOPLEFT
)


class _OcrRectsOnlyModel(BaseOcrModel):
    """Minimal concrete `BaseOcrModel`: only the rect selection is under test."""

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        raise NotImplementedError

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return OcrOptions


def _make_model() -> _OcrRectsOnlyModel:
    return _OcrRectsOnlyModel(
        enabled=True,
        artifacts_path=None,
        options=OcrOptions(
            kind="test", lang=["en"], mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS
        ),
        accelerator_options=AcceleratorOptions(),
    )


def _make_page(page_backend: PdfPageBackend, cluster_bbox: BoundingBox) -> Page:
    page = Page(page_no=0)
    page._backend = page_backend
    page.size = page_backend.get_size()
    page.predictions.layout = LayoutPrediction(
        clusters=[Cluster(id=0, label=DocItemLabel.TEXT, bbox=cluster_bbox)]
    )
    return page


def _load_first_page(backend_cls):
    doc_backend = InputDocument(
        path_or_stream=FIXTURE,
        format=InputFormat.PDF,
        backend=backend_cls,
    )._backend

    # The threaded backend streams pages and rejects random access.
    if backend_cls is ThreadedDoclingParseDocumentBackend:
        return doc_backend, next(iter(doc_backend.iter_pages()))
    return doc_backend, doc_backend.load_page(0)


@pytest.mark.parametrize(
    "backend_cls",
    [
        # Spatial-index path (no `intersects_with`), the default PDF backend.
        DoclingParseDocumentBackend,
        # Native-query paths.
        ThreadedDoclingParseDocumentBackend,
        PyPdfiumDocumentBackend,
    ],
)
def test_invisible_text_still_needs_ocr(backend_cls):
    """A cluster covered only by invisible text has no readable text: it must be OCR'd."""
    doc_backend, page_backend = _load_first_page(backend_cls)
    model = _make_model()

    try:
        invisible_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, INVISIBLE_LINE)
        )
        assert len(invisible_rects) == 1
        assert invisible_rects[0].intersection_over_self(INVISIBLE_LINE) > 0

        visible_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, VISIBLE_LINE)
        )
        assert visible_rects == []
    finally:
        doc_backend.unload()
