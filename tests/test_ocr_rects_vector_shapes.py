# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OCR rect selection must not OCR programmatic text just because a vector shape crosses it."""

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

# One Helvetica line with a stroked rule 6pt below its baseline, nothing else on the page.
FIXTURE = Path("./tests/data/pdf/text_with_vector_rule.pdf")

# Text line plus the rule under it, in top-left page coordinates (page height 792).
RULED_TEXT = BoundingBox(l=60, t=70, r=380, b=105, coord_origin=CoordOrigin.TOPLEFT)
# A region with neither text nor shapes.
EMPTY_REGION = BoundingBox(l=60, t=400, r=380, b=450, coord_origin=CoordOrigin.TOPLEFT)


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
        # Spatial-index path (no `has_content_in`).
        DoclingParseDocumentBackend,
        # Native-query paths, which can see vector shapes.
        ThreadedDoclingParseDocumentBackend,
        PyPdfiumDocumentBackend,
    ],
)
def test_vector_rule_does_not_force_ocr_of_programmatic_text(backend_cls):
    """A text cluster crossed by a rule keeps its text layer; an empty cluster is OCR'd."""
    doc_backend, page_backend = _load_first_page(backend_cls)
    model = _make_model()

    try:
        # Sanity: the fixture really has programmatic text there.
        assert "Programmatic text" in page_backend.get_text_in_rect(RULED_TEXT)

        ruled_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, RULED_TEXT)
        )
        assert ruled_rects == []

        empty_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, EMPTY_REGION)
        )
        assert len(empty_rects) == 1
        assert empty_rects[0].intersection_over_self(EMPTY_REGION) > 0
    finally:
        doc_backend.unload()
