# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OCR rect selection must not OCR programmatic text just because a vector shape crosses it."""

from collections.abc import Iterable
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.labels import DocItemLabel

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
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

# A 300x300pt page with Helvetica 18 "Native" at (40,245) and a 24pt-high "HI" built from
# filled rectangles at (150,245) -- drawing coordinates in the PDF bottom-left origin --
# the same glyphs alone at y=145, and a filled highlight behind native text at y=45.
# No bitmaps, no stroked lines.
GLYPH_FIXTURE = Path("./tests/data/pdf/text_with_vector_glyphs.pdf")

# One cluster holding both the native text and the vector-outlined glyphs (page height 300).
MIXED_TEXT_AND_GLYPHS = BoundingBox(
    l=30, t=25, r=220, b=75, coord_origin=CoordOrigin.TOPLEFT
)
# The same glyphs with no programmatic text beside them.
VECTOR_GLYPHS_ONLY = BoundingBox(
    l=140, t=125, r=200, b=175, coord_origin=CoordOrigin.TOPLEFT
)
# A filled highlight with native text sitting on it: a shape the text layer explains.
TEXT_BACKED_SHAPE = BoundingBox(
    l=30, t=225, r=180, b=260, coord_origin=CoordOrigin.TOPLEFT
)

# Two constructions a backend's stroked-segment report alone would misread, each
# sharing its cluster with native text (page height 300).
EDGE_FIXTURE = Path("./tests/data/pdf/vector_glyph_edge_cases.pdf")

# Letterforms drawn as stroked axis-aligned segments, reported exactly as a rule is.
STROKED_GLYPHS = BoundingBox(l=30, t=70, r=220, b=105, coord_origin=CoordOrigin.TOPLEFT)
# A rule crossing filled letterforms, merged with them into one connected shape.
RULE_THROUGH_GLYPHS = BoundingBox(
    l=30, t=170, r=220, b=205, coord_origin=CoordOrigin.TOPLEFT
)
# A rule drawn as a filled rectangle, which no stroked-segment report mentions.
FILLED_RULE = BoundingBox(l=30, t=235, r=280, b=270, coord_origin=CoordOrigin.TOPLEFT)

# Two constructions where a connected shape reaches well beyond the cluster it
# touches, so what the text layer accounts for must be judged inside the cluster
# (page height 300).
MERGED_FILL_FIXTURE = Path("./tests/data/pdf/vector_glyphs_merged_fill.pdf")

# Native text beside filled letterforms that connected-shape merging fuses with a
# large panel below them. The panel carries native text far outside the cluster.
GLYPHS_FUSED_WITH_PANEL = BoundingBox(
    l=30, t=25, r=220, b=75, coord_origin=CoordOrigin.TOPLEFT
)
# Native text whose cluster is crossed, by 2pt, by the top of a text-bearing panel.
PANEL_EDGE_IN_CLUSTER = BoundingBox(
    l=30, t=125, r=130, b=175, coord_origin=CoordOrigin.TOPLEFT
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


def _load_first_page(backend_cls, fixture: Path = FIXTURE):
    doc_backend = InputDocument(
        path_or_stream=fixture,
        format=InputFormat.PDF,
        backend=backend_cls,
    )._backend

    # The threaded backend streams pages and rejects random access.
    if backend_cls is ThreadedDoclingParseDocumentBackend:
        return doc_backend, next(iter(doc_backend.iter_pages()))
    return doc_backend, doc_backend.load_page(0)


@pytest.mark.parametrize(
    ("backend_cls", "native_queries"),
    [
        # Native-query paths, which can see vector shapes.
        (ThreadedDoclingParseDocumentBackend, True),
        (PyPdfiumDocumentBackend, True),
        # Spatial-index path, which indexes connected shapes when the backend has them.
        (ThreadedDoclingParseDocumentBackend, False),
    ],
    ids=["threaded", "pypdfium2", "threaded-spatial-index"],
)
def test_vector_rule_does_not_force_ocr_of_programmatic_text(
    backend_cls, native_queries, monkeypatch
):
    """A text cluster crossed by a rule keeps its text layer; an empty cluster is OCR'd."""
    doc_backend, page_backend = _load_first_page(backend_cls)
    model = _make_model()
    if not native_queries:
        # No backend lacks `has_content_in` any more; force the fallback path.
        monkeypatch.setattr(page_backend, "has_content_in", lambda **kwargs: None)

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


@pytest.mark.parametrize(
    ("backend_cls", "native_queries"),
    [
        (ThreadedDoclingParseDocumentBackend, True),
        (PyPdfiumDocumentBackend, True),
        (ThreadedDoclingParseDocumentBackend, False),
    ],
    ids=["threaded", "pypdfium2", "threaded-spatial-index"],
)
def test_vector_outlined_glyphs_still_force_ocr(
    backend_cls, native_queries, monkeypatch
):
    """Vector-outlined glyphs need OCR even when native text shares their cluster.

    Regression for the content loss reported on #4209: gating on the mere *presence*
    of programmatic text drops a cluster whose text layer only covers part of it, so
    glyphs drawn as filled paths are never recognised.
    """
    doc_backend, page_backend = _load_first_page(backend_cls, GLYPH_FIXTURE)
    model = _make_model()
    if not native_queries:
        # No backend lacks `has_content_in` any more; force the fallback path.
        monkeypatch.setattr(page_backend, "has_content_in", lambda **kwargs: None)

    try:
        # Sanity: the text layer carries "Native" and nothing of the vector "HI".
        assert "Native" in page_backend.get_text_in_rect(MIXED_TEXT_AND_GLYPHS)
        assert page_backend.get_text_in_rect(VECTOR_GLYPHS_ONLY).strip() == ""

        mixed_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, MIXED_TEXT_AND_GLYPHS)
        )
        assert len(mixed_rects) == 1
        assert mixed_rects[0].intersection_over_self(MIXED_TEXT_AND_GLYPHS) > 0

        vector_only_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, VECTOR_GLYPHS_ONLY)
        )
        assert len(vector_only_rects) == 1

        # A shape the text layer does explain stays out of OCR, like a rule does.
        backed_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, TEXT_BACKED_SHAPE)
        )
        assert backed_rects == []
    finally:
        doc_backend.unload()


@pytest.mark.parametrize(
    ("backend_cls", "native_queries"),
    [
        (ThreadedDoclingParseDocumentBackend, True),
        (PyPdfiumDocumentBackend, True),
        (ThreadedDoclingParseDocumentBackend, False),
    ],
    ids=["threaded", "pypdfium2", "threaded-spatial-index"],
)
@pytest.mark.parametrize(
    "region",
    [STROKED_GLYPHS, RULE_THROUGH_GLYPHS],
    ids=["stroked-letterforms", "rule-through-glyphs"],
)
def test_shape_extent_outranks_the_stroked_segment_report(
    region, backend_cls, native_queries, monkeypatch
):
    """A stroked-segment report alone must not clear a cluster of outlined glyphs.

    Letterforms drawn as stroked axis-aligned segments are reported just as a rule
    is, and a rule crossing a glyph merges with it into a single connected shape
    that overlaps a reported segment. Both still need OCR, so the shape's extent
    has to agree before it counts as a rule.
    """
    doc_backend, page_backend = _load_first_page(backend_cls, EDGE_FIXTURE)
    model = _make_model()
    if not native_queries:
        # No backend lacks `has_content_in` any more; force the fallback path.
        monkeypatch.setattr(page_backend, "has_content_in", lambda **kwargs: None)

    try:
        # Sanity: the cluster really does carry programmatic text as well.
        assert "Native" in page_backend.get_text_in_rect(region)

        rects = model._find_pdf_aware_layout_ocr_rects(_make_page(page_backend, region))
        assert len(rects) == 1
        assert rects[0].intersection_over_self(region) > 0
    finally:
        doc_backend.unload()


@pytest.mark.parametrize(
    ("backend_cls", "native_queries"),
    [
        (ThreadedDoclingParseDocumentBackend, True),
        (PyPdfiumDocumentBackend, True),
        (ThreadedDoclingParseDocumentBackend, False),
    ],
    ids=["threaded", "pypdfium2", "threaded-spatial-index"],
)
def test_a_rule_drawn_as_a_filled_rectangle_does_not_force_ocr(
    backend_cls, native_queries, monkeypatch
):
    """Plenty of producers draw rules as filled rectangles rather than strokes.

    `get_shape_lines` reports only stroked segments and so never mentions these,
    which is why the shape's extent, not that report, has to decide what a rule is.
    """
    doc_backend, page_backend = _load_first_page(backend_cls, EDGE_FIXTURE)
    model = _make_model()
    if not native_queries:
        # No backend lacks `has_content_in` any more; force the fallback path.
        monkeypatch.setattr(page_backend, "has_content_in", lambda **kwargs: None)

    try:
        # Clipped at the bbox edge by some backends, so match the start of the line.
        assert "Programmatic" in page_backend.get_text_in_rect(FILLED_RULE)
        rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, FILLED_RULE)
        )
        assert rects == []
    finally:
        doc_backend.unload()


@pytest.mark.parametrize(
    ("backend_cls", "native_queries"),
    [
        (ThreadedDoclingParseDocumentBackend, True),
        (PyPdfiumDocumentBackend, True),
        (ThreadedDoclingParseDocumentBackend, False),
    ],
    ids=["threaded", "pypdfium2", "threaded-spatial-index"],
)
def test_text_behind_a_shape_is_judged_inside_the_cluster(
    backend_cls, native_queries, monkeypatch
):
    """Only the part of a shape inside the cluster can be accounted for by text.

    A connected shape may extend far past the cluster: outlined glyphs fused with a
    page-wide panel, say. Text under the panel elsewhere on the page says nothing
    about the glyphs, so probing the whole shape would wrongly clear the cluster.
    Conversely, a text-bearing panel whose edge just crosses a cluster leaves only
    a sliver inside it, which must not force OCR any more than a rule does.
    """
    doc_backend, page_backend = _load_first_page(backend_cls, MERGED_FILL_FIXTURE)
    model = _make_model()
    if not native_queries:
        # No backend lacks `has_content_in` any more; force the fallback path.
        monkeypatch.setattr(page_backend, "has_content_in", lambda **kwargs: None)

    try:
        # Sanity: the fused shape reaches past the cluster, and the panel's text
        # lies outside it.
        shapes = page_backend.get_connected_shape_bounding_boxes() or []
        fused = [s for s in shapes if model._boxes_touch(s, GLYPHS_FUSED_WITH_PANEL)]
        assert len(fused) == 1 and fused[0].b > GLYPHS_FUSED_WITH_PANEL.b
        assert "Panel text" in page_backend.get_text_in_rect(fused[0])
        assert "Panel text" not in page_backend.get_text_in_rect(
            GLYPHS_FUSED_WITH_PANEL
        )

        fused_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, GLYPHS_FUSED_WITH_PANEL)
        )
        assert len(fused_rects) == 1
        assert fused_rects[0].intersection_over_self(GLYPHS_FUSED_WITH_PANEL) > 0

        edge_rects = model._find_pdf_aware_layout_ocr_rects(
            _make_page(page_backend, PANEL_EDGE_IN_CLUSTER)
        )
        assert edge_rects == []
    finally:
        doc_backend.unload()
