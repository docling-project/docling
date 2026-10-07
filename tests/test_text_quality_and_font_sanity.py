# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from typing import Optional

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, SegmentedPdfPage, TextCell
from PIL import Image

from docling.backend.pdf_backend import PdfPageBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.pipeline_options import EasyOcrOptions, OcrMode
from docling.models.base_ocr_model import BaseOcrModel, _MergeCellsPriority
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.utils.text_quality import rate_text_quality


def _make_text_cell(
    text: str,
    left: float,
    bottom: float,
    right: float,
    top: float,
    from_ocr: bool = False,
) -> TextCell:
    rect = BoundingRectangle(
        r_x0=left,
        r_y0=bottom,
        r_x1=right,
        r_y1=bottom,
        r_x2=right,
        r_y2=top,
        r_x3=left,
        r_y3=top,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )
    return TextCell(
        index=0,
        text=text,
        orig=text,
        rect=rect,
        from_ocr=from_ocr,
        confidence=0.99 if from_ocr else 1.0,
    )


@pytest.mark.parametrize(
    "valid_text",
    [
        "",  # Empty string
        "The quick brown fox jumps over the lazy dog.",  # Standard English
        "L'extraction d'information n'est pas facile pour les documents complexes.",  # French with apostrophes
        "Die Überprüfung der Jahresabrechnung für Großprojekte ist abgeschlossen.",  # German umlauts
        "Học viện Công nghệ Bưu chính Viễn thông thông báo lịch thi.",  # Vietnamese diacritics
        "अनुक्रमणिका: महाराष्ट्र शासन निर्णय क्रमांक २०२४",  # Devanagari Hindi/Marathi
        "Revenue grew by 25.5% to $1.2B in Q3 ($1,200.50 per share).",  # Financial & numbers
        "Chapter 1: An in-depth overview (see Table 2.1 & Section 4).",  # Natural punctuation
    ],
)
def test_rate_text_quality_valid_multilingual(valid_text: str):
    score = rate_text_quality(valid_text)
    assert score == 1.0


@pytest.mark.parametrize(
    "corrupt_text",
    [
        "Corrupted document with \ufffd replacement character",  # Unicode replacement char
        "Section \ue001 Title with PUA character",  # Private Use Area BMP
        "Supplementary \U000f0000 PUA Plane 15",  # PUA Plane 15
        "Text with unprintable \x03 control byte",  # Control character
        "Text with \x1b ANSI escape code",  # Control character
        "Document segment GLYPH<0A12F> broken font",  # Glyph tag
        "/G102/G304/G506 broken encoding",  # Slash-G font token sequence
        "/token1 /token2 /token3 garbage",  # Slash number pattern
        "d%b&c *f(g+ h^j~k mojibake",  # 8-bit font remapping mojibake
    ],
)
def test_rate_text_quality_corrupted_patterns(corrupt_text: str):
    score = rate_text_quality(corrupt_text)
    assert score == 0.0


def test_rate_text_quality_fragmented_words_penalty():
    # Fragmented pattern e.g. a/bc.de/fg.hi (repeated pattern)
    frag_text = "a/bc.de/fg.hi j/kl.mn/op.qr s/tu.vw/xy.za"
    score = rate_text_quality(frag_text)
    assert 0.0 <= score < 1.0


def test_page_preprocessing_model_delegates_to_rate_text_quality():
    model = PagePreprocessingModel(options=PagePreprocessingOptions(images_scale=None))
    assert model.rate_text_quality("Clean document") == 1.0
    assert model.rate_text_quality("Corrupt \ufffd document") == 0.0


class DummyOcrModel(BaseOcrModel):
    @classmethod
    def get_options_type(cls):
        return EasyOcrOptions

    def __call__(self, conv_res, page_batch):
        return page_batch


class DummyPageBackend(PdfPageBackend):
    def __init__(self, text_in_rect: str = ""):
        self._text_in_rect = text_in_rect

    @property
    def page_no(self) -> int:
        return 0

    def is_valid(self) -> bool:
        return True

    def get_size(self) -> Size:
        return Size(width=612, height=792)

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        return Image.new("RGB", (100, 100))

    def unload(self) -> None:
        pass

    def has_content_in(
        self,
        *,
        bbox: BoundingBox,
        chars: bool = False,
        shapes: bool = True,
        bitmaps: bool = True,
    ) -> bool:
        if chars:
            return True
        return False

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        return self._text_in_rect

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        return None

    def get_text_cells(self) -> Iterable[TextCell]:
        return []

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        return []


def test_merge_ocr_and_pdf_cells_prioritizes_clean_pdf_and_ocr_over_corrupt_pdf():
    model = DummyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
        accelerator_options=AcceleratorOptions(),
    )

    clean_pdf = _make_text_cell("Annual Report 2024", 0, 70, 100, 90, from_ocr=False)
    corrupt_pdf = _make_text_cell("d%b&c *f(g+ h^j~k", 0, 10, 100, 30, from_ocr=False)
    ocr_cell_clean = _make_text_cell("Financial Summary", 0, 10, 100, 30, from_ocr=True)

    merged = model._merge_ocr_and_pdf_cells(
        ocr_cells=[ocr_cell_clean],
        pdf_cells=[clean_pdf, corrupt_pdf],
        priority=_MergeCellsPriority.PDF_FIRST,
    )

    merged_texts = [c.text for c in merged]
    # Clean PDF text must be preserved
    assert "Annual Report 2024" in merged_texts
    # OCR text must replace corrupt PDF text
    assert "Financial Summary" in merged_texts
    # Corrupt PDF text must not be present
    assert "d%b&c *f(g+ h^j~k" not in merged_texts


def test_merge_ocr_and_pdf_cells_fallback_keeps_non_overlapping_corrupt_pdf():
    model = DummyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
        accelerator_options=AcceleratorOptions(),
    )

    corrupt_pdf = _make_text_cell("d%b&c *f(g+ h^j~k", 0, 10, 100, 30, from_ocr=False)
    # OCR produced no cells for this region
    merged = model._merge_ocr_and_pdf_cells(
        ocr_cells=[],
        pdf_cells=[corrupt_pdf],
        priority=_MergeCellsPriority.PDF_FIRST,
    )
    # Non-overlapping corrupt cell kept as fallback
    assert len(merged) == 1
    assert merged[0].text == "d%b&c *f(g+ h^j~k"


def test_find_pdf_aware_layout_ocr_rects_includes_corrupt_clusters():
    model = DummyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
        accelerator_options=AcceleratorOptions(),
    )

    cluster_bbox = BoundingBox(
        l=10, t=10, r=200, b=100, coord_origin=CoordOrigin.TOPLEFT
    )
    cluster = Cluster(
        id=0,
        label=DocItemLabel.TEXT,
        bbox=cluster_bbox,
        confidence=1.0,
        cells=[],
    )

    page = Page(page_no=0, size=Size(width=612, height=792))
    page._backend = DummyPageBackend(text_in_rect="\ue001 \ue002 corrupt PUA cluster")
    page.predictions.layout = LayoutPrediction(clusters=[cluster])

    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)
    # The cluster with corrupt text must be routed for OCR
    assert len(ocr_rects) == 1
    assert ocr_rects[0].l <= cluster_bbox.l
    assert ocr_rects[0].r >= cluster_bbox.r


def test_find_pdf_aware_layout_ocr_rects_skips_clean_clusters_and_code():
    model = DummyOcrModel(
        enabled=True,
        artifacts_path=None,
        options=EasyOcrOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
        accelerator_options=AcceleratorOptions(),
    )

    clean_cluster = Cluster(
        id=0,
        label=DocItemLabel.TEXT,
        bbox=BoundingBox(l=10, t=10, r=200, b=100, coord_origin=CoordOrigin.TOPLEFT),
        confidence=1.0,
        cells=[],
    )
    code_cluster = Cluster(
        id=1,
        label=DocItemLabel.CODE,
        bbox=BoundingBox(l=10, t=120, r=200, b=200, coord_origin=CoordOrigin.TOPLEFT),
        confidence=1.0,
        cells=[],
    )

    page = Page(page_no=0, size=Size(width=612, height=792))
    page._backend = DummyPageBackend(text_in_rect="Clean English paragraph")
    page.predictions.layout = LayoutPrediction(clusters=[clean_cluster, code_cluster])

    ocr_rects = model._find_pdf_aware_layout_ocr_rects(page)
    # Clean text and code clusters should not generate OCR rects
    assert len(ocr_rects) == 0
