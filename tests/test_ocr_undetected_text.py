# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Scanned text outside layout detections must still reach OCR."""

import ctypes
from collections.abc import Iterable
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from docling_core.types.doc import BoundingBox, DocItemLabel
from PIL import Image, ImageDraw

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, InputFormat, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import OcrMode, OcrOptions
from docling.models.base_ocr_model import BaseOcrModel


class _OcrRectsOnlyModel(BaseOcrModel):
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        raise NotImplementedError

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        return OcrOptions


@pytest.mark.parametrize(
    "backend_cls",
    [PyPdfiumDocumentBackend, ThreadedDoclingParseDocumentBackend],
    ids=["pypdfium2", "threaded"],
)
@pytest.mark.parametrize(
    "detected", [True, False], ids=["partial-layout", "empty-layout"]
)
@pytest.mark.parametrize("mode", [OcrMode.DEFAULT, OcrMode.PDF_AWARE_LAYOUT_REGIONS])
def test_undetected_scanned_line_reaches_ocr(
    backend_cls, detected, mode, tmp_path: Path
) -> None:
    # A raster-only page with an intro, an isolated UPDATE line and a heading.
    # The detector misses the UPDATE line, or returns no regions at all.
    image = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Introductory paragraph", fill="black")
    draw.text((20, 35), "Its second line ends here.", fill="black")
    draw.text((20, 130), "UPDATE: Nedim testing the sync!!!", fill="black")
    draw.text((20, 230), "Section 1: Access Control Requirements", fill="black")
    source = tmp_path / "isolated_scanned_line.pdf"
    image.save(source, resolution=72)
    doc_backend = InputDocument(
        path_or_stream=source,
        format=InputFormat.PDF,
        backend=backend_cls,
    )._backend
    try:
        backend = (
            next(iter(doc_backend.iter_pages()))
            if backend_cls is ThreadedDoclingParseDocumentBackend
            else doc_backend.load_page(0)
        )
        page = Page(page_no=1, size=backend.get_size())
        page._backend = backend
        assert list(backend.get_text_cells()) == []
        page.predictions.layout = LayoutPrediction(
            clusters=[
                Cluster(
                    id=0,
                    label=DocItemLabel.TEXT,
                    bbox=BoundingBox(l=15, t=15, r=280, b=50),
                    confidence=0.9,
                )
            ]
            if detected
            else []
        )
        model = _OcrRectsOnlyModel(
            enabled=True,
            artifacts_path=None,
            options=OcrOptions(lang=["en"], mode=mode),
            accelerator_options=AcceleratorOptions(),
        )

        rects = model.get_ocr_rects(page)

        missing_line = BoundingBox(l=20, t=130, r=155, b=140)
        assert any(
            missing_line.intersection_over_self(rect) > 0.99 for rect in rects
        ), "OCR input omits the isolated scanned line outside layout detections"
        assert len(rects) == 1  # The full-page bitmap is processed once.

        # Explicit layout-only mode retains its contract.
        model.options.mode = OcrMode.LAYOUT_REGIONS
        layout_rects = model.get_ocr_rects(page)
        assert not any(
            missing_line.intersection_over_self(rect) > 0 for rect in layout_rects
        )
    finally:
        doc_backend.unload()


def _native_text_with_bitmap_pdf(path: Path, bitmap_size: int) -> Path:
    """Create a PDF with visible native text and a background or small image."""
    image = Image.new("RGB", (bitmap_size, bitmap_size), "white")
    ImageDraw.Draw(image).text((5, 5), "Logo", fill="black")
    pdf = pdfium.PdfDocument.new()
    try:
        pdf_page = pdf.new_page(300, 300)
        picture = pdfium.PdfImage.new(pdf)
        picture.set_bitmap(pdfium.PdfBitmap.from_pil(image))
        picture.set_matrix(pdfium.PdfMatrix().scale(bitmap_size, bitmap_size))
        pdf_page.insert_obj(picture)

        font = pdfium_c.FPDFText_LoadStandardFont(pdf.raw, b"Helvetica")
        native_text = pdfium_c.FPDFPageObj_CreateTextObj(pdf.raw, font, 12.0)
        text_bytes = "Native PDF heading\0".encode("utf-16-le")
        pdfium_c.FPDFText_SetText(
            native_text,
            ctypes.cast(
                ctypes.c_char_p(text_bytes), ctypes.POINTER(pdfium_c.FPDF_WCHAR)
            ),
        )
        pdfium_c.FPDFPageObj_Transform(native_text, 1, 0, 0, 1, 20, 250)
        pdfium_c.FPDFPage_InsertObject(pdf_page.raw, native_text)
        pdf_page.gen_content()
        pdf.save(path)
    finally:
        pdf.close()
    return path


@pytest.mark.parametrize(
    "backend_cls",
    [PyPdfiumDocumentBackend, ThreadedDoclingParseDocumentBackend],
    ids=["pypdfium2", "threaded"],
)
@pytest.mark.parametrize("bitmap_size", [300, 40], ids=["background", "decoration"])
@pytest.mark.parametrize("detected", [True, False], ids=["layout", "empty-layout"])
def test_native_text_page_does_not_ocr_unselected_bitmaps(
    backend_cls, bitmap_size: int, detected: bool, tmp_path: Path
) -> None:
    source = _native_text_with_bitmap_pdf(
        tmp_path / "native_with_bitmap.pdf", bitmap_size
    )
    doc_backend = InputDocument(
        path_or_stream=source,
        format=InputFormat.PDF,
        backend=backend_cls,
    )._backend
    try:
        backend = (
            next(iter(doc_backend.iter_pages()))
            if backend_cls is ThreadedDoclingParseDocumentBackend
            else doc_backend.load_page(0)
        )
        page = Page(page_no=1, size=backend.get_size())
        page._backend = backend
        assert list(backend.get_text_cells())
        assert list(backend.get_bitmap_rects())
        page.predictions.layout = LayoutPrediction(
            clusters=[
                Cluster(
                    id=0,
                    label=DocItemLabel.TEXT,
                    bbox=BoundingBox(l=15, t=25, r=230, b=65),
                    confidence=0.9,
                )
            ]
            if detected
            else []
        )
        model = _OcrRectsOnlyModel(
            enabled=True,
            artifacts_path=None,
            options=OcrOptions(lang=["en"], mode=OcrMode.DEFAULT),
            accelerator_options=AcceleratorOptions(),
        )

        rects = model.get_ocr_rects(page)

        # Only a layout-selected region may reach OCR. Bitmap presence alone
        # must not convert a small decoration or an entire page into an OCR crop.
        assert all(
            rect.area() < page.size.width * page.size.height * 0.5 for rect in rects
        )
        if not detected or bitmap_size == 40:
            assert rects == []
    finally:
        doc_backend.unload()
