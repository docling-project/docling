# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Office rasterization must serialize PDFium access, including cleanup."""

from collections.abc import Callable
from pathlib import Path

import docx
import pypdfium2
import pytest
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches

from docling.backend.docx.drawingml.utils import get_pil_from_dml_docx
from docling.backend.iwork.chart_image import render_chart
from docling.backend.iwork.content import Chart, ChartKind, ChartSeries
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.utils.locks import pypdfium2_lock


@pytest.fixture(
    params=[
        "docx",
        "pptx_metafile",
        "pptx_chart",
        "xlsx_metafile",
        "xlsx_chart",
        "iwork_chart",
    ]
)
def office_render(request, tmp_path: Path) -> Callable[[], Image.Image | None]:
    def converter(source: Path, output: Path) -> None:
        assert source.exists()
        assert not pypdfium2_lock.locked(), (
            "LibreOffice must run outside the PDFium lock"
        )
        image = Image.new("RGB", (100, 80), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 79, 59), fill="black")
        image.save(output, "PDF", resolution=72)

    if request.param == "docx":
        return lambda: get_pil_from_dml_docx(docx.Document(), converter)
    if request.param == "iwork_chart":
        chart = Chart(ChartKind.COLUMN, "Chart", ("A",), (ChartSeries("S", (1.0,)),))
        return lambda: render_chart(chart, None, converter)
    if request.param.startswith("pptx"):
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        data = CategoryChartData()
        data.categories = ["A"]
        data.add_series("S", [1])
        shape = slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            Inches(1),
            Inches(1),
            Inches(4),
            Inches(3),
            data,
        )
        path = tmp_path / "chart.pptx"
        presentation.save(path)
        document = InputDocument(path, InputFormat.PPTX, MsPowerpointDocumentBackend)
        backend = document._backend
        assert isinstance(backend, MsPowerpointDocumentBackend)
        backend.pptx_to_pdf_converter_init = True
        backend.pptx_to_pdf_converter = converter
        if request.param == "pptx_metafile":
            return lambda: backend._rasterize_metafile(b"metafile")
        return lambda: backend._render_chart_image(0, shape.shape_id)

    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["A", 1])
    chart = BarChart()
    chart.add_data(Reference(sheet, min_col=2, min_row=1))
    chart.set_categories(Reference(sheet, min_col=1, min_row=1))
    sheet.add_chart(chart)
    path = tmp_path / "chart.xlsx"
    workbook.save(path)
    document = InputDocument(path, InputFormat.XLSX, MsExcelDocumentBackend)
    backend = document._backend
    assert isinstance(backend, MsExcelDocumentBackend)
    backend.xlsx_to_pdf_converter_init = True
    backend.xlsx_to_pdf_converter = converter
    if request.param == "xlsx_metafile":
        return lambda: backend._convert_emf_to_pil(b"metafile")
    assert backend.workbook is not None
    return lambda: backend._render_chart_image(
        backend.workbook.worksheets[0]._charts[0]
    )


@pytest.mark.parametrize("failure", [None, "page", "render"])
def test_office_pdfium_lifecycle_is_locked(office_render, monkeypatch, failure):
    """Use real PDFium objects; observe locking and inject boundary failures."""
    documents = []
    pages = []
    locked_calls = []
    real_init = pypdfium2.PdfDocument.__init__
    real_get_page = pypdfium2.PdfDocument.get_page
    real_render = pypdfium2.PdfPage.render
    real_document_close = pypdfium2.PdfDocument.close
    real_page_close = pypdfium2.PdfPage.close

    def init(self, *args, **kwargs):
        locked_calls.append(pypdfium2_lock.locked())
        real_init(self, *args, **kwargs)
        documents.append(self)

    def get_page(self, *args, **kwargs):
        locked_calls.append(pypdfium2_lock.locked())
        if failure == "page":
            raise ValueError("injected page failure")
        page = real_get_page(self, *args, **kwargs)
        pages.append(page)
        return page

    def render(self, *args, **kwargs):
        locked_calls.append(pypdfium2_lock.locked())
        if failure == "render":
            raise ValueError("injected render failure")
        return real_render(self, *args, **kwargs)

    def document_close(self):
        locked_calls.append(pypdfium2_lock.locked())
        return real_document_close(self)

    def page_close(self):
        locked_calls.append(pypdfium2_lock.locked())
        return real_page_close(self)

    monkeypatch.setattr(pypdfium2.PdfDocument, "__init__", init)
    monkeypatch.setattr(pypdfium2.PdfDocument, "get_page", get_page)
    monkeypatch.setattr(pypdfium2.PdfPage, "render", render)
    monkeypatch.setattr(pypdfium2.PdfDocument, "close", document_close)
    monkeypatch.setattr(pypdfium2.PdfPage, "close", page_close)
    try:
        try:
            image = office_render()
        except ValueError as exc:
            # DrawingML propagates errors; the other paths return None.
            assert str(exc) == f"injected {failure} failure"
            assert failure is not None
            image = None
        assert documents, "the Office path must open a real PDF"
        assert all(locked_calls), "Office PDFium calls must hold pypdfium2_lock"
        assert all(not page.raw for page in pages), (
            "PDF pages must close on failure too"
        )
        assert all(not pdf.raw for pdf in documents), (
            "PDF documents must close on failure too"
        )
        assert not pypdfium2_lock.locked()
        if failure is None:
            assert image is not None
            assert image.width < 200 and image.height < 160
            assert image.convert("L").getextrema()[0] < 30
        else:
            assert image is None
    finally:
        # Keep references until after the assertions, so GC cannot hide leaks.
        with pypdfium2_lock:
            for page in pages:
                real_page_close(page)
            for pdf in documents:
                real_document_close(pdf)
