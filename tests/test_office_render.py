# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Office backends rasterize their LibreOffice output via docling-parse.

docling-parse renders through its own native engine, so the Office paths need
neither `pypdfium2` nor the shared `pypdfium2_lock` that the PDF backends hold
for thread safety.
"""

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

import docling.backend.docx.drawingml.utils as drawingml_utils
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
def office_render(request, tmp_path: Path):
    def converter(source: Path, output: Path) -> None:
        assert source.exists()
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


def test_office_render_bypasses_pdfium(office_render, monkeypatch):
    """No Office path may touch PDFium while rendering."""

    def forbidden(self, *args, **kwargs):
        raise AssertionError("Office rendering must not use PDFium")

    monkeypatch.setattr(pypdfium2.PdfDocument, "__init__", forbidden)

    image = office_render()
    assert image is not None
    # The drawn rectangle spans 60x40 pt; the crop trims to its ink.
    assert image.width < 200 and image.height < 160
    assert image.convert("L").getextrema()[0] < 30


def test_office_render_works_while_pdfium_lock_is_held(office_render):
    """The Office paths must not serialize against PDF-based backends.

    `PyPdfiumDocumentBackend` holds `pypdfium2_lock` for the whole parse; a
    concurrent Office conversion used to crash or corrupt that parse. With
    docling-parse the Office path ignores the lock entirely.
    """
    with pypdfium2_lock:
        image = office_render()
        assert pypdfium2_lock.locked(), "the lock must stay held during the render"
    assert image is not None


def test_office_render_reports_docling_parse_failure(office_render, monkeypatch):
    """A docling-parse failure keeps the per-path error semantics."""

    class FailingParser:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            raise RuntimeError("injected parse failure")

    monkeypatch.setattr(drawingml_utils, "_DOCLING_PARSE_AVAILABLE", True)
    monkeypatch.setattr(drawingml_utils, "DoclingThreadedPdfParser", FailingParser)

    try:
        image = office_render()
    except RuntimeError as exc:
        # DrawingML propagates errors; the other paths return None.
        assert str(exc) == "injected parse failure"
        return
    assert image is None
