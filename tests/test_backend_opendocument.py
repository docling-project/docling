"""Tests for the OpenDocument (ODT/ODS/ODP) backends.

Fixtures are built on the fly with ``odfdo`` rather than committed as binary
files so the suite stays small and the inputs are obvious from the code.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.opendocument_backend import (
    OdpDocumentBackend,
    OdsDocumentBackend,
    OdtDocumentBackend,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter

pytest.importorskip("odfdo")
from odfdo import (
    Document as OdfDocument,
    DrawPage,
    Frame,
    Header,
    List as OdfList,
    ListItem,
    Paragraph,
    Table,
)

pytestmark = pytest.mark.cross_platform


def _build_odt(path: Path) -> Path:
    doc = OdfDocument("text")
    body = doc.body
    body.clear()
    body.append(Header(1, "Title One"))
    body.append(Paragraph("An introductory paragraph."))
    body.append(Header(2, "Subsection"))
    lst = OdfList()
    lst.append(ListItem("First item"))
    lst.append(ListItem("Second item"))
    body.append(lst)
    tbl = Table("Demo", width=2, height=2)
    tbl.set_value("A1", "h1")
    tbl.set_value("B1", "h2")
    tbl.set_value("A2", "v1")
    tbl.set_value("B2", "v2")
    body.append(tbl)
    doc.save(str(path))
    return path


def _build_ods(path: Path) -> Path:
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()
    sheet1 = Table("Sheet1", width=3, height=2)
    sheet1.set_value("A1", "a")
    sheet1.set_value("B1", "b")
    sheet1.set_value("C1", "c")
    sheet1.set_value("A2", 1)
    sheet1.set_value("B2", 2)
    sheet1.set_value("C2", 3)
    sheet2 = Table("Sheet2", width=2, height=2)
    sheet2.set_value("A1", "x")
    sheet2.set_value("B1", "y")
    body.append(sheet1)
    body.append(sheet2)
    doc.save(str(path))
    return path


def _build_odp(path: Path) -> Path:
    doc = OdfDocument("presentation")
    body = doc.body
    body.clear()
    page1 = DrawPage("page1", name="Slide One")
    page1.append(
        Frame.text_frame(
            ["Headline Slide"], size=("10cm", "2cm"), position=("1cm", "1cm")
        )
    )
    page1.append(
        Frame.text_frame(
            ["First bullet", "Second bullet"],
            size=("10cm", "5cm"),
            position=("1cm", "4cm"),
        )
    )
    body.append(page1)
    page2 = DrawPage("page2", name="Slide Two")
    page2.append(
        Frame.text_frame(
            ["Second Slide Heading"], size=("10cm", "2cm"), position=("1cm", "1cm")
        )
    )
    body.append(page2)
    doc.save(str(path))
    return path


@pytest.fixture(scope="module")
def odt_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_odt(tmp_path_factory.mktemp("odf") / "demo.odt")


@pytest.fixture(scope="module")
def ods_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_ods(tmp_path_factory.mktemp("odf") / "demo.ods")


@pytest.fixture(scope="module")
def odp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_odp(tmp_path_factory.mktemp("odf") / "demo.odp")


def test_odt_supported_formats():
    assert OdtDocumentBackend.supported_formats() == {InputFormat.ODT}
    assert OdsDocumentBackend.supported_formats() == {InputFormat.ODS}
    assert OdpDocumentBackend.supported_formats() == {InputFormat.ODP}
    assert OdtDocumentBackend.supports_pagination() is False
    assert OdsDocumentBackend.supports_pagination() is True
    assert OdpDocumentBackend.supports_pagination() is True


def test_odt_conversion(odt_path: Path):
    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(odt_path)
    doc = res.document

    assert [h.text for h in doc.texts if h.label == "section_header"] == [
        "Title One",
        "Subsection",
    ]
    assert any(t.text == "An introductory paragraph." for t in doc.texts)

    list_items = [t.text for t in doc.texts if t.label == "list_item"]
    assert list_items == ["First item", "Second item"]

    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.data.num_rows == 2
    assert table.data.num_cols == 2
    cell_texts = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c.text
        for c in table.data.table_cells
    }
    assert cell_texts == {(0, 0): "h1", (0, 1): "h2", (1, 0): "v1", (1, 1): "v2"}


def test_ods_conversion(ods_path: Path):
    res = DocumentConverter(allowed_formats=[InputFormat.ODS]).convert(ods_path)
    doc = res.document

    assert len(doc.tables) == 2
    t1, t2 = doc.tables
    assert (t1.data.num_rows, t1.data.num_cols) == (2, 3)
    assert (t2.data.num_rows, t2.data.num_cols) == (2, 2)
    sheet1_cells = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c.text
        for c in t1.data.table_cells
    }
    assert sheet1_cells[(0, 0)] == "a"
    assert sheet1_cells[(1, 2)] == "3"


def test_odp_conversion(odp_path: Path):
    res = DocumentConverter(allowed_formats=[InputFormat.ODP]).convert(odp_path)
    doc = res.document

    titles = [t.text for t in doc.texts if t.label == "title"]
    assert titles == ["Slide One", "Slide Two"]

    # Slide bodies should contribute these text items
    body_texts = {t.text for t in doc.texts if t.label == "text"}
    assert {
        "Headline Slide",
        "First bullet",
        "Second bullet",
        "Second Slide Heading",
    } <= body_texts


def test_ods_merged_cells(tmp_path: Path):
    path = tmp_path / "merged.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()
    t = Table("S", width=3, height=3)
    t.set_value("A1", "merged")
    t.set_value("B1", "b")
    t.set_value("C1", "c")
    t.set_value("B2", "d")
    t.set_value("C2", "e")
    t.set_value("A3", "x")
    t.set_value("B3", "y")
    t.set_value("C3", "z")
    t.set_span([0, 0, 0, 1])  # A1 spans two rows
    body.append(t)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODS]).convert(path)
    table = res.document.tables[0]
    anchor = next(
        c
        for c in table.data.table_cells
        if c.start_row_offset_idx == 0 and c.start_col_offset_idx == 0
    )
    assert anchor.row_span == 2
    assert anchor.col_span == 1
    assert anchor.text == "merged"


def test_odt_mime_detection_without_extension(odt_path: Path):
    # No filename extension forces the conversion pipeline to detect the format
    # by inspecting the zip contents (mimetype file).
    data = odt_path.read_bytes()
    stream = DocumentStream(name="anonymous_blob", stream=BytesIO(data))
    res = DocumentConverter().convert(stream)
    assert res.input.format == InputFormat.ODT


def test_invalid_odf_document_type(tmp_path: Path, odt_path: Path):
    in_doc = InputDocument(
        path_or_stream=odt_path,
        format=InputFormat.ODS,
        backend=OdsDocumentBackend,
        filename=odt_path.name,
    )
    assert not in_doc.valid
