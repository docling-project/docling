"""Tests for the OpenDocument Format (ODF) backends (ODT/ODS/ODP).

This module includes two types of tests:
1. Unit tests with fixtures built on the fly using `odfdo` to keep the suite
   small and make inputs obvious from the code.
2. End-to-end tests using binary ODF files from tests/data/odf/regression/ to verify
   complete conversion workflows including JSON, ITXT, and Markdown exports.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import (
    ImageRefMode,
    PictureClassificationLabel,
    PictureItem,
    RichTableCell,
    Script,
    TableItem,
    TextItem,
)
from PIL import Image

from docling.backend.opendocument_backend import (
    OdpDocumentBackend,
    OdsDocumentBackend,
    OdtDocumentBackend,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

pytest.importorskip("odfdo")
from odfdo import (
    Document as OdfDocument,
    DrawPage,
    Frame,
    Header,
    List as OdfList,
    ListItem,
    Paragraph,
    Span,
    Style,
    Table,
)

pytestmark = pytest.mark.cross_platform

_log = logging.getLogger(__name__)

GENERATE = GEN_TEST_DATA


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
    # Sheet2 only has data in the first row, so it should be trimmed to 1 row
    assert (t2.data.num_rows, t2.data.num_cols) == (1, 2)
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


def test_odt_rich_table_cell_text(tmp_path: Path):
    path = tmp_path / "rich_table.odt"
    doc = OdfDocument("text")
    body = doc.body
    body.clear()

    table = Table("Rich", width=1, height=1)
    cell = table.get_cell("A1")
    cell.append(Paragraph("First paragraph"))
    lst = OdfList()
    lst.append(ListItem("Nested list item"))
    cell.append(lst)
    table.set_cell("A1", cell)
    body.append(table)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    result_doc = res.document
    table = result_doc.tables[0]
    cell = table.data.table_cells[0]
    assert isinstance(cell, RichTableCell)
    assert cell.text == "First paragraph\nNested list item"

    group = cell.ref.resolve(result_doc)
    child_texts: list[str] = []
    for item, _level in result_doc.iterate_items(root=group):
        if isinstance(item, TextItem):
            child_texts.append(item.text)
    assert child_texts == ["First paragraph", "Nested list item"]


def test_ods_rich_table_cell_defines_data_bounds(tmp_path: Path):
    path = tmp_path / "rich_table.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()

    table = Table("Rich", width=2, height=1)
    cell = table.get_cell("A1")
    cell.append(Paragraph("Rich title"))
    table.set_cell("A1", cell)
    table.set_value("B1", "plain")
    body.append(table)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODS]).convert(path)
    result_doc = res.document
    table = result_doc.tables[0]
    cell_texts = {
        (c.start_row_offset_idx, c.start_col_offset_idx): c.text
        for c in table.data.table_cells
    }
    rich_cell = cell_texts[(0, 0)]
    assert table.data.num_cols == 2
    assert rich_cell == "Rich title"
    assert isinstance(table.data.table_cells[0], RichTableCell)
    assert cell_texts[(0, 1)] == "plain"


def test_ods_table_cell_image_creates_rich_cell_picture(tmp_path: Path):
    image_path = tmp_path / "cell_image.png"
    Image.new("RGB", (2, 2), "red").save(image_path)

    path = tmp_path / "image_cell.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()

    table = Table("ImageCell", width=1, height=1)
    body.append(table)
    image_ref = doc.add_file(str(image_path))
    frame = Frame.image_frame(image_ref, size=("1cm", "1cm"))
    table.set_cell_image("A1", frame)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODS]).convert(path)
    result_doc = res.document
    table = result_doc.tables[0]
    cell = table.data.table_cells[0]
    assert isinstance(cell, RichTableCell)

    group = cell.ref.resolve(result_doc)
    child_items = [child.resolve(result_doc) for child in group.children]
    pictures = [item for item in child_items if isinstance(item, PictureItem)]
    assert len(pictures) == 1
    assert pictures[0].image is not None


def test_odt_ordered_nested_list(tmp_path: Path):
    path = tmp_path / "ordered_nested.odt"
    doc = OdfDocument("text")
    body = doc.body
    body.clear()

    style = Style("list", name="Numbered")
    style.set_level_style(1, num_format="1", suffix=".")
    style.set_level_style(2, num_format="1", suffix=")")
    doc.insert_style(style)

    outer = OdfList(style="Numbered")
    first = ListItem()
    first.append(Paragraph("One"))
    nested = OdfList(style="Numbered")
    nested.append(ListItem("Nested one"))
    first.append(nested)
    outer.append(first)
    outer.append(ListItem("Two"))
    body.append(outer)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    list_items = [
        item
        for item, _level in res.document.iterate_items()
        if isinstance(item, TextItem) and item.label == "list_item"
    ]

    assert [(item.text, item.enumerated, item.marker) for item in list_items] == [
        ("One", True, "1."),
        ("Nested one", True, "1)"),
        ("Two", True, "2."),
    ]
    nested_group = list_items[1].parent.resolve(res.document)
    assert nested_group.parent == list_items[0].get_ref()


def test_odt_text_document_nested_lists():
    path = Path("tests/data/odf/regression/text_document_01.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    list_items = [
        item
        for item, _level in res.document.iterate_items()
        if isinstance(item, TextItem) and item.label == "list_item"
    ]

    assert [(item.text, item.enumerated, item.marker) for item in list_items] == [
        ("numbered list 1", True, "1."),
        ("numbered list 2", True, "2."),
        ("bullet list 2.1", False, ""),
        ("bullet list 2.2", False, ""),
        ("bullet list 2.2.1", False, ""),
        ("bullet list 2.3", False, ""),
        ("numbered list 3", True, "3."),
        ("bullet list 3.0.1", False, ""),
    ]

    item_by_text = {item.text: item for item in list_items}
    parent_group_2_1 = item_by_text["bullet list 2.1"].parent.resolve(res.document)
    parent_group_2_2_1 = item_by_text["bullet list 2.2.1"].parent.resolve(res.document)
    parent_group_3_0_1 = item_by_text["bullet list 3.0.1"].parent.resolve(res.document)

    assert parent_group_2_1.parent == item_by_text["numbered list 2"].get_ref()
    assert parent_group_2_2_1.parent == item_by_text["bullet list 2.2"].get_ref()
    assert parent_group_3_0_1.parent == item_by_text["numbered list 3"].get_ref()


def test_odt_text_document_title_and_subtitle():
    path = Path("tests/data/odf/regression/text_document_01.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)

    titles = [item.text for item in res.document.texts if item.label == "title"]
    level_1_headings = [
        item.text
        for item in res.document.texts
        if item.label == "section_header" and item.level == 1
    ]

    assert titles == ["Title"]
    assert level_1_headings == ["Subtitle", "heading level 1.0"]


def test_odt_text_document_text_formatting():
    path = Path("tests/data/odf/regression/text_document_01.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    formatted_by_text = {
        item.text: item.formatting for item in res.document.texts if item.formatting
    }

    assert formatted_by_text["Lorem Ipsum is not simply random text"].bold
    assert formatted_by_text[
        "a Latin professor at Hampden-Sydney College in Virginia"
    ].italic
    assert formatted_by_text[
        "and going through the cites of the word in classical literature"
    ].underline
    assert formatted_by_text["The Extremes of Good and Evil"].strikethrough

    markdown = res.document.export_to_markdown(compact_tables=True)
    assert "**Lorem Ipsum is not simply random text**" in markdown
    assert "*a Latin professor at Hampden-Sydney College in Virginia*" in markdown
    assert "~~The Extremes of Good and Evil~~" in markdown


def test_odt_text_document_script_formatting():
    path = Path("tests/data/odf/regression/text_document_01.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    formula_runs = [
        item for item in res.document.texts if item.text in {"X", "2", " + Y", " = Z"}
    ]

    assert [
        (item.text, item.formatting.script if item.formatting else None)
        for item in formula_runs
    ] == [
        ("X", None),
        ("2", Script.SUPER),
        (" + Y", None),
        ("2", Script.SUB),
        (" = Z", None),
    ]


def test_odt_text_document_embedded_chart():
    path = Path("tests/data/odf/regression/text_document_02.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    charts = [
        item
        for item in res.document.pictures
        if isinstance(item, PictureItem)
        and item.meta is not None
        and item.meta.tabular_chart is not None
    ]

    assert len(charts) == 1
    chart = charts[0]
    assert chart.label == "picture"
    assert chart.meta is not None
    assert chart.meta.classification is not None
    assert (
        chart.meta.classification.predictions[0].class_name
        == PictureClassificationLabel.BAR_CHART
    )
    assert chart.meta.tabular_chart is not None

    table_data = chart.meta.tabular_chart.chart_data
    assert table_data.num_rows == 5
    assert table_data.num_cols == 4
    cell_texts = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in table_data.table_cells
    }
    assert cell_texts[(0, 1)] == "Column 1"
    assert cell_texts[(1, 0)] == "Row 1"
    assert cell_texts[(1, 1)] == "9.1"
    assert cell_texts[(4, 3)] == "6.2"


def test_odt_text_document_embedded_bitmap_image():
    path = Path("tests/data/odf/regression/text_document_02.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    bitmap_pictures = [
        item
        for item in res.document.pictures
        if isinstance(item, PictureItem)
        and item.image is not None
        and (item.meta is None or item.meta.tabular_chart is None)
    ]

    assert len(bitmap_pictures) == 1


def test_odt_table_cell_image_creates_rich_cell_picture(tmp_path: Path):
    image_path = tmp_path / "cell_image.png"
    Image.new("RGB", (2, 2), "red").save(image_path)

    path = tmp_path / "image_cell.odt"
    doc = OdfDocument("text")
    body = doc.body
    body.clear()

    table = Table("ImageCell", width=1, height=1)
    cell = table.get_cell("A1")
    image_ref = doc.add_file(str(image_path))
    cell.append(Frame.image_frame(image_ref, size=("1cm", "1cm")))
    table.set_cell("A1", cell)
    body.append(table)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    table = res.document.tables[0]
    cell = table.data.table_cells[0]
    assert isinstance(cell, RichTableCell)

    group = cell.ref.resolve(res.document)
    child_items = [child.resolve(res.document) for child in group.children]
    pictures = [item for item in child_items if isinstance(item, PictureItem)]
    assert len(pictures) == 1
    assert pictures[0].image is not None


def _rich_cell_descendants(doc: DoclingDocument, cell: RichTableCell):
    group = cell.ref.resolve(doc)
    return [
        item
        for item, _level in doc.iterate_items(root=group)
        if item.get_ref() != group.get_ref()
    ]


def test_odt_text_document_rich_table_cells():
    path = Path("tests/data/odf/regression/text_document_03.odt")

    res = DocumentConverter(allowed_formats=[InputFormat.ODT]).convert(path)
    rich_cells = [
        cell
        for table in res.document.tables
        for cell in table.data.table_cells
        if isinstance(cell, RichTableCell)
    ]

    list_cell = next(
        cell
        for cell in rich_cells
        if cell.text == "This is a list:\nA First\nA Second\nA Third"
    )
    list_descendants = _rich_cell_descendants(res.document, list_cell)
    assert any(
        isinstance(item, TextItem) and item.text == "This is a list:"
        for item in list_descendants
    )
    assert [
        item.text
        for item in list_descendants
        if getattr(item, "label", None) == "list_item"
    ] == ["A First", "A Second", "A Third"]

    multi_paragraph_cell = next(
        cell
        for cell in rich_cells
        if cell.text == "This is a paragraph\nThis is another paragraph"
    )
    multi_paragraph_texts = [
        item.text
        for item in _rich_cell_descendants(res.document, multi_paragraph_cell)
        if isinstance(item, TextItem)
    ]
    assert multi_paragraph_texts == [
        "This is a paragraph",
        "This is another paragraph",
    ]

    nested_table_cell = next(
        cell for cell in rich_cells if cell.text.startswith("Rich cell\nA nested table")
    )
    assert any(
        isinstance(item, TableItem)
        for item in _rich_cell_descendants(res.document, nested_table_cell)
    )

    picture_cell = next(
        cell for cell in rich_cells if cell.text == "Text and picture\nOwner avatar"
    )
    picture_descendants = _rich_cell_descendants(res.document, picture_cell)
    assert any(
        isinstance(item, TextItem) and item.text == "Text and picture"
        for item in picture_descendants
    )
    assert any(
        isinstance(item, PictureItem) and item.image is not None
        for item in picture_descendants
    )
    assert all("(Pictures/" not in cell.text for cell in rich_cells)


def test_odp_textbox_text_formatting(tmp_path: Path):
    path = tmp_path / "formatted_text.odp"
    doc = OdfDocument("presentation")
    body = doc.body
    body.clear()

    for name, props in {
        "Bold": {"fo:font-weight": "bold"},
        "Italic": {"fo:font-style": "italic"},
        "Underline": {"style:text-underline-style": "solid"},
        "Strike": {"style:text-line-through-style": "solid"},
    }.items():
        style = Style("text", name=name)
        style.set_properties(props)
        doc.insert_style(style)

    paragraph = Paragraph()
    paragraph.append(Span("bold", style="Bold"))
    paragraph.append(Span(" "))
    paragraph.append(Span("italic", style="Italic"))
    paragraph.append(Span(" "))
    paragraph.append(Span("underline", style="Underline"))
    paragraph.append(Span(" "))
    paragraph.append(Span("strike", style="Strike"))

    page = DrawPage("page1", name="Slide One")
    page.append(
        Frame.text_frame(
            paragraph,
            size=("10cm", "5cm"),
            position=("1cm", "1cm"),
        )
    )
    body.append(page)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODP]).convert(path)
    formatted_by_text = {
        item.text: item.formatting for item in res.document.texts if item.formatting
    }

    assert formatted_by_text["bold"].bold
    assert formatted_by_text["italic"].italic
    assert formatted_by_text["underline"].underline
    assert formatted_by_text["strike"].strikethrough


def test_odp_nested_textbox_list(tmp_path: Path):
    path = tmp_path / "nested_list.odp"
    doc = OdfDocument("presentation")
    body = doc.body
    body.clear()

    page = DrawPage("page1", name="Slide One")
    outer = OdfList()
    first = ListItem()
    first.append(Paragraph("Outer"))
    nested = OdfList()
    nested.append(ListItem("Nested"))
    first.append(nested)
    outer.append(first)
    page.append(
        Frame.text_frame(
            outer,
            size=("10cm", "5cm"),
            position=("1cm", "1cm"),
        )
    )
    body.append(page)
    doc.save(str(path))

    res = DocumentConverter(allowed_formats=[InputFormat.ODP]).convert(path)
    list_items = [
        item
        for item, _level in res.document.iterate_items()
        if isinstance(item, TextItem) and item.label == "list_item"
    ]

    assert [item.text for item in list_items] == ["Outer", "Nested"]
    nested_group = list_items[1].parent.resolve(res.document)
    assert nested_group.parent == list_items[0].get_ref()


def test_odp_presentation_with_mixed_slide_content():
    path = Path("tests/data/odf/regression/odf_presentation_02.odp")

    res = DocumentConverter(allowed_formats=[InputFormat.ODP]).convert(path)
    doc = res.document

    titles = [item.text for item in doc.texts if item.label == "title"]
    assert titles == [
        "Text",
        "Chart",
        "Table",
        "Photo",
        "Complex list",
        "Table with pictures",
    ]

    assert any(
        isinstance(item, TextItem)
        and item.label == "text"
        and item.text.startswith("Lorem ipsum dolor sit amet")
        for item in doc.texts
    )
    assert any(
        isinstance(item, PictureItem)
        and item.meta is not None
        and item.meta.tabular_chart is not None
        for item in doc.pictures
    )
    slides = {group.name: group for group in doc.groups if group.name is not None}
    assert any(
        isinstance(item, PictureItem) and item.image is not None
        for item, _level in doc.iterate_items(root=slides["slide-3"])
    )
    assert any(
        isinstance(item, PictureItem) and item.image is not None
        for item, _level in doc.iterate_items(root=slides["slide-5"])
    )
    assert any(
        table.data.num_rows == 1
        and table.data.num_cols == 1
        and table.data.table_cells[0].text == ""
        for table in doc.tables
    )


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


@pytest.fixture(scope="module")
def odf_paths() -> list[Path]:
    """Collect all ODF files from tests/data/odf directory."""
    directory = Path("./tests/data/odf/regression/")

    # List all ODF files (odt, ods, odp) in the directory
    odf_files = sorted(directory.glob("*.od[tsp]"))

    return odf_files


@pytest.fixture(scope="module")
def odf_documents(odf_paths) -> list[tuple[Path, DoclingDocument]]:
    """Convert all ODF files and return documents with their groundtruth paths."""
    documents: list[tuple[Path, DoclingDocument]] = []

    converter = DocumentConverter(
        allowed_formats=[InputFormat.ODT, InputFormat.ODS, InputFormat.ODP]
    )

    for odf_path in odf_paths:
        _log.debug(f"converting {odf_path}")

        gt_path = odf_path.parent.parent / "groundtruth" / odf_path.name

        conv_result: ConversionResult = converter.convert(odf_path)

        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {odf_path}"
        documents.append((gt_path, doc))

    return documents


def _test_e2e_odf_conversions_impl(odf_documents: list[tuple[Path, DoclingDocument]]):
    """Test end-to-end ODF conversions including JSON, HTML, ITXT, and Markdown exports."""
    for gt_path, doc in odf_documents:
        # Export to Markdown
        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            f"export to markdown failed on {gt_path}"
        )

        # Export to indented text
        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(
            pred_itxt, str(gt_path) + ".itxt", generate=GENERATE, fuzzy=True
        ), f"export to indented-text failed on {gt_path}"

        # Export to HTML
        pred_html: str = doc.export_to_html(image_mode=ImageRefMode.EMBEDDED)
        assert verify_export(pred_html, str(gt_path) + ".html", generate=GENERATE), (
            f"export to html failed on {gt_path}"
        )

        # Verify DoclingDocument JSON
        assert verify_document(
            doc, str(gt_path) + ".json", generate=GENERATE, fuzzy=True
        ), f"DoclingDocument verification failed on {gt_path}"


def test_e2e_odf_conversions(odf_documents):
    """Test end-to-end conversions for all ODF files."""
    _test_e2e_odf_conversions_impl(odf_documents)


def test_ods_treat_singleton_as_text(tmp_path: Path):
    """Test that singleton cells are treated as TextItem when option is enabled.

    When treat_singleton_as_text option is enabled, 1x1 tables should be
    converted to TextItem instead of TableItem.
    """
    from docling.datamodel.backend_options import OdsBackendOptions
    from docling.document_converter import OdsFormatOption

    # Create test ODS with a title (1x1 table) and a data table
    path = tmp_path / "test_singleton.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()

    sheet = Table("Sheet1", width=2, height=8)
    # Title in A1 (singleton)
    sheet.set_value("A1", "Number of freshwater ducks per year")

    # Data table starting at A3
    sheet.set_value("A3", "Year")
    sheet.set_value("B3", "Freshwater Ducks")
    sheet.set_value("A4", 2019)
    sheet.set_value("B4", 120)
    sheet.set_value("A5", 2020)
    sheet.set_value("B5", 135)

    body.append(sheet)
    doc.save(str(path))

    # Test with treat_singleton_as_text=True
    options = OdsBackendOptions(treat_singleton_as_text=True)
    format_options = {InputFormat.ODS: OdsFormatOption(backend_options=options)}
    converter = DocumentConverter(
        allowed_formats=[InputFormat.ODS], format_options=format_options
    )

    conv_result: ConversionResult = converter.convert(path)
    result_doc: DoclingDocument = conv_result.document

    # With treat_singleton_as_text=True, the singleton title cell should be a TextItem
    texts = list(result_doc.texts)
    tables = list(result_doc.tables)

    assert len(texts) == 1, f"Should have 1 text item (the title), got {len(texts)}"
    assert len(tables) == 1, f"Should have 1 table, got {len(tables)}"

    # Verify the text item contains the title
    assert texts[0].text == "Number of freshwater ducks per year", (
        f"Text should be 'Number of freshwater ducks per year', got '{texts[0].text}'"
    )

    # Verify table dimensions (should be the data table only)
    table = tables[0]
    assert table.data.num_rows == 3, (
        f"Table should have 3 rows, got {table.data.num_rows}"
    )
    assert table.data.num_cols == 2, (
        f"Table should have 2 columns, got {table.data.num_cols}"
    )


def test_ods_gap_tolerance(tmp_path: Path):
    """Test the effect of gap_tolerance on table detection.

    Verifies:
    1. Tolerance 0 (Default): Gaps cause splits into separate tables.
    2. Tolerance 1: Gaps are bridged, merging tables.
    """
    from docling.datamodel.backend_options import OdsBackendOptions
    from docling.document_converter import OdsFormatOption

    # Create test ODS with data separated by an empty column
    path = tmp_path / "test_gap.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()

    sheet = Table("Sheet1", width=4, height=3)
    # Column A has data
    sheet.set_value("A1", "Col A")
    sheet.set_value("A2", "Data 1")
    sheet.set_value("A3", "Data 2")

    # Column B is empty (gap)

    # Column C has data
    sheet.set_value("C1", "Col C")
    sheet.set_value("C2", "Data 3")
    sheet.set_value("C3", "Data 4")

    body.append(sheet)
    doc.save(str(path))

    # Test with gap_tolerance=0 (strict - should split)
    options_strict = OdsBackendOptions(gap_tolerance=0)
    format_options_strict = {
        InputFormat.ODS: OdsFormatOption(backend_options=options_strict)
    }
    converter_strict = DocumentConverter(
        allowed_formats=[InputFormat.ODS], format_options=format_options_strict
    )
    doc_strict = converter_strict.convert(path).document

    # With gap_tolerance=0, should have 2 separate tables
    tables_strict = list(doc_strict.tables)
    assert len(tables_strict) == 2, (
        f"With gap_tolerance=0, should have 2 tables, got {len(tables_strict)}"
    )

    # Test with gap_tolerance=1 (should merge)
    options_merged = OdsBackendOptions(gap_tolerance=1)
    format_options_merged = {
        InputFormat.ODS: OdsFormatOption(backend_options=options_merged)
    }
    converter_merged = DocumentConverter(
        allowed_formats=[InputFormat.ODS], format_options=format_options_merged
    )
    doc_merged = converter_merged.convert(path).document

    # With gap_tolerance=1, should merge into 1 table
    tables_merged = list(doc_merged.tables)
    assert len(tables_merged) == 1, (
        f"With gap_tolerance=1, should have 1 table, got {len(tables_merged)}"
    )

    # Verify the merged table spans from column 0 to column 2 (A to C)
    merged_table = tables_merged[0]
    assert merged_table.prov[0].bbox.l == 0, "Merged table should start at column 0"
    assert merged_table.prov[0].bbox.r == 3, "Merged table should end at column 3"


def test_ods_sheet_names_filter(tmp_path: Path):
    """Test that sheet_names option filters sheets correctly."""
    from docling.datamodel.backend_options import OdsBackendOptions
    from docling.document_converter import OdsFormatOption

    # Create test ODS with 3 sheets
    path = tmp_path / "test_sheets.ods"
    doc = OdfDocument("spreadsheet")
    body = doc.body
    body.clear()

    sheet1 = Table("Sheet1", width=2, height=1)
    sheet1.set_value("A1", "Sheet1 Data")

    sheet2 = Table("Sheet2", width=2, height=1)
    sheet2.set_value("A1", "Sheet2 Data")

    sheet3 = Table("Sheet3", width=2, height=1)
    sheet3.set_value("A1", "Sheet3 Data")

    body.append(sheet1)
    body.append(sheet2)
    body.append(sheet3)
    doc.save(str(path))

    # Test 1: Convert all sheets (default)
    converter_all = DocumentConverter(allowed_formats=[InputFormat.ODS])
    doc_all = converter_all.convert(path).document
    assert len(doc_all.pages) == 3, f"Should have 3 pages, got {len(doc_all.pages)}"
    assert len(doc_all.groups) == 3, f"Should have 3 groups, got {len(doc_all.groups)}"

    # Test 2: Convert only Sheet1 and Sheet3
    options_filtered = OdsBackendOptions(sheet_names=["Sheet1", "Sheet3"])
    format_options_filtered = {
        InputFormat.ODS: OdsFormatOption(backend_options=options_filtered)
    }
    converter_filtered = DocumentConverter(
        allowed_formats=[InputFormat.ODS], format_options=format_options_filtered
    )
    doc_filtered = converter_filtered.convert(path).document

    assert len(doc_filtered.pages) == 2, (
        f"Should have 2 pages, got {len(doc_filtered.pages)}"
    )
    assert len(doc_filtered.groups) == 2, (
        f"Should have 2 groups, got {len(doc_filtered.groups)}"
    )

    group_names = [g.name for g in doc_filtered.groups]
    assert "sheet: Sheet1" in group_names, "Sheet1 should be included"
    assert "sheet: Sheet3" in group_names, "Sheet3 should be included"
    assert "sheet: Sheet2" not in group_names, "Sheet2 should be filtered out"

    # Test 3: Convert only Sheet2
    options_single = OdsBackendOptions(sheet_names=["Sheet2"])
    format_options_single = {
        InputFormat.ODS: OdsFormatOption(backend_options=options_single)
    }
    converter_single = DocumentConverter(
        allowed_formats=[InputFormat.ODS], format_options=format_options_single
    )
    doc_single = converter_single.convert(path).document

    assert len(doc_single.pages) == 1, (
        f"Should have 1 page, got {len(doc_single.pages)}"
    )
    assert len(doc_single.groups) == 1, (
        f"Should have 1 group, got {len(doc_single.groups)}"
    )
    assert doc_single.groups[0].name == "sheet: Sheet2", "Should only have Sheet2"
