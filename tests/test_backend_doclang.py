from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.labels import DocItemLabel

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter

DOCLANG_XML = """<doclang>
  <heading>DocLang Title</heading>
  <text>Hello world</text>
  <table>
    <fcel/><text>H1</text><fcel/><text>H2</text><nl/>
    <fcel/><text>C1</text><fcel/><text>C2</text><nl/>
  </table>
</doclang>
"""

ROUNDTRIP_GT_PATHS = [
    Path("tests/data/md_deepseek/groundtruth/deepseek_simple.md.json"),
    Path("tests/data/csv/groundtruth/csv-semicolon.csv.json"),
]


def _table_grid_text(table: TableItem) -> list[list[str]]:
    return [[cell.text for cell in row] for row in table.data.grid]


@pytest.mark.parametrize("suffix", [".dclg", ".dclg.xml"])
def test_doclang_backend_converts_path(tmp_path: Path, suffix: str):
    doc_path = tmp_path / f"sample{suffix}"
    doc_path.write_text(DOCLANG_XML, encoding="utf-8")

    result = DocumentConverter(allowed_formats=[InputFormat.XML_DOCLANG]).convert(
        doc_path
    )

    assert result.input.format == InputFormat.XML_DOCLANG
    assert result.document.texts[0].label == DocItemLabel.TITLE
    assert result.document.texts[0].text == "DocLang Title"
    assert result.document.texts[1].text == "Hello world"
    assert len(result.document.tables) == 1
    table = result.document.tables[0]
    assert table.data.num_rows == 2
    assert table.data.num_cols == 2
    assert [[cell.text for cell in row] for row in table.data.grid] == [
        ["H1", "H2"],
        ["C1", "C2"],
    ]


def test_doclang_backend_converts_stream():
    stream = DocumentStream(
        name="sample.dclg.xml", stream=BytesIO(DOCLANG_XML.encode("utf-8"))
    )

    result = DocumentConverter(allowed_formats=[InputFormat.XML_DOCLANG]).convert(
        stream
    )

    assert result.input.format == InputFormat.XML_DOCLANG
    assert result.document.export_to_markdown().startswith("# DocLang Title")


def test_doclang_convert_string():
    result = DocumentConverter(
        allowed_formats=[InputFormat.XML_DOCLANG]
    ).convert_string(DOCLANG_XML, format=InputFormat.XML_DOCLANG)

    assert result.input.format == InputFormat.XML_DOCLANG
    assert result.document.export_to_markdown().startswith("# DocLang Title")


@pytest.mark.parametrize("gt_path", ROUNDTRIP_GT_PATHS)
def test_docling_document_doclang_roundtrip_from_groundtruth(gt_path: Path):
    original_doc = DoclingDocument.load_from_json(gt_path)

    result = DocumentConverter(
        allowed_formats=[InputFormat.XML_DOCLANG]
    ).convert_string(
        original_doc.export_to_doclang(),
        format=InputFormat.XML_DOCLANG,
        name=gt_path.name,
    )
    roundtrip_doc = result.document

    assert roundtrip_doc.export_to_markdown() == original_doc.export_to_markdown()
    assert [(item.label, item.text) for item in roundtrip_doc.texts] == [
        (item.label, item.text) for item in original_doc.texts
    ]
    assert [_table_grid_text(table) for table in roundtrip_doc.tables] == [
        _table_grid_text(table) for table in original_doc.tables
    ]
    assert len(roundtrip_doc.pictures) == len(original_doc.pictures)


@pytest.mark.parametrize("name", ["sample.dclg", "sample.dclg.xml", "sample.xml"])
def test_doclang_guess_format_by_extension(tmp_path: Path, name: str):
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    doc_path = tmp_path / name
    doc_path.write_text(DOCLANG_XML, encoding="utf-8")

    assert dci._guess_format(doc_path) == InputFormat.XML_DOCLANG

    stream = DocumentStream(name=name, stream=BytesIO(DOCLANG_XML.encode("utf-8")))
    assert dci._guess_format(stream) == InputFormat.XML_DOCLANG


def test_doclang_backend_converts_generic_xml_extension(tmp_path: Path):
    doc_path = tmp_path / "sample.xml"
    doc_path.write_text(DOCLANG_XML, encoding="utf-8")

    result = DocumentConverter(allowed_formats=[InputFormat.XML_DOCLANG]).convert(
        doc_path
    )

    assert result.input.format == InputFormat.XML_DOCLANG
    assert result.document.export_to_markdown().startswith("# DocLang Title")
