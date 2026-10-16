# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the Word, Excel and PowerPoint documents embedded in a Word file.

The fixture was made in Microsoft Word. It embeds an Excel workbook, a
PowerPoint presentation, a Word document (which embeds a workbook of its own)
and a text file, which Word stores as a binary OLE package.
"""

import logging
import zipfile
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import (
    DocItem,
    DoclingDocument,
    GroupItem,
    NodeItem,
    PictureItem,
    RichTableCell,
    TableItem,
    TextItem,
)
from lxml import etree

import docling.backend.msexcel_backend as msexcel_backend_module
import docling.backend.msword_backend as msword_backend_module
from docling.datamodel.backend_options import MsWordBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter, WordFormatOption

FIXTURE = Path("./tests/data/docx/embedded_objects/docx_embedded_objects.docx")
RICH_CELLS_DOCX = Path("./tests/data/docx/sources/docx_rich_cells.docx")
EXCEL_PART = "word/embeddings/Microsoft_Excel_Worksheet.xlsx"
WORD_PART = "word/embeddings/Microsoft_Word_Document.docx"
EXCEL_GROUP = "embedded: Microsoft_Excel_Worksheet.xlsx"
POWERPOINT_GROUP = "embedded: Microsoft_PowerPoint_Presentation.pptx"
WORD_GROUP = "embedded: Microsoft_Word_Document.docx"


@pytest.fixture(autouse=True)
def no_libreoffice(monkeypatch):
    # The previews are EMF pictures. Rendering them needs LibreOffice and is
    # not what these tests check.
    monkeypatch.setattr(
        msword_backend_module, "get_docx_to_pdf_converter", lambda: None
    )


def _convert(data: bytes, process_embedded_objects: bool = True) -> DoclingDocument:
    converter = DocumentConverter(
        format_options={
            InputFormat.DOCX: WordFormatOption(
                backend_options=MsWordBackendOptions(
                    process_embedded_objects=process_embedded_objects
                )
            )
        }
    )
    stream = DocumentStream(name="embedded_objects.docx", stream=BytesIO(data))
    return converter.convert(stream).document


def _rewrite_parts(data: bytes, replacements: dict[str, bytes]) -> bytes:
    out = BytesIO()
    with (
        zipfile.ZipFile(BytesIO(data)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            target.writestr(
                info, replacements.get(info.filename, source.read(info.filename))
            )
    return out.getvalue()


def _embedded_groups(doc: DoclingDocument) -> dict[str, GroupItem]:
    return {
        group.name: group for group in doc.groups if group.name.startswith("embedded: ")
    }


def _items_under(doc: DoclingDocument, node: NodeItem) -> list[NodeItem]:
    return [
        item
        for item, _ in doc.iterate_items(root=node, traverse_pictures=True)
        if item is not node
    ]


def _texts_under(doc: DoclingDocument, node: NodeItem) -> list[str]:
    return [item.text for item in _items_under(doc, node) if isinstance(item, TextItem)]


def _describe(item: NodeItem) -> str:
    if isinstance(item, GroupItem):
        return item.name
    if isinstance(item, TextItem):
        return item.text
    assert isinstance(item, DocItem)
    return item.label


def test_embedded_objects_are_not_converted_by_default():
    doc = _convert(FIXTURE.read_bytes(), process_embedded_objects=False)

    assert not _embedded_groups(doc)
    assert len(doc.pictures) == 4
    assert not doc.tables


def test_embedded_documents_are_converted_after_their_preview():
    doc = _convert(FIXTURE.read_bytes())

    groups = _embedded_groups(doc)
    assert list(groups) == [EXCEL_GROUP, POWERPOINT_GROUP, WORD_GROUP]
    for group in groups.values():
        assert isinstance(group.children[0].resolve(doc), PictureItem)

    section_ref = groups[EXCEL_GROUP].parent
    assert section_ref is not None
    section = section_ref.resolve(doc)
    assert [_describe(ref.resolve(doc)) for ref in section.children][:9] == [
        "The budget workbook is embedded below.",
        EXCEL_GROUP,
        "The roadmap slide is embedded below.",
        POWERPOINT_GROUP,
        "The meeting notes document is embedded below.",
        WORD_GROUP,
        "A text file is embedded as a package.",
        "picture",
        "End of report.",
    ]

    tables = [
        item
        for item in _items_under(doc, groups[EXCEL_GROUP])
        if isinstance(item, TableItem)
    ]
    assert len(tables) == 1
    assert [[cell.text for cell in row] for row in tables[0].data.grid] == [
        ["Item", "Amount"],
        ["Rent", "1200"],
        ["Travel", "300"],
        ["Total", "1500"],
    ]
    assert _texts_under(doc, groups[POWERPOINT_GROUP]) == [
        "Roadmap",
        "Ship embedded object parsing",
    ]
    assert _texts_under(doc, groups[WORD_GROUP])[:2] == [
        "Meeting notes: approve the budget.",
        "Next meeting is on Friday.",
    ]


def test_objects_embedded_in_an_embedded_document_are_not_converted():
    doc = _convert(FIXTURE.read_bytes())

    word_items = _items_under(doc, _embedded_groups(doc)[WORD_GROUP])
    # The preview of the Word document, then the preview of its own workbook.
    assert sum(isinstance(item, PictureItem) for item in word_items) == 2
    assert not any(isinstance(item, TableItem) for item in word_items)
    assert len(doc.tables) == 1


def test_embedded_content_exports_without_page_references():
    # The Excel and PowerPoint backends give their items page provenance, but a
    # Word document has no pages. DocTags export fails on a dangling page.
    doc = _convert(FIXTURE.read_bytes())

    assert "<fcel>Rent<fcel>1200" in doc.export_to_doctags()


def test_rich_table_cells_of_an_embedded_document_point_to_their_copies():
    data = _rewrite_parts(
        FIXTURE.read_bytes(), {WORD_PART: RICH_CELLS_DOCX.read_bytes()}
    )
    doc = _convert(data)

    word_tables = [
        item
        for item in _items_under(doc, _embedded_groups(doc)[WORD_GROUP])
        if isinstance(item, TableItem)
    ]
    rich_cells = [
        (table, cell)
        for table in word_tables
        for cell in table.data.table_cells
        if isinstance(cell, RichTableCell)
    ]
    assert rich_cells
    for table, cell in rich_cells:
        assert cell.ref.resolve(doc).parent.cref == table.self_ref


def test_broken_embedded_document_keeps_its_preview(caplog):
    data = _rewrite_parts(FIXTURE.read_bytes(), {EXCEL_PART: b"not a workbook"})

    with caplog.at_level(logging.WARNING, logger=msword_backend_module.__name__):
        doc = _convert(data)

    assert list(_embedded_groups(doc)) == [POWERPOINT_GROUP, WORD_GROUP]
    assert len(doc.pictures) == 5
    assert "Microsoft_Excel_Worksheet.xlsx" in caplog.text


def test_missing_extra_keeps_the_preview(monkeypatch, caplog):
    monkeypatch.setattr(msexcel_backend_module, "_OPENPYXL_AVAILABLE", False)

    with caplog.at_level(logging.WARNING, logger=msword_backend_module.__name__):
        doc = _convert(FIXTURE.read_bytes())

    assert list(_embedded_groups(doc)) == [POWERPOINT_GROUP, WORD_GROUP]
    assert "docling-slim[format-xlsx]" in caplog.text


def test_part_shared_by_two_objects_is_converted_once():
    # A crafted file can point many objects at one large part. Each part is
    # converted only once, so the work stays bounded by the file content.
    with zipfile.ZipFile(FIXTURE) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
    namespaces = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "o": "urn:schemas-microsoft-com:office:office",
    }
    paragraph = document.xpath(
        "//w:p[.//o:OLEObject[@ProgID='Excel.Sheet.12']]", namespaces=namespaces
    )[0]
    paragraph.addnext(deepcopy(paragraph))
    data = _rewrite_parts(
        FIXTURE.read_bytes(), {"word/document.xml": etree.tostring(document)}
    )

    doc = _convert(data)

    assert list(_embedded_groups(doc)) == [EXCEL_GROUP, POWERPOINT_GROUP, WORD_GROUP]
    assert len(doc.tables) == 1
    assert len(doc.pictures) == 6


def test_linked_object_keeps_its_preview():
    with zipfile.ZipFile(FIXTURE) as archive:
        rels = etree.fromstring(archive.read("word/_rels/document.xml.rels"))
    for rel in rels:
        if rel.get("Target") == "embeddings/Microsoft_Excel_Worksheet.xlsx":
            rel.set("Target", "file:///C:/Reports/budget.xlsx")
            rel.set("TargetMode", "External")
    data = _rewrite_parts(
        FIXTURE.read_bytes(),
        {"word/_rels/document.xml.rels": etree.tostring(rels)},
    )

    doc = _convert(data)

    assert list(_embedded_groups(doc)) == [POWERPOINT_GROUP, WORD_GROUP]
    assert len(doc.pictures) == 5
