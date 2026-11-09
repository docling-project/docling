# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for Word tables nested inside table cells.

Kept separate from ``test_backend_msword.py`` so that file stays under the
repository's per-file line limit.
"""

from pathlib import Path

from docling_core.types.doc import DoclingDocument, GroupItem, TableItem
from docx import Document

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


def _convert(document, tmp_path: Path) -> DoclingDocument:
    path = tmp_path / "nested_tables.docx"
    document.save(path)
    converter = DocumentConverter(allowed_formats=[InputFormat.DOCX])
    return converter.convert(path).document


def _rich_cell_group(doc: DoclingDocument, table: TableItem) -> GroupItem:
    groups = [child.resolve(doc) for child in table.children]
    assert len(groups) == 1, "the outer table must hold one rich cell group"
    group = groups[0]
    assert isinstance(group, GroupItem)
    return group


def test_single_cell_table_in_cell_stays_inside_the_cell(tmp_path):
    """A 1x1 table unwraps into its surroundings. Inside a table cell those
    surroundings are the cell, so its content must join the rich cell group
    and not be emitted after the outer table.
    """
    document = Document()
    outer = document.add_table(rows=1, cols=2)
    outer.cell(0, 0).text = "left"
    inner = outer.cell(0, 1).add_table(rows=1, cols=1)
    inner.cell(0, 0).text = "boxed"
    document.add_paragraph("after the table")

    doc = _convert(document, tmp_path)

    assert len(doc.tables) == 1
    group = _rich_cell_group(doc, doc.tables[0])
    group_texts = [child.resolve(doc).text for child in group.children]
    assert [text for text in group_texts if text.strip()] == ["boxed"]

    body_texts = [
        item.text
        for item in doc.texts
        if item.text.strip() and item.parent.resolve(doc) is doc.body
    ]
    assert body_texts == ["after the table"]

    table_lines = [
        line for line in doc.export_to_markdown().splitlines() if line.startswith("|")
    ]
    assert any("left" in line and "boxed" in line for line in table_lines)


def test_single_cell_table_in_cell_keeps_its_place_between_paragraphs(tmp_path):
    """Content of a nested 1x1 table keeps its position among the cell's own
    paragraphs.
    """
    document = Document()
    outer = document.add_table(rows=1, cols=2)
    outer.cell(0, 0).text = "left"
    cell = outer.cell(0, 1)
    cell.paragraphs[0].text = "before"
    inner = cell.add_table(rows=1, cols=1)
    inner.cell(0, 0).text = "boxed"
    cell.add_paragraph("after")

    doc = _convert(document, tmp_path)

    group = _rich_cell_group(doc, doc.tables[0])
    group_texts = [child.resolve(doc).text for child in group.children]
    assert [text for text in group_texts if text.strip()] == [
        "before",
        "boxed",
        "after",
    ]
