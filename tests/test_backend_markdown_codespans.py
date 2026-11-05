# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import CodeItem, DocItemLabel, RichTableCell

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter

pytestmark = pytest.mark.cross_platform


@pytest.mark.parametrize("container", ["paragraph", "heading", "list", "table"])
@pytest.mark.parametrize(
    ("code_span", "expected"),
    [
        ("` x`", " x"),
        ("`x `", "x "),
        ("`  x  `", " x "),
        ("`   `", "   "),
        ("`\tx\t`", "\tx\t"),
        ("`\u00a0x\u00a0`", "\u00a0x\u00a0"),
        ("` x `", "x"),
    ],
)
def test_code_span_content_keeps_normalized_whitespace(
    container: str, code_span: str, expected: str
) -> None:
    # CommonMark 6.1 removes one space at each end only when both ends are
    # spaces and the span is not all spaces. Marko has already done that.
    sources = {
        "paragraph": f"Before {code_span} after.",
        "heading": f"## {code_span}\n\nAfter",
        "list": f"- {code_span}\n\nAfter",
        "table": f"| Code |\n| --- |\n| {code_span} |",
    }
    result = DocumentConverter(allowed_formats=[InputFormat.MD]).convert_string(
        sources[container], format=InputFormat.MD
    )
    assert result.status == ConversionStatus.SUCCESS
    doc = result.document

    if container in ("paragraph", "table"):
        items = [item for item in doc.texts if isinstance(item, CodeItem)]
    else:
        label = (
            DocItemLabel.SECTION_HEADER
            if container == "heading"
            else DocItemLabel.LIST_ITEM
        )
        items = [item for item in doc.texts if item.label == label]
        assert not any(isinstance(item, CodeItem) for item in doc.texts)
        assert doc.texts[-1].text == "After"
        assert doc.texts[-1].parent == doc.body.get_ref()

    assert len(items) == 1
    assert items[0].text == expected
    assert items[0].orig == expected

    if container == "table":
        cell = doc.tables[0].data.table_cells[1]
        assert isinstance(cell, RichTableCell)
        assert cell.text == expected
        group = cell.ref.resolve(doc)
        inline = group.children[0].resolve(doc)
        assert [ref.resolve(doc) for ref in inline.children] == items
        assert group.parent == doc.tables[0].get_ref()
        assert items[0].parent == inline.get_ref()
        assert doc.body.children == [doc.tables[0].get_ref()]


def test_plain_table_cell_padding_is_still_trimmed() -> None:
    result = DocumentConverter(allowed_formats=[InputFormat.MD]).convert_string(
        "| Header |\n| --- |\n|   plain text   |", format=InputFormat.MD
    )
    cell = result.document.tables[0].data.table_cells[1]
    assert cell.text == "plain text"
    assert not isinstance(cell, RichTableCell)
