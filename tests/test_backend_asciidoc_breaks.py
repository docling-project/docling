# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest
from docling_core.types.doc import DoclingDocument, TextItem

from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def _convert_adoc(source: str) -> DoclingDocument:
    in_doc = InputDocument(
        path_or_stream=BytesIO(source.encode("utf-8")),
        format=InputFormat.ASCIIDOC,
        backend=AsciiDocBackend,
        filename="test.adoc",
    )
    return in_doc._backend.convert()


def _texts(doc: DoclingDocument) -> list[tuple[str, str]]:
    return [
        (type(item).__name__, item.text)
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem)
    ]


@pytest.mark.parametrize(
    "marker",
    ["'''", "''''''", "---", "***", "___", "- - -", "* * *", "_ _ _", "<<<"],
)
def test_break_line_is_not_content(marker: str) -> None:
    """A thematic break or a page break separates blocks and adds no text."""
    doc = _convert_adoc(f"First paragraph.\n\n{marker}\n\nSecond paragraph.\n")

    assert _texts(doc) == [
        ("TextItem", "First paragraph."),
        ("TextItem", "Second paragraph."),
    ]


@pytest.mark.parametrize("text", ["-- not a break", "<<anchor>>", "**", "''"])
def test_other_lines_stay_text(text: str) -> None:
    """Lines that only look like a break keep their text."""
    doc = _convert_adoc(f"Before.\n\n{text}\n\nAfter.\n")

    assert _texts(doc) == [
        ("TextItem", "Before."),
        ("TextItem", text),
        ("TextItem", "After."),
    ]


def test_break_line_inside_a_paragraph_stays_text() -> None:
    """Only a line that starts a block is a break."""
    doc = _convert_adoc("First line.\n'''\nSecond line.\n")

    assert _texts(doc) == [("TextItem", "First line. ''' Second line.")]
