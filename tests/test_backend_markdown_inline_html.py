# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import CodeItem, DocItemLabel, ListItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

pytestmark = pytest.mark.cross_platform


@pytest.mark.parametrize(
    "tag",
    ["<br>", "<br/>", "<br />", "<BR>", '<br class="break">', '<br title="a > b">'],
)
def test_inline_html_breaks_in_paragraph_heading_and_list(tag: str) -> None:
    markdown = (
        f"# Release{tag}notes\n\n"
        f"First line{tag}Second line\n\n"
        f"- Step one{tag}then step two\n"
        "- Next item\n\n"
        "Hard\\\nbreak\n"
    )
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )

    # The heading retains ownership of its inline text. Plain paragraphs and
    # list items use the same document structure as Markdown hard breaks.
    heading = doc.texts[0]
    assert heading.label == DocItemLabel.TITLE
    assert heading.text == "Release\nnotes"
    assert heading.parent is not None
    assert heading.parent.resolve(doc) is doc.body
    assert [(item.label, item.text) for item in doc.texts[1:]] == [
        (DocItemLabel.TEXT, "First line\nSecond line"),
        (DocItemLabel.LIST_ITEM, "Step one\nthen step two"),
        (DocItemLabel.LIST_ITEM, "Next item"),
        (DocItemLabel.TEXT, "Hard\nbreak"),
    ]
    items = [item for item in doc.texts if isinstance(item, ListItem)]
    assert items[0].parent == items[1].parent
    exported = doc.export_to_markdown()
    # The Markdown serializer normalizes heading newlines to spaces.
    assert exported.startswith("# Release notes\n\nFirst line  \nSecond line")
    assert "- Next item" in exported


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("First<br>**second**", "First  \n**second**"),
        ("**First**<br>second", "**First**  \nsecond"),
        ("First<br>`second`", "First  \n`second`"),
        ("`First`<br>second", "`First`  \nsecond"),
        (
            "First<br>[second](https://example.com/docs)",
            "First  \n[second](https://example.com/docs)",
        ),
        ("First<br><br>second", "First\n\nsecond"),
    ],
)
def test_inline_html_breaks_preserve_inline_boundaries(
    markdown: str, expected: str
) -> None:
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )
    assert doc.export_to_markdown() == expected


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("First\\<br>second", "First<br>second"),
        ("First&lt;br&gt;second", "First<br>second"),
        ("First<bravo>second", "Firstsecond"),
    ],
)
def test_inline_html_break_detection_does_not_match_literal_text(
    markdown: str, expected: str
) -> None:
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )
    assert "".join(item.text for item in doc.texts) == expected


@pytest.mark.parametrize("ending", ["<br>", "<br><br>", "<br>\n"])
def test_inline_html_breaks_stay_in_their_paragraph(ending: str) -> None:
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(f"First{ending}\n\nSecond", format=InputFormat.MD)
        .document
    )
    assert [item.text for item in doc.texts] == ["First", "Second"]
    assert [ref.resolve(doc) for ref in doc.body.children] == doc.texts
    assert doc.export_to_markdown() == "First\n\nSecond"


def test_inline_html_breaks_preserve_code_and_table_cells() -> None:
    markdown = (
        "`First<br>second`\n\n"
        "```html\nFirst<br>second\n```\n\n"
        "| Header |\n|---|\n| First<br>second |\n"
    )
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )
    assert [item.text for item in doc.texts if isinstance(item, CodeItem)] == [
        "First<br>second",
        "First<br>second",
    ]
    assert doc.tables[0].data.table_cells[-1].text == "First<br>second"


@pytest.mark.parametrize("title", ["", ' "Caption"'])
def test_inline_html_breaks_preserve_image_order_and_caption(title: str) -> None:
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(
            f"First<br>![](missing.png{title})<br>Second", format=InputFormat.MD
        )
        .document
    )
    group = doc.body.children[0].resolve(doc)
    children = [ref.resolve(doc) for ref in group.children]
    assert [child.label for child in children] == [
        DocItemLabel.TEXT,
        DocItemLabel.TEXT,
        DocItemLabel.PICTURE,
        DocItemLabel.TEXT,
    ]
    assert [child.text for child in children if isinstance(child, TextItem)] == [
        "First",
        "\n",
        "\nSecond",
    ]
    assert [ref.resolve(doc).text for ref in doc.pictures[0].captions] == (
        ["Caption"] if title else []
    )
    for ref in doc.pictures[0].captions:
        caption = ref.resolve(doc)
        assert caption.parent is not None
        assert caption.parent.resolve(doc) is doc.pictures[0]
    exported = doc.export_to_markdown()
    assert exported.index("First") < exported.index("<!-- image -->")
    assert exported.index("<!-- image -->") < exported.index("Second")


def test_inline_html_break_preserves_formatted_heading_content() -> None:
    doc = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string("# First<br>**second**\n\nAfter", format=InputFormat.MD)
        .document
    )
    heading = doc.texts[0]
    assert heading.label == DocItemLabel.TITLE
    group = heading.children[0].resolve(doc)
    runs = [ref.resolve(doc) for ref in group.children]
    assert [run.text for run in runs] == ["First", "\nsecond"]
    assert runs[1].formatting is not None
    assert runs[1].formatting.bold
    assert doc.body.children[-1].resolve(doc).text == "After"
