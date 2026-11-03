# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import CodeItem, DoclingDocument, GroupItem, RichTableCell

from docling.datamodel.backend_options import MarkdownBackendOptions
from tests.test_backend_markdown import _convert_markdown, _png_data_uri

pytestmark = pytest.mark.cross_platform


_HTML_BLOCK_BODY = """\
```cpp
#include <stdio.h>
Dict<str> d;
printf("<div>");
```

```bash
conda create -n <CONDA_NAME>
conda activate <CONDA_NAME>
```

Use `<T>` here.

| a | b |
|---|---|
| x<y> | 2 |

See [one](https://a.example) and **bold**
See [two](https://b.example) and *italic*

5. first
6. second
"""


def _items(doc: DoclingDocument) -> list[tuple[str, str]]:
    return [(str(t.label), t.text) for t in doc.texts]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("", ""),
        ("<!-- c -->\n\n", ""),
        ("", "\n<!-- c -->\n"),
        ("<!-- c -->\n\n", "\n<!-- c -->\n\n<br>\n"),
    ],
    ids=["no_html_block", "leading", "trailing", "both"],
)
def test_html_comment_does_not_change_surrounding_content(before: str, after: str):
    """An HTML block leaves the Markdown content around it unchanged."""
    options = MarkdownBackendOptions()
    expected = _convert_markdown(_HTML_BLOCK_BODY, options)
    doc = _convert_markdown(f"{before}{_HTML_BLOCK_BODY}{after}", options)

    codes = [t.text for t in doc.texts if isinstance(t, CodeItem)]
    assert codes == [
        '#include <stdio.h>\nDict<str> d;\nprintf("<div>");',
        "conda create -n <CONDA_NAME>\nconda activate <CONDA_NAME>",
        "<T>",
    ]
    assert [c.text for c in doc.tables[0].data.table_cells] == ["a", "b", "x<y>", "2"]
    assert [t.marker for t in doc.texts if t.label == "list_item"] == ["5.", "6."]
    assert _items(doc) == _items(expected)
    assert doc.export_to_markdown() == expected.export_to_markdown()


def test_adjacent_paragraphs_stay_separate_around_html_block():
    markdown = (
        "<!-- c -->\n\n"
        "See [one](https://a.example) and **bold**\n\n"
        "See [two](https://b.example)\n"
    )
    doc = _convert_markdown(markdown, MarkdownBackendOptions())

    assert [
        [child.resolve(doc).text for child in group.children] for group in doc.groups
    ] == [["See", "one", "and", "bold"], ["See", "two"]]


def test_markdown_picture_survives_html_block():
    markdown = f"<!-- c -->\n\n![alt]({_png_data_uri(7, 5)})\n"
    doc = _convert_markdown(markdown, MarkdownBackendOptions(fetch_images=True))

    assert len(doc.pictures) == 1
    image = doc.pictures[0].get_image(doc)
    assert image is not None
    assert image.size == (7, 5)


def test_html_block_content_is_converted():
    markdown = (
        "# Title\n\n"
        "Intro `<T>`\n\n"
        "<h2>Section</h2>\n\n"
        "<table><tr><th>H1</th><th>H2</th></tr><tr><td>1</td><td>2</td></tr></table>\n\n"
        "<details><summary>More</summary>detail</details>\n\n"
        f'<p><img src="{_png_data_uri(3, 2)}" alt="logo"></p>\n\n'
        "<div>open <b>tag\n\n"
        "after\n"
    )
    doc = _convert_markdown(markdown, MarkdownBackendOptions(fetch_images=True))

    assert [(str(t.label), t.text) for t in doc.texts] == [
        ("title", "Title"),
        ("text", "Intro"),
        ("code", "<T>"),
        ("section_header", "Section"),
        ("text", "More"),
        ("text", "detail"),
        ("caption", "logo"),
        ("text", "open"),
        ("text", "tag"),
        ("text", "after"),
    ]
    assert [[c.text for c in row] for row in doc.tables[0].data.grid] == [
        ["H1", "H2"],
        ["1", "2"],
    ]
    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is not None


def test_html_block_inside_list_item():
    markdown = "- a\n  <div>block</div>\n- b\n\n  <!-- c -->\n- `<T>`\n"
    doc = _convert_markdown(markdown, MarkdownBackendOptions())

    assert _items(doc) == [
        ("list_item", "a"),
        ("text", "block"),
        ("list_item", "b"),
        ("list_item", "<T>"),
    ]
    assert len(doc.groups) == 1
    assert doc.groups[0].label == "list"


def test_html_block_references_stay_valid():
    """Items an HTML block refers to by reference still resolve after conversion."""
    markdown = (
        "first\n\nsecond\n\n"
        '<figure><img src="a.png"><figcaption>Cap</figcaption></figure>\n\n'
        "<table><tr><td><p>x</p><p>y</p></td><td>z</td></tr></table>\n"
    )
    doc = _convert_markdown(markdown, MarkdownBackendOptions())

    assert doc.validate_tree(doc.body)
    assert [c.resolve(doc).text for c in doc.pictures[0].captions] == ["Cap"]
    table = doc.tables[0]
    rich_cells = [c for c in table.data.table_cells if isinstance(c, RichTableCell)]
    assert len(rich_cells) == 1
    cell_group = rich_cells[0].ref.resolve(doc)
    assert isinstance(cell_group, GroupItem)
    assert cell_group.parent == table.get_ref()
    assert [c.resolve(doc).text for c in cell_group.children] == ["x", "y"]
