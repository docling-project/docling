# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from docling_core.types.doc import CodeItem, DoclingDocument

from docling.datamodel.backend_options import HTMLBackendOptions, MarkdownBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    MarkdownFormatOption,
)


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        (
            "```python\n\n    value = '<a>&amp;</a>'  \n\n```\n",
            "\n    value = '<a>&amp;</a>'  \n\n",
        ),
        (
            '```html\n<a href="https://example.com">&amp;</a>\n```\n',
            '<a href="https://example.com">&amp;</a>\n',
        ),
        (
            "```python\n\n    value = '<a>&amp;</a>'  \n\n```\n\n"
            "<p>Embedded HTML triggers a second conversion.</p>\n",
            "\n    value = '<a>&amp;</a>'  \n\n",
        ),
        (
            "<pre><code>\n    value = '&lt;a&gt;&amp;amp;&lt;/a&gt;'  \n\n"
            "</code></pre>\n",
            "\n    value = '<a>&amp;</a>'  \n\n",
        ),
        (
            "    first  \n    second\n",
            "first  \nsecond\n",
        ),
    ],
)
def test_github_code_survives_dclx(
    markdown: str, expected: str, tmp_path: Path
) -> None:
    document = (
        DocumentConverter(
            allowed_formats=[InputFormat.MD],
            format_options={
                InputFormat.MD: MarkdownFormatOption(
                    backend_options=MarkdownBackendOptions(
                        preserve_code_whitespace=True
                    )
                )
            },
        )
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )

    archive = tmp_path / "conversation.dclx"
    document.save_as_doclang_archive(archive)
    loaded = DoclingDocument.load_from_doclang_archive(archive)

    for candidate in (document, loaded):
        snippets = [item.text for item in candidate.texts if isinstance(item, CodeItem)]
        assert snippets == [expected]


@pytest.mark.parametrize("preserve", [False, True])
def test_html_code_whitespace_option(preserve: bool, tmp_path: Path) -> None:
    source = "<pre><code>\n    value = '&lt;tag&gt;'  \n\n</code></pre>"
    document = (
        DocumentConverter(
            allowed_formats=[InputFormat.HTML],
            format_options={
                InputFormat.HTML: HTMLFormatOption(
                    backend_options=HTMLBackendOptions(
                        preserve_code_whitespace=preserve
                    )
                )
            },
        )
        .convert_string(source, format=InputFormat.HTML)
        .document
    )

    archive = tmp_path / "html-code.dclx"
    document.save_as_doclang_archive(archive)
    loaded = DoclingDocument.load_from_doclang_archive(archive)

    expected = "\n    value = '<tag>'  \n\n" if preserve else "    value = '<tag>'"
    for candidate in (document, loaded):
        snippets = [item.text for item in candidate.texts if isinstance(item, CodeItem)]
        assert snippets == [expected]


@pytest.mark.parametrize(
    "markdown",
    [
        "[target](https://example.com/?a=1&b=2#part)\n",
        '<p><a href="https://example.com/?a=1&amp;b=2#part">target</a></p>\n',
        "| Name | Description |\n"
        "| --- | --- |\n"
        "| [target](https://example.com/?a=1&b=2#part) | Bug label |\n",
    ],
)
def test_github_link_survives_dclx(markdown: str, tmp_path: Path) -> None:
    document = (
        DocumentConverter(allowed_formats=[InputFormat.MD])
        .convert_string(markdown, format=InputFormat.MD)
        .document
    )

    archive = tmp_path / "conversation.dclx"
    document.save_as_doclang_archive(archive)
    loaded = DoclingDocument.load_from_doclang_archive(archive)

    for candidate in (document, loaded):
        links = [
            (item.text, str(item.hyperlink))
            for item in candidate.texts
            if item.hyperlink is not None
        ]
        assert links == [("target", "https://example.com/?a=1&b=2#part")]
