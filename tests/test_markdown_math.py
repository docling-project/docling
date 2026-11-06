# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from docling.utils.markdown_math import create_markdown_parser


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "Price $5-$10 and $x$; the answer is $5$ or $2+2$.",
            "Price $5-$10 and <math>x</math>; the answer is <math>5</math> or <math>2+2</math>.",
        ),
        (
            "about ~5 m and ~10 m, where $x$ is the length",
            "about ~5 m and ~10 m, where <math>x</math> is the length",
        ),
        (
            "| A | B |\n| --- | --- |\n| $x$ | y |",
            "| A | B |\n| --- | --- |\n| <math>x</math> | y |",
        ),
        ("> [!NOTE]\n> Value $x$.", "&gt; [!NOTE]\n&gt; Value <math>x</math>."),
        (
            "See www.example.com and $x$.",
            "See www.example.com and <math>x</math>.",
        ),
    ],
)
def test_inline_math_preserves_native_text_syntax(source: str, expected: str) -> None:
    assert create_markdown_parser(inline_only=True).convert(source) == (
        f"<p>{expected}</p>\n"
    )


def test_full_markdown_still_parses_tables_with_math() -> None:
    rendered = create_markdown_parser().convert("| A | B |\n| --- | --- |\n| $x$ | y |")
    assert "<table>" in rendered
    assert "<td><math>x</math></td>" in rendered


@pytest.mark.parametrize("inline_only", [False, True])
def test_cached_parser_checks_optional_dependency(
    inline_only: bool, monkeypatch
) -> None:
    parser = create_markdown_parser(inline_only=inline_only)
    assert "<math>x</math>" in parser.convert("$x$")
    monkeypatch.setitem(sys.modules, "marko", None)
    with pytest.raises(ImportError, match="format-markdown"):
        create_markdown_parser(inline_only=inline_only)


def test_reused_parsers_do_not_leak_document_state() -> None:
    full = create_markdown_parser()
    inline = create_markdown_parser(inline_only=True)
    assert '<a href="https://example.com">name</a>' in full.convert(
        "[name][ref] and $x$\n\n[ref]: https://example.com"
    )
    assert inline.convert("# Next $y$") == "<p># Next <math>y</math></p>\n"
    assert full.convert("[name][ref] and $z$") == (
        "<p>[name][ref] and <math>z</math></p>\n"
    )


def test_concurrent_conversions_keep_references_and_math_in_their_document() -> None:
    def convert(index: int) -> str:
        parser = create_markdown_parser()
        return parser.convert(
            f"[name][ref] and $x_{index}$\n\n[ref]: https://example.com/{index}"
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        rendered = list(executor.map(convert, range(32)))
    assert rendered == [
        f'<p><a href="https://example.com/{index}">name</a> and <math>x_{index}</math></p>\n'
        for index in range(32)
    ]


def test_math_extension_composes_with_markdown_and_keeps_parser_modes_isolated() -> (
    None
):
    import marko

    from docling.utils.markdown_math import create_math_extension

    source = r"# $x_i$ and `\(code\)`; $5-$10"
    expected = "<h1><math>x_i</math> and <code>\\(code\\)</code>; $5-$10</h1>\n"
    assert (
        marko.Markdown(extensions=[create_math_extension()]).convert(source) == expected
    )

    def convert(index: int) -> str:
        inline = index % 2 == 0
        return create_markdown_parser(inline_only=inline).convert(source)

    with ThreadPoolExecutor(max_workers=4) as executor:
        rendered = list(executor.map(convert, range(16)))
    assert rendered == [
        "<p># <math>x_i</math> and <code>\\(code\\)</code>; $5-$10</p>\n"
        if index % 2 == 0
        else expected
        for index in range(16)
    ]
