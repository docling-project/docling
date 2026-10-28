# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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
        (
            r"Cite \[9\] and \[Smith et al., 2020\]; \[x_1\].",
            'Cite [9] and [Smith et al., 2020]; <math display="block">x_1</math>.',
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
