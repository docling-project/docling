# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Math-aware Markdown rendering for native VLM response adapters."""

from __future__ import annotations

import html
import re
from threading import local
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import marko

_MATH = re.compile(
    r"(?<!\\)(?:\$\$([\s\S]*?)(?<!\\)\$\$"
    # A dollar before a digit starts an amount, not a closing math delimiter.
    r"|(?<!\$)\$(?![\s$])([^\n$]*?)(?<![\\\s])\$(?![\d$])"
    r"|\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\])"
)


class _MarkdownParsers(local):
    def __init__(self) -> None:
        self.by_mode: dict[bool, marko.Markdown] = {}


_parsers = _MarkdownParsers()


def create_markdown_parser(*, inline_only: bool = False) -> marko.Markdown:
    """Render LaTeX as literal math nodes and keep code and currency literal.

    ``inline_only`` leaves the enclosing native block type in control instead
    of interpreting its text as new Markdown lists, headings, or tables.

    Reuse one parser per mode and thread: Marko instances are not thread-safe.
    """
    # Markdown is an optional dependency for slim installations.
    try:
        import marko
    except ImportError as exc:
        raise ImportError(
            "Markdown parsing requires `pip install 'docling-slim[format-markdown]'`."
        ) from exc
    from marko.helpers import MarkoExtension

    if inline_only in _parsers.by_mode:
        return _parsers.by_mode[inline_only]

    class Math(marko.inline.InlineElement):
        children: str
        pattern = _MATH
        priority = 8
        parse_group = 0

    class CodeSpan(marko.inline.CodeSpan):
        override = True
        priority = 9

    class LiteralCode(marko.inline.InlineElement):
        children: str
        pattern = re.compile(r"<code\b[^>]*>[\s\S]*?</code>", re.IGNORECASE)
        priority = 10
        parse_group = 0

    class MathRenderer:
        def render_math(self, element: Math) -> str:
            match = _MATH.fullmatch(element.children)
            if match is None:
                return html.escape(element.children)
            latex = next(group for group in match.groups() if group is not None)
            display = match.group(1) is not None or match.group(4) is not None
            attrs = ' display="block"' if display else ""
            return f"<math{attrs}>{html.escape(latex.strip())}</math>"

        def render_literal_code(self, element: LiteralCode) -> str:
            return element.children

    class Paragraph(marko.block.Paragraph):
        override = True

        @classmethod
        def break_paragraph(
            cls, source: marko.source.Source, lazy: bool = False
        ) -> bool:
            return False

        @staticmethod
        def is_setext_heading(line: str) -> bool:
            return False

    class InlineParser(marko.Parser):
        def __init__(self) -> None:
            super().__init__()
            self.block_elements = {
                name: element
                for name, element in self.block_elements.items()
                if name in {"Document", "Paragraph", "BlankLine"}
            }

    markdown = marko.Markdown(
        parser=InlineParser if inline_only else marko.Parser,
        extensions=[
            # GFM adds block elements after InlineParser filters them, and its
            # strikethrough and autolinks change literal native block content.
            *([] if inline_only else ["gfm"]),
            MarkoExtension(
                elements=[
                    Math,
                    CodeSpan,
                    LiteralCode,
                    *([Paragraph] if inline_only else []),
                ],
                renderer_mixins=[MathRenderer],
            ),
        ],
    )
    _parsers.by_mode[inline_only] = markdown
    return markdown
