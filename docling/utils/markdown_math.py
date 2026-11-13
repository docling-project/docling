# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Math-aware Markdown rendering for native VLM response adapters."""

from __future__ import annotations

import html
import re
from copy import copy
from dataclasses import dataclass
from threading import local
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import marko
    from marko.helpers import MarkoExtension

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


@dataclass(eq=False)
class MathData:
    """Literal LaTeX and scope carried by a math-aware Markdown node."""

    children: str
    latex: str
    display: bool


@dataclass(eq=False)
class LiteralCodeData:
    """An HTML code span protected from math parsing."""

    children: str
    code: str


def _require_marko() -> ModuleType:
    # Markdown is an optional dependency for slim installations.
    try:
        import marko
    except ImportError as exc:
        raise ImportError(
            "Markdown parsing requires `pip install 'docling-slim[format-markdown]'`."
        ) from exc
    return marko


def _contains_math(children: list, *, display_only: bool = False) -> bool:
    marko = _require_marko()
    return any(
        (isinstance(child, MathData) and (not display_only or child.display))
        or (
            isinstance(child, marko.inline.InlineElement)
            and isinstance(child.children, list)
            and _contains_math(child.children, display_only=display_only)
        )
        for child in children
    )


def _split_math_runs(children: list) -> list[tuple[list, bool]]:
    """Split at display formulas while retaining their inline wrapper context."""
    marko = _require_marko()
    runs: list[tuple[list, bool]] = []
    for child in children:
        if isinstance(child, MathData):
            parts = [([child], child.display)]
        elif (
            isinstance(child, marko.inline.InlineElement)
            and isinstance(child.children, list)
            and _contains_math(child.children, display_only=True)
        ):
            parts = []
            for nodes, display in _split_math_runs(child.children):
                wrapper = copy(child)
                wrapper.children = nodes
                parts.append(([wrapper], display))
        else:
            parts = [([child], False)]
        for nodes, display in parts:
            if runs and not display and not runs[-1][1]:
                runs[-1][0].extend(nodes)
            else:
                runs.append((nodes, display))
    return runs


def _split_display_math(children: list) -> list:
    """Split a paragraph at display formulas without consuming adjacent blocks."""
    marko = _require_marko()
    blocks = []
    for nodes, display in _split_math_runs(children):
        if display:
            blocks.extend(nodes)
            continue
        while nodes and isinstance(nodes[-1], marko.inline.LineBreak):
            nodes.pop()
        while nodes and isinstance(nodes[0], marko.inline.LineBreak):
            nodes.pop(0)
        if nodes:
            paragraph = marko.block.Paragraph([])
            paragraph.children = nodes
            blocks.append(paragraph)
    return blocks


def create_math_extension() -> MarkoExtension:
    """Create the shared math extension without selecting a Markdown dialect."""
    marko = _require_marko()
    from marko.helpers import MarkoExtension

    class Math(MathData, marko.inline.InlineElement):
        pattern = _MATH
        priority = 8
        parse_group = 0

        def __init__(self, match: re.Match[str]) -> None:
            super().__init__(
                children=match.group(0),
                latex=next(
                    group for group in match.groups() if group is not None
                ).strip(),
                display=match.group(1) is not None or match.group(4) is not None,
            )

    class CodeSpan(marko.inline.CodeSpan):
        override = True
        priority = 9

    class LiteralCode(LiteralCodeData, marko.inline.InlineElement):
        pattern = re.compile(r"<code\b[^>]*>[\s\S]*?</code>", re.IGNORECASE)
        priority = 10
        parse_group = 0

        def __init__(self, match: re.Match[str]) -> None:
            # Load the HTML text extractor only when an HTML code span is used.
            from docling.utils.chandra_utils import _HTMLTreeParser

            raw = match.group(0)
            super().__init__(
                children=raw,
                code=_HTMLTreeParser(raw).root.text(),
            )

    class MathRenderer:
        def render_math(self, element: Math) -> str:
            attrs = ' display="block"' if element.display else ""
            return f"<math{attrs}>{html.escape(element.latex)}</math>"

        def render_literal_code(self, element: LiteralCode) -> str:
            return element.children

    return MarkoExtension(
        elements=[Math, CodeSpan, LiteralCode], renderer_mixins=[MathRenderer]
    )


def create_markdown_parser(*, inline_only: bool = False) -> marko.Markdown:
    """Render LaTeX as literal math nodes and keep code and currency literal.

    ``inline_only`` leaves the enclosing native block type in control instead
    of interpreting its text as new Markdown lists, headings, or tables.

    Reuse one parser per mode and thread: Marko instances are not thread-safe.
    """
    marko = _require_marko()
    if inline_only in _parsers.by_mode:
        return _parsers.by_mode[inline_only]
    extension = create_math_extension()
    from marko.helpers import MarkoExtension

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
            extension,
            *([MarkoExtension(elements=[Paragraph])] if inline_only else []),
        ],
    )
    _parsers.by_mode[inline_only] = markdown
    return markdown
