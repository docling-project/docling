# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, List, Optional

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    NodeItem,
)

from docling.backend.latex.constants import (
    MACROS_ACCENTS,
    MACROS_CITATION,
    MACROS_COLOR_INLINE,
    MACROS_ESCAPED,
    MACROS_IGNORED,
    MACROS_LETTERS,
    MACROS_SPACING,
    MACROS_STRUCTURAL,
    MACROS_TEXT_FORMATTING,
    MACROS_TEXT_STYLE,
)
from docling.backend.latex.utils.latex_context import LATEX_CONTEXT_DB

if TYPE_CHECKING:
    from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latex2text import LatexNodes2Text
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexGroupNode,
        LatexMacroNode,
        LatexMathNode,
        LatexWalker,
        LatexWalkerParseError,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


class TextHelperMixin:
    if TYPE_CHECKING:
        _custom_macros: dict[str, str]
        _custom_macro_num_args: dict[str, int]

        def _process_nodes(
            self,
            nodes: Any,
            doc: Any,
            parent: Any = ...,
            formatting: Any = ...,
            text_label: Any = ...,
        ) -> None: ...
        def _extract_macro_arg(self, node: Any) -> str: ...
        def _expand_macros(self, latex_str: str) -> str: ...
        def _expand_custom_macro_invocation(
            self, node: Any, following_nodes: Any
        ) -> tuple[str, int]: ...
        def _parse_latex_fragment_to_text(self, latex_fragment: str) -> str: ...

    def _char_macro_to_text(self, node: LatexMacroNode) -> str | None:
        """Return the Unicode text of an accent or letter macro.

        ``\\'e`` gives ``é``, ``\\~n`` gives ``ñ`` and ``\\ss`` gives ``ß``.
        ``None`` means the node is not one of these: another macro, an accent
        that was parsed without an argument, or a name the document redefines.
        """
        name = node.macroname
        if name in self._custom_macros:
            return None
        if name in MACROS_ACCENTS:
            args = node.nodeargd.argnlist if node.nodeargd else []
            arg = next((a for a in args if a is not None), None)
            if arg is None:
                return None
            if self._uses_custom_macro(arg):
                text = self._accent_custom_macro_arg(name, arg)
            else:
                text = LatexNodes2Text().nodelist_to_text([node])
        elif name in MACROS_LETTERS:
            text = LatexNodes2Text().nodelist_to_text([node])
        else:
            return None
        # \~{} and \^{} are the usual way to typeset a literal ~ or ^, which
        # latex2text renders as an empty string.
        if not text and name in ("~", "^"):
            return name
        return text

    def _accent_custom_macro_arg(self, name: str, arg) -> str:
        """Apply accent ``name`` to an argument that uses document macros.

        latex2text does not know the document's macros and would drop them,
        so expand them first, one level at a time to follow chains such as
        ``\\newcommand{\\vowel}{\\letter}``: ``\\'{\\vowel}`` -> ``\\'{\\letter}``
        -> ``\\'{e}``. An argument that never resolves, e.g. a macro defined
        in terms of itself, is kept without the accent.
        """
        nodes = [arg]
        text = ""
        for _ in range(10):
            text = self._nodes_to_text(nodes)
            try:
                walker = LatexWalker(
                    text, tolerant_parsing=True, latex_context=LATEX_CONTEXT_DB
                )
                nodes, _, _ = walker.get_latex_nodes()
            except LatexWalkerParseError:
                return text
            if not any(self._uses_custom_macro(n) for n in nodes):
                base = "".join(n.latex_verbatim() for n in nodes)
                return LatexNodes2Text().latex_to_text(f"\\{name}{{{base}}}")
        return text

    def _uses_custom_macro(self, node) -> bool:
        """Whether ``node`` or anything nested in it is a document macro."""
        if isinstance(node, LatexMacroNode):
            if node.macroname in self._custom_macros:
                return True
            args = node.nodeargd.argnlist if node.nodeargd else []
            return any(self._uses_custom_macro(a) for a in args if a is not None)
        if isinstance(node, LatexGroupNode):
            return any(self._uses_custom_macro(n) for n in node.nodelist or [])
        return False

    def _process_chars_node(
        self,
        node: LatexCharsNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: List[str],
        flush_fn: Callable[[], None],
    ):
        text = node.chars

        if "\n\n" in text:
            parts = text.split("\n\n")

            text_buffer.append(parts[0])

            flush_fn()

            for part in parts[1:-1]:
                part_stripped = part.strip()
                if part_stripped:
                    doc.add_text(
                        parent=parent,
                        label=text_label or DocItemLabel.PARAGRAPH,
                        text=part_stripped,
                        formatting=formatting,
                    )

            text_buffer.append(parts[-1])
        else:
            text_buffer.append(text)

    def _process_group_node(
        self,
        node: LatexGroupNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: List[str],
        flush_fn: Callable[[], None],
    ):
        if node.nodelist and self._is_text_only_group(node):
            group_text = self._nodes_to_text(node.nodelist)
            if group_text:
                text_buffer.append(group_text)
        elif node.nodelist:
            flush_fn()
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

    def _extract_verbatim_content(self, latex_str: str, env_name: str) -> str:
        pattern = rf"\\begin\{{{re.escape(env_name)}\}}(?:\[.*?\])?(.*?)\\end\{{{re.escape(env_name)}\}}"
        match = re.search(pattern, latex_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return latex_str

    def _macro_node_to_text(self, node: LatexMacroNode, following_nodes) -> tuple:
        """Return ``(text, consumed_following)`` for a single macro node."""
        consumed = 0
        char_text = self._char_macro_to_text(node)
        if char_text is not None:
            return (char_text, consumed)
        if node.macroname in (MACROS_TEXT_FORMATTING | MACROS_TEXT_STYLE):
            text = self._extract_macro_arg(node)
            return (text or "", consumed)
        if node.macroname in MACROS_COLOR_INLINE:
            if node.nodeargd and node.nodeargd.argnlist:
                text_arg = node.nodeargd.argnlist[-1]
                if text_arg is not None and hasattr(text_arg, "nodelist"):
                    return (self._nodes_to_text(text_arg.nodelist), consumed)
            return ("", consumed)
        if node.macroname in MACROS_CITATION:
            return (node.latex_verbatim(), consumed)
        if node.macroname == "\\":
            return ("\n", consumed)
        if node.macroname in ["~"]:
            return (" ", consumed)
        if node.macroname == "item":
            if node.nodeargd and node.nodeargd.argnlist:
                arg = node.nodeargd.argnlist[0]
                if arg:
                    opt_text = arg.latex_verbatim().strip("[] ")
                    return (f"{opt_text}: ", consumed)
            return ("", consumed)
        if node.macroname in MACROS_ESCAPED:
            return (node.macroname, consumed)
        if node.macroname in self._custom_macros:
            expansion, consumed = self._expand_custom_macro_invocation(
                node, following_nodes
            )
            if self._custom_macro_num_args.get(node.macroname, 0) > 0:
                return (self._parse_latex_fragment_to_text(expansion), consumed)
            return (expansion, consumed)
        if node.macroname in MACROS_SPACING or node.macroname in MACROS_IGNORED:
            return ("", consumed)
        arg_parts = []
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in node.nodeargd.argnlist:
                if arg is not None:
                    if hasattr(arg, "nodelist"):
                        text = self._nodes_to_text(arg.nodelist)
                        if text:
                            arg_parts.append(text)
                    else:
                        text = arg.latex_verbatim().strip("{} ")
                        if text:
                            arg_parts.append(text)
        return (" ".join(arg_parts), consumed)

    def _nodes_to_text(self, nodes) -> str:
        text_parts = []

        idx = 0
        while idx < len(nodes):
            node = nodes[idx]
            consumed_following = 0
            if isinstance(node, LatexCharsNode):
                text_parts.append(node.chars)
            elif isinstance(node, LatexGroupNode):
                text_parts.append(self._nodes_to_text(node.nodelist))
            elif isinstance(node, LatexMacroNode):
                text, consumed_following = self._macro_node_to_text(
                    node, nodes[idx + 1 :]
                )
                if text:
                    text_parts.append(text)
            elif isinstance(node, LatexMathNode):
                text_parts.append(self._expand_macros(node.latex_verbatim()))
            elif isinstance(node, LatexEnvironmentNode):
                if node.envname in ["equation", "align", "gather"]:
                    text_parts.append(node.latex_verbatim())
                else:
                    text_parts.append(self._nodes_to_text(node.nodelist))
            idx += 1 + consumed_following

        result = "".join(text_parts)
        result = re.sub(r" +", " ", result)
        result = re.sub(r"\n\n+", "\n\n", result)
        return result.strip()

    def _is_text_only_group(self, node: LatexGroupNode) -> bool:
        if not node.nodelist:
            return True

        for n in node.nodelist:
            if isinstance(n, LatexEnvironmentNode):
                return False
            elif isinstance(n, LatexMacroNode):
                if n.macroname in MACROS_STRUCTURAL:
                    return False
            elif isinstance(n, LatexGroupNode):
                if not self._is_text_only_group(n):
                    return False

        return True
