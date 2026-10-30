# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

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
from docling.backend.latex.handlers.math import InlineFormula
from docling.backend.latex.utils.latex_context import LATEX_CONTEXT_DB

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latex2text import LatexNodes2Text
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexGroupNode,
        LatexMacroNode,
        LatexMathNode,
        LatexSpecialsNode,
        LatexWalker,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


# A {...} group, allowing one level of nested braces.
_BRACED_GROUP = r"\{(?:[^{}]|\{[^{}]*\})*\}"
# A [...] optional argument whose braced values may contain brackets,
# as in [caption={A [b] c}].
_OPTIONAL_ARGUMENT = rf"\[(?:[^\[\]{{}}]|{_BRACED_GROUP})*\]"


def iter_child_nodelists(node: Any) -> Iterator[list]:
    """Yield the node lists of ``node`` and of its group arguments."""
    if isinstance(node, (LatexGroupNode, LatexEnvironmentNode, LatexMathNode)):
        if node.nodelist is not None:
            yield node.nodelist
    if isinstance(node, (LatexMacroNode, LatexEnvironmentNode, LatexSpecialsNode)):
        if node.nodeargd and node.nodeargd.argnlist:
            for arg in node.nodeargd.argnlist:
                if isinstance(arg, LatexGroupNode) and arg.nodelist is not None:
                    yield arg.nodelist


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
        ``None`` means the node is not one of these: another macro, or a name
        the document redefines.
        """
        name = node.macroname
        if name in self._custom_macros:
            return None
        if name in MACROS_ACCENTS:
            args = node.nodeargd.argnlist if node.nodeargd else []
            arg = next((a for a in args if a is not None), None)
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
            walker = LatexWalker(
                text, tolerant_parsing=True, latex_context=LATEX_CONTEXT_DB
            )
            nodes, _, _ = walker.get_latex_nodes()
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
        text_buffer: list[str | InlineFormula],
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
        text_buffer: list[str | InlineFormula],
        flush_fn: Callable[[], None],
    ):
        if (
            node.nodelist
            and self._is_text_only_group(node)
            and self._contains_math(node.nodelist)
        ):
            self._append_inline_nodes(
                node.nodelist,
                doc,
                parent,
                formatting,
                text_label,
                text_buffer,
                flush_fn,
            )
        elif node.nodelist and self._is_text_only_group(node):
            group_text = self._nodes_to_text(node.nodelist)
            if group_text:
                text_buffer.append(group_text)
        elif node.nodelist:
            flush_fn()
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

    def _contains_math(self, nodes: list) -> bool:
        for node in nodes:
            if isinstance(node, LatexMathNode):
                return True
            if isinstance(node, LatexEnvironmentNode) and node.envname == "math":
                return True
            if isinstance(node, LatexMacroNode):
                if node.macroname == "ensuremath":
                    return True
                if node.macroname in self._custom_macros:
                    expansion = self._custom_macros[node.macroname]
                    if r"\ensuremath" in expansion or "$" in expansion:
                        return True
            for children in iter_child_nodelists(node):
                if self._contains_math(children):
                    return True
        return False

    def _append_inline_nodes(
        self,
        nodes: list,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: list[str | InlineFormula],
        flush_fn: Callable[[], None],
    ) -> None:
        index = 0
        while index < len(nodes):
            node = nodes[index]
            consumed = 0
            if isinstance(node, LatexCharsNode):
                self._process_chars_node(
                    node, doc, parent, formatting, text_label, text_buffer, flush_fn
                )
            elif isinstance(node, LatexMathNode):
                self._process_math_node(node, doc, parent, text_buffer, flush_fn)
            elif isinstance(node, LatexEnvironmentNode) and node.envname == "math":
                math_text = self._clean_math(node.latex_verbatim(), node.envname)
                if math_text:
                    text_buffer.append(InlineFormula(math_text))
            elif isinstance(node, LatexGroupNode):
                self._append_inline_nodes(
                    node.nodelist or [],
                    doc,
                    parent,
                    formatting,
                    text_label,
                    text_buffer,
                    flush_fn,
                )
            elif isinstance(node, LatexMacroNode):
                inline_args = self._inline_math_argument_nodes(node)
                if inline_args is not None:
                    self._append_inline_nodes(
                        inline_args,
                        doc,
                        parent,
                        formatting,
                        text_label,
                        text_buffer,
                        flush_fn,
                    )
                else:
                    consumed = self._process_macro_node_inline(
                        node,
                        doc,
                        parent,
                        formatting,
                        text_label,
                        text_buffer,
                        flush_fn,
                        nodes[index + 1 :],
                    )
            index += 1 + consumed

    def _inline_math_argument_nodes(self, node: LatexMacroNode) -> list | None:
        if node.macroname not in (
            MACROS_TEXT_FORMATTING | MACROS_TEXT_STYLE | MACROS_COLOR_INLINE
        ):
            return None
        if not node.nodeargd or not node.nodeargd.argnlist:
            return None
        argument = node.nodeargd.argnlist[-1]
        if not isinstance(argument, LatexGroupNode) or not argument.nodelist:
            return None
        if self._contains_math(argument.nodelist):
            return argument.nodelist
        return None

    def _extract_verbatim_content(self, latex_str: str, env_name: str) -> str:
        name = re.escape(env_name)
        # minted takes the code's language as a mandatory {argument}.
        language = _BRACED_GROUP if env_name == "minted" else ""
        pattern = (
            rf"\\begin\{{{name}\}}(?:{_OPTIONAL_ARGUMENT})?{language}"
            rf"(.*?)\\end\{{{name}\}}"
        )
        match = re.search(pattern, latex_str, re.DOTALL)
        if match:
            return re.sub(r"\A(?:[ \t]*\r?\n)+", "", match.group(1)).rstrip()
        return latex_str

    def _item_term(self, node: LatexMacroNode) -> str:
        """Return the text of the ``\\item[...]`` term, or ``""`` if it has none."""
        if node.nodeargd and node.nodeargd.argnlist:
            arg = node.nodeargd.argnlist[0]
            if isinstance(arg, LatexGroupNode):
                return self._nodes_to_text(arg.nodelist).strip()
        return ""

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
                if isinstance(text_arg, LatexGroupNode):
                    return (self._nodes_to_text(text_arg.nodelist), consumed)
            return ("", consumed)
        if node.macroname in MACROS_CITATION:
            return (node.latex_verbatim(), consumed)
        if node.macroname == "\\":
            return ("\n", consumed)
        if node.macroname in ["~"]:
            return (" ", consumed)
        if node.macroname == "item":
            term = self._item_term(node)
            return (f"{term}: " if term else "", consumed)
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
                    if isinstance(arg, LatexGroupNode):
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
