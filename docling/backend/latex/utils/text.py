import re
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from typing import Any

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    NodeItem,
)
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
)

from docling.backend.latex.constants import (
    MACROS_CITATION,
    MACROS_COLOR_INLINE,
    MACROS_ESCAPED,
    MACROS_IGNORED,
    MACROS_SPACING,
    MACROS_STRUCTURAL,
    MACROS_TEXT_FORMATTING,
    MACROS_TEXT_STYLE,
)


class TextHelperMixin:
    if TYPE_CHECKING:
        _custom_macros: dict[str, str]
        _custom_macro_num_args: dict[str, int]

        def _process_nodes(
            self,
            nodes: "Any",
            doc: "Any",
            parent: "Any" = ...,
            formatting: "Any" = ...,
            text_label: "Any" = ...,
        ) -> None: ...
        def _extract_macro_arg(self, node: "Any") -> str: ...
        def _expand_macros(self, latex_str: str) -> str: ...
        def _expand_custom_macro_invocation(
            self, node: "Any", following_nodes: "Any"
        ) -> tuple[str, int]: ...
        def _parse_latex_fragment_to_text(self, latex_fragment: str) -> str: ...

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

            first_part = parts[0].strip()
            if first_part:
                text_buffer.append(first_part)

            flush_fn()

            for part in parts[1:]:
                part_stripped = part.strip()
                if part_stripped:
                    doc.add_text(
                        parent=parent,
                        label=text_label or DocItemLabel.PARAGRAPH,
                        text=part_stripped,
                        formatting=formatting,
                    )
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
