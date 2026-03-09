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
    MACROS_ESCAPED,
    MACROS_STRUCTURAL,
    MACROS_TEXT_FORMATTING,
)


class TextHelperMixin:
    if TYPE_CHECKING:
        _custom_macros: dict[str, str]

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

    def _process_chars_node(
        self,
        node: LatexCharsNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem],
        formatting: Optional[Formatting],
        text_label: Optional[DocItemLabel],
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
        parent: Optional[NodeItem],
        formatting: Optional[Formatting],
        text_label: Optional[DocItemLabel],
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

    def _nodes_to_text(self, nodes) -> str:
        text_parts = []

        for node in nodes:
            if isinstance(node, LatexCharsNode):
                text_parts.append(node.chars)

            elif isinstance(node, LatexGroupNode):
                text_parts.append(self._nodes_to_text(node.nodelist))

            elif isinstance(node, LatexMacroNode):
                if node.macroname in MACROS_TEXT_FORMATTING:
                    text = self._extract_macro_arg(node)
                    if text:
                        text_parts.append(text)
                elif node.macroname in MACROS_CITATION:
                    text_parts.append(node.latex_verbatim())
                elif node.macroname == "\\":
                    text_parts.append("\n")
                elif node.macroname in ["~"]:
                    text_parts.append(" ")
                elif node.macroname == "item":
                    if node.nodeargd and node.nodeargd.argnlist:
                        arg = node.nodeargd.argnlist[0]
                        if arg:
                            opt_text = arg.latex_verbatim().strip("[] ")
                            text_parts.append(f"{opt_text}: ")
                elif node.macroname in MACROS_ESCAPED:
                    text_parts.append(node.macroname)
                elif node.macroname in self._custom_macros:
                    expansion = self._custom_macros[node.macroname]
                    text_parts.append(expansion)
                else:
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
                    if arg_parts:
                        text_parts.append(" ".join(arg_parts))

            elif isinstance(node, LatexMathNode):
                text_parts.append(self._expand_macros(node.latex_verbatim()))

            elif isinstance(node, LatexEnvironmentNode):
                if node.envname in ["equation", "align", "gather"]:
                    text_parts.append(node.latex_verbatim())
                else:
                    text_parts.append(self._nodes_to_text(node.nodelist))

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
