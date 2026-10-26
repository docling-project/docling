# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from docling_core.types.doc.document import DocItemLabel, DoclingDocument, NodeItem

from docling.backend.latex.constants import ENV_MATH_CLEAN, ENV_MATH_DISPLAY_PREFIXES
from docling.backend.latex.utils.whitespace import collapse_source_layout_whitespace

if TYPE_CHECKING:
    from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import LatexMathNode
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


@dataclass(frozen=True)
class InlineFormula:
    text: str


class MathHandlerMixin:
    if TYPE_CHECKING:

        def _expand_macros(self, latex_str: str) -> str: ...

    def _process_math_node(
        self,
        node: LatexMathNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        text_buffer: list[str | InlineFormula],
        flush_fn: Callable[[], None],
    ):
        is_display = node.displaytype == "display"

        if not is_display:
            math_verbatim = node.latex_verbatim()
            is_display = math_verbatim.startswith(ENV_MATH_DISPLAY_PREFIXES)

        if is_display:
            flush_fn()
            math_text = self._clean_math(node.latex_verbatim(), "display")
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)
        else:
            math_text = self._clean_math(node.latex_verbatim(), "inline")
            if math_text:
                text_buffer.append(InlineFormula(math_text))

    def _clean_math(self, latex_str: str, env_name: str) -> str:
        if env_name in ENV_MATH_CLEAN:
            pattern = rf"\\begin\{{{re.escape(env_name)}\}}(.*?)\\end\{{{re.escape(env_name)}\}}"
            match = re.search(pattern, latex_str, re.DOTALL)
            if match:
                latex_str = match.group(1)

        latex_str = latex_str.strip()

        if latex_str.startswith("$$") and latex_str.endswith("$$"):
            latex_str = latex_str[2:-2]
        elif latex_str.startswith("$") and latex_str.endswith("$"):
            latex_str = latex_str[1:-1]
        elif latex_str.startswith("\\[") and latex_str.endswith("\\]"):
            latex_str = latex_str[2:-2]
        elif latex_str.startswith("\\(") and latex_str.endswith("\\)"):
            latex_str = latex_str[2:-2]

        latex_str = re.sub(r"\\label\{.*?\}", "", latex_str)
        latex_str = self._expand_macros(latex_str)

        latex_str = latex_str.strip()
        if env_name == "inline":
            latex_str = collapse_source_layout_whitespace(latex_str)
        return latex_str
