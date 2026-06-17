import re
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from typing import Any

from docling_core.types.doc.document import DocItemLabel, DoclingDocument, NodeItem
from pylatexenc.latexwalker import LatexMathNode

from docling.backend.latex.constants import ENV_MATH_CLEAN, ENV_MATH_DISPLAY_PREFIXES


class MathHandlerMixin:
    if TYPE_CHECKING:

        def _expand_macros(self, latex_str: str) -> str: ...

    def _process_math_node(
        self,
        node: LatexMathNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        text_buffer: List[str],
        flush_fn: Callable[[], None],
    ):
        is_display = getattr(node, "displaytype", None) == "display"

        if not is_display:
            math_verbatim = node.latex_verbatim()
            is_display = math_verbatim.startswith(ENV_MATH_DISPLAY_PREFIXES)

        if is_display:
            flush_fn()
            math_text = self._clean_math(node.latex_verbatim(), "display")
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)
        else:
            text_buffer.append(self._expand_macros(node.latex_verbatim()))

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

        return latex_str.strip()
