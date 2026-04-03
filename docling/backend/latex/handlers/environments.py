import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from typing import Any

from docling_core.types.doc import CodeLanguageLabel
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    GroupLabel,
    NodeItem,
    PictureMeta,
)
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode

from docling.backend.latex.constants import ENV_LIST, ENV_MATH, ENV_QUOTE, ENV_THEOREM

_log = logging.getLogger(__name__)
_TIKZ_END_PATTERN = re.compile(r"\\end\s*\{\s*tikzpicture\s*\}")


class EnvironmentHandlerMixin:
    if TYPE_CHECKING:

        def _process_nodes(
            self,
            nodes: "Any",
            doc: "Any",
            parent: "Any" = ...,
            formatting: "Any" = ...,
            text_label: "Any" = ...,
        ) -> None: ...
        def _clean_math(self, latex_str: str, env_name: str) -> str: ...
        def _parse_table(self, node: "Any") -> "Any": ...
        def _extract_verbatim_content(self, latex_str: str, env_name: str) -> str: ...
        def _extract_macro_arg(self, node: "Any") -> str: ...

    def _find_document_env(self, nodes, depth: int = 0):
        if nodes is None or depth > 10:
            return None
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                return node
            if hasattr(node, "nodelist") and node.nodelist:
                result = self._find_document_env(node.nodelist, depth + 1)
                if result:
                    return result
            if hasattr(node, "nodeargd") and node.nodeargd:
                argnlist = getattr(node.nodeargd, "argnlist", None)
                if argnlist:
                    for arg in argnlist:
                        if hasattr(arg, "nodelist") and arg.nodelist:
                            result = self._find_document_env(arg.nodelist, depth + 1)
                            if result:
                                return result
        return None

    def _process_environment(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        if node.envname == "document":
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname == "abstract":
            doc.add_heading(parent=parent, text="Abstract", level=1)
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname.replace("*", "") in ENV_MATH:
            math_text = self._clean_math(node.latex_verbatim(), node.envname)
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)

        elif node.envname == "math":
            math_text = self._clean_math(node.latex_verbatim(), node.envname)
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)

        elif node.envname == "subequations":
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname.replace("*", "") in ENV_THEOREM:
            env_title = node.envname.replace("*", "").capitalize()
            doc.add_text(
                parent=parent,
                label=DocItemLabel.TEXT,
                text=f"**{env_title}.**",
            )
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname == "proof":
            doc.add_text(
                parent=parent,
                label=DocItemLabel.TEXT,
                text="*Proof.*",
            )
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)
            body_latex = node.latex_verbatim() if node else ""
            if "\\qed" not in body_latex and "\\qedsymbol" not in body_latex:
                doc.add_text(
                    parent=parent,
                    label=DocItemLabel.TEXT,
                    text="\u25fb",
                )

        elif node.envname in ENV_QUOTE:
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname in ENV_LIST:
            self._process_list(node, doc, parent, formatting, text_label)

        elif node.envname == "tabular":
            table_data = self._parse_table(node)
            if table_data:
                doc.add_table(parent=parent, data=table_data)

        elif node.envname in ["table", "table*"]:
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname in ["figure", "figure*"]:
            self._process_figure(node, doc, parent, formatting, text_label)

        elif node.envname == "tikzpicture":
            self._process_tikzpicture(node, doc, parent, formatting, text_label)

        elif node.envname in ["verbatim", "lstlisting", "minted"]:
            code_text = self._extract_verbatim_content(
                node.latex_verbatim(), node.envname
            )
            doc.add_text(parent=parent, label=DocItemLabel.CODE, text=code_text)

        elif node.envname == "thebibliography":
            doc.add_heading(parent=parent, text="References", level=1)
            self._process_bibliography(node, doc, parent, formatting)

        elif node.envname in ["filecontents", "filecontents*"]:
            pass

        else:
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

    def _process_tikzpicture(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        tikz_raw = self._extract_tikzpicture_atomic(node)
        if tikz_raw is None:
            _log.warning(
                "tikzpicture extraction failed, using recursive environment fallback"
            )
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)
            return

        if "code" not in PictureMeta.model_fields:
            if getattr(parent, "name", None) == "figure":
                figure_group = parent
            else:
                figure_group = doc.add_group(
                    parent=parent, name="figure", label=GroupLabel.SECTION
                )
            doc.add_text(
                parent=figure_group,
                label=DocItemLabel.CODE,
                text=tikz_raw,
                formatting=formatting,
            )
            return

        latex_language = next(
            (
                language
                for language in CodeLanguageLabel
                if str(language.value).strip().lower() == "latex"
            ),
            CodeLanguageLabel.UNKNOWN,
        )
        pic = doc.add_picture(parent=parent)
        pic.meta = PictureMeta(
            code={
                "text": tikz_raw,
                "language": latex_language,
            }
        )

    def _extract_tikzpicture_atomic(self, node: LatexEnvironmentNode) -> str | None:
        raw = node.latex_verbatim()
        if _TIKZ_END_PATTERN.search(raw) is None:
            return None
        if not self._validate_tikz_nodelist(node.nodelist, 0):
            return None
        return raw

    def _validate_tikz_nodelist(self, nodes, depth: int = 0) -> bool:
        if nodes is None:
            return True
        if depth > 50:
            return False

        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "tikzpicture":
                nested_raw = node.latex_verbatim()
                if _TIKZ_END_PATTERN.search(nested_raw) is None:
                    return False

            if hasattr(node, "nodelist") and node.nodelist is not None:
                if not self._validate_tikz_nodelist(node.nodelist, depth + 1):
                    return False

            if hasattr(node, "nodeargd") and node.nodeargd:
                argnlist = getattr(node.nodeargd, "argnlist", None)
                if argnlist:
                    for arg in argnlist:
                        if hasattr(arg, "nodelist") and arg.nodelist is not None:
                            if not self._validate_tikz_nodelist(
                                arg.nodelist, depth + 1
                            ):
                                return False

        return True

    def _process_figure(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        figure_group = doc.add_group(
            parent=parent, name="figure", label=GroupLabel.SECTION
        )
        self._process_nodes(node.nodelist, doc, figure_group, formatting, text_label)

    def _process_list(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        list_group = doc.add_group(parent=parent, name="list", label=GroupLabel.LIST)

        items = []
        current_item: list = []

        if node.nodelist is not None:
            for n in node.nodelist:
                if isinstance(n, LatexMacroNode) and n.macroname == "item":
                    if current_item:
                        items.append(current_item)
                    current_item = []

                    if n.nodeargd and n.nodeargd.argnlist:
                        current_item.append(n)
                else:
                    current_item.append(n)

        if current_item:
            items.append(current_item)

        for item_nodes in items:
            self._process_nodes(
                item_nodes,
                doc,
                list_group,
                formatting,
                text_label=DocItemLabel.LIST_ITEM,
            )

    def _process_bibliography(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
    ):
        bib_group = doc.add_group(
            parent=parent, name="bibliography", label=GroupLabel.LIST
        )

        items = []
        current_item: list = []
        current_key = ""

        if node.nodelist is not None:
            for n in node.nodelist:
                if isinstance(n, LatexMacroNode) and n.macroname == "bibitem":
                    if current_item:
                        items.append((current_key, current_item))
                    current_item = []
                    current_key = self._extract_macro_arg(n)
                else:
                    current_item.append(n)

        if current_item:
            items.append((current_key, current_item))

        for key, item_nodes in items:
            if key:
                doc.add_text(
                    parent=bib_group,
                    label=DocItemLabel.LIST_ITEM,
                    text=f"[{key}] ",
                    formatting=formatting,
                )

            self._process_nodes(
                item_nodes,
                doc,
                bib_group,
                formatting,
                text_label=DocItemLabel.LIST_ITEM,
            )
