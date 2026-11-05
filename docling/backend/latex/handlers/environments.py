# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from docling_core.types.doc import CodeLanguageLabel
from docling_core.types.doc.document import (
    CodeMetaField,
    DocItemLabel,
    DoclingDocument,
    Formatting,
    GroupLabel,
    NodeItem,
    PictureMeta,
)

from docling.backend.latex.constants import ENV_LIST, ENV_MATH, ENV_QUOTE, ENV_THEOREM
from docling.backend.latex.utils.text import iter_child_nodelists

if TYPE_CHECKING:
    from concurrent.futures import Future, ThreadPoolExecutor
    from typing import Any

    from docling.backend.latex.engines.tectonic import TectonicEngine
    from docling.datamodel.backend_options import (
        BaseBackendOptions,
        LatexBackendOptions,
    )

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__

_log = logging.getLogger(__name__)
_TIKZ_END_PATTERN = re.compile(r"\\end\s*\{\s*tikzpicture\s*\}")


class EnvironmentHandlerMixin:
    if TYPE_CHECKING:
        options: BaseBackendOptions
        latex_preamble: str
        _source_dir: Path | None
        _tectonic_engine: TectonicEngine | None
        _tikz_executor: ThreadPoolExecutor | None
        _tikz_futures: list[Future[Any]]
        _tikz_untrusted_warned: bool
        _suppress_image_path_caption: bool

        def _process_nodes(
            self,
            nodes: Any,
            doc: Any,
            parent: Any = ...,
            formatting: Any = ...,
            text_label: Any = ...,
        ) -> None: ...
        def _clean_math(self, latex_str: str, env_name: str) -> str: ...
        def _parse_table(self, node: Any) -> Any: ...
        def _add_rich_table_cells(self, parsed: Any, doc: Any, table: Any) -> None: ...
        def _extract_verbatim_content(self, latex_str: str, env_name: str) -> str: ...
        def _extract_macro_arg(self, node: Any) -> str: ...
        def _nodes_to_text(self, nodes: Any) -> str: ...
        def _add_caption(
            self, node: Any, doc: Any, parent: Any, formatting: Any = ...
        ) -> None: ...

    def _find_document_env(self, nodes, depth: int = 0):
        if nodes is None or depth > 10:
            return None
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                return node
            for child_nodes in iter_child_nodelists(node):
                result = self._find_document_env(child_nodes, depth + 1)
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

        elif node.envname in {"tabular", "tabular*", "tabularx", "longtable"}:
            parsed_table = self._parse_table(node)
            if parsed_table:
                table = doc.add_table(parent=parent, data=parsed_table.data)
                self._add_rich_table_cells(parsed_table, doc, table)
                if node.envname == "longtable":
                    caption_start = len(doc.texts)
                    for child in node.nodelist or []:
                        if (
                            isinstance(child, LatexMacroNode)
                            and child.macroname == "caption"
                        ):
                            self._add_caption(child, doc, parent, formatting)
                    table.captions.extend(
                        caption.get_ref()
                        for caption in doc.texts[caption_start:]
                        if caption.label == DocItemLabel.CAPTION
                    )

        elif node.envname in ["table", "table*"]:
            table_start = len(doc.tables)
            caption_start = len(doc.texts)
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)
            if len(doc.tables) > table_start:
                doc.tables[table_start].captions.extend(
                    caption.get_ref()
                    for caption in doc.texts[caption_start:]
                    if caption.label == DocItemLabel.CAPTION
                )

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

        opts = cast("LatexBackendOptions", self.options)
        if opts.tikz_engine == "tectonic" and not opts.tikz_engine_trusted_input:
            self._warn_tikz_untrusted_input()
        elif opts.tikz_engine == "tectonic":
            try:
                from docling.backend.latex.engines.tectonic import TectonicEngine

                if self._tectonic_engine is None:
                    self._tectonic_engine = TectonicEngine(
                        timeout=opts.tikz_engine_timeout,
                        allow_shell_escape=opts.tikz_engine_allow_shell_escape,
                    )

                pic = doc.add_picture(parent=parent)

                def render_task(engine, raw_tikz, picture_item, preamble, source_root):
                    def set_code_fallback():
                        picture_item.meta = PictureMeta(
                            code=CodeMetaField(
                                text=raw_tikz,
                                language=CodeLanguageLabel.TIKZ,
                            )
                        )

                    try:
                        img_ref = engine.render(
                            raw_tikz, preamble=preamble, source_root=source_root
                        )
                        if img_ref:
                            picture_item.image = img_ref
                        else:
                            # Any Tectonic failure should degrade to raw TikZ.
                            set_code_fallback()
                    except Exception as e:
                        _log.warning(f"Async Tectonic rendering failed: {e}")
                        set_code_fallback()

                preamble = self.latex_preamble
                source_root = self._source_dir
                if self._tikz_executor:
                    future = self._tikz_executor.submit(
                        render_task,
                        self._tectonic_engine,
                        tikz_raw,
                        pic,
                        preamble,
                        source_root,
                    )
                    self._tikz_futures.append(future)
                else:
                    render_task(
                        self._tectonic_engine, tikz_raw, pic, preamble, source_root
                    )
                return
            except Exception as e:
                _log.warning(f"Failed to setup Tectonic async task: {e}")

        pic = doc.add_picture(parent=parent)
        pic.meta = PictureMeta(
            code=CodeMetaField(
                text=tikz_raw,
                language=CodeLanguageLabel.TIKZ,
            )
        )

    def _warn_tikz_untrusted_input(self) -> None:
        if self._tikz_untrusted_warned:
            return
        self._tikz_untrusted_warned = True
        _log.warning(
            "TikZ rendering with Tectonic is skipped because the LaTeX input "
            "is not marked as trusted. Tectonic can read and write files "
            "outside the working directory. Set "
            "LatexBackendOptions(tikz_engine_trusted_input=True) only for "
            "trusted input, or run the conversion in an isolated environment."
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

            for child_nodes in iter_child_nodelists(node):
                if not self._validate_tikz_nodelist(child_nodes, depth + 1):
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
        picture_start = len(doc.pictures)
        caption_start = len(doc.texts)
        previous_suppression = self._suppress_image_path_caption
        self._suppress_image_path_caption = True
        try:
            self._process_nodes(
                node.nodelist, doc, figure_group, formatting, text_label
            )
        finally:
            self._suppress_image_path_caption = previous_suppression
        if len(doc.pictures) > picture_start:
            doc.pictures[picture_start].captions.extend(
                caption.get_ref()
                for caption in doc.texts[caption_start:]
                if caption.label == DocItemLabel.CAPTION
            )

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
            body = self._nodes_to_text(item_nodes).strip()
            if body:
                doc.add_list_item(
                    parent=bib_group,
                    text=f"[{key}] {body}" if key else body,
                    formatting=formatting,
                )
