# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import os
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, NodeItem
from docling_core.types.doc.document import Formatting

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.latex.constants import MACROS_IGNORED
from docling.backend.latex.handlers.environments import EnvironmentHandlerMixin
from docling.backend.latex.handlers.macros import MacroHandlerMixin
from docling.backend.latex.handlers.math import InlineFormula, MathHandlerMixin
from docling.backend.latex.handlers.metadata import LatexMetadataHandler
from docling.backend.latex.utils.encoding import decode_latex_content
from docling.backend.latex.utils.latex_context import LATEX_CONTEXT_DB
from docling.backend.latex.utils.table import TableHelperMixin
from docling.backend.latex.utils.text import TextHelperMixin
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)
_SOURCE_LAYOUT_WHITESPACE = re.compile(r"[ \t\r\n]*[\t\r\n][ \t\r\n]*")

_PYLATEXENC_AVAILABLE: bool = False
_PYLATEXENC_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexGroupNode,
        LatexMacroNode,
        LatexMathNode,
        LatexWalker,
    )
    from pylatexenc.macrospec import MacroSpec

    _PYLATEXENC_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _PYLATEXENC_IMPORT_ERROR = e

_INSTALL_HINT = (
    "The 'pylatexenc' package is required to process LaTeX files. "
    "Install it with `pip install 'docling-slim[format-latex]'`."
)

if TYPE_CHECKING:
    import concurrent.futures

    from docling.backend.latex.engines.tectonic import TectonicEngine


class LatexDocumentBackend(
    DeclarativeDocumentBackend,
    MacroHandlerMixin,
    EnvironmentHandlerMixin,
    MathHandlerMixin,
    TextHelperMixin,
    TableHelperMixin,
):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: Optional[LatexBackendOptions] = None,
    ):
        if not _PYLATEXENC_AVAILABLE:
            raise ImportError(_INSTALL_HINT) from _PYLATEXENC_IMPORT_ERROR
        if options is None:
            options = LatexBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.options = LatexBackendOptions.model_validate(self.options)
        # Directory that relative \input, \include and \includegraphics paths
        # resolve against. A stream has none unless source_uri gives it.
        self._source_dir: Path | None = None
        if isinstance(self.path_or_stream, Path):
            self._source_dir = self.path_or_stream.parent
        elif self.options.source_uri is not None:
            self._source_dir = Path(self.options.source_uri).parent
        self.labels: dict[str, bool] = {}
        self._custom_macros: dict[str, str] = {}
        self._custom_macro_num_args: dict[str, int] = {}
        self._metadata_handler = LatexMetadataHandler(self)
        self._masked_macro_spans: list[tuple[int, int]] = []
        self._suppress_image_path_caption = False
        self._input_stack: set[str] = set()
        self._tectonic_engine: TectonicEngine | None = None
        self._tikz_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._tikz_futures: list[concurrent.futures.Future] = []
        self.latex_text = decode_latex_content(self.path_or_stream)

    def is_valid(self) -> bool:
        text = self.latex_text.strip()
        if not text:
            return False
        return True

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.LATEX}

    def _do_parse_and_process(self, doc: DoclingDocument) -> DoclingDocument:
        import concurrent.futures

        preprocessed_text = self._preprocess_custom_macros(self.latex_text)

        # Extract preamble: everything before \begin{document}
        preamble_match = re.search(
            r"^(.*?)\\begin\{document\}", preprocessed_text, re.DOTALL
        )
        if preamble_match:
            raw_preamble = preamble_match.group(1)
            # Remove the original \documentclass declaration
            self.latex_preamble = re.sub(
                r"\\documentclass(?:\[[^\]]*\])?\s*\{[^}]*\}", "", raw_preamble
            )
        else:
            self.latex_preamble = ""

        parser_context = LATEX_CONTEXT_DB.filter_context()
        zero_arg_macros = [
            MacroSpec(name, "")
            for name, count in self._custom_macro_num_args.items()
            if count == 0
        ]
        if zero_arg_macros:
            parser_context.add_context_category(
                "docling-source-macros", macros=zero_arg_macros, prepend=True
            )
        walker = LatexWalker(
            preprocessed_text, tolerant_parsing=True, latex_context=parser_context
        )

        try:
            nodes, _pos, _len = walker.get_latex_nodes()
        except Exception as e:
            _log.warning(f"LaTeX parsing failed: {e}. Using fallback text extraction.")
            doc.add_text(label=DocItemLabel.TEXT, text=self.latex_text)
            return doc

        self._preserve_invisible_commands(nodes, doc)

        workers = max(1, (os.cpu_count() or 2) - 1)
        self._tikz_executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._tikz_futures = []  # type: ignore

        try:
            self._extract_custom_macros(nodes)
            self._metadata_handler.process_preamble(nodes, doc)

            doc_node = self._find_document_env(nodes)

            if doc_node:
                self._process_nodes(doc_node.nodelist, doc)
            else:
                self._process_nodes(nodes, doc)

            if self._tikz_futures:
                concurrent.futures.wait(self._tikz_futures)

        except Exception as e:
            _log.error(f"Error processing LaTeX nodes: {e}")
        finally:
            self._tikz_executor.shutdown(wait=False)

        return doc

    def _preserve_invisible_commands(self, nodes, doc: DoclingDocument) -> None:
        """Keep masked definitions and ignored preamble commands in source order."""
        spans = list(self._masked_macro_spans)
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                break
            if isinstance(node, LatexMacroNode) and node.macroname in MACROS_IGNORED:
                spans.append((node.pos, node.pos + node.len))

        for start, end in sorted(spans):
            doc.add_text(
                label=DocItemLabel.TEXT,
                text=self.latex_text[start:end],
                content_layer=ContentLayer.INVISIBLE,
            )

    def convert(self) -> DoclingDocument:
        doc = DoclingDocument(name=self.file.stem)
        opts = cast(LatexBackendOptions, self.options)
        timeout = opts.parse_timeout

        if timeout is None:
            return self._normalize_document_text(self._do_parse_and_process(doc))

        result_container: list[DoclingDocument] = []
        error_container: list[Exception] = []

        def _worker_fn():
            try:
                res = self._do_parse_and_process(doc)
                result_container.append(res)
            except Exception as e:
                error_container.append(e)

        thread = threading.Thread(target=_worker_fn, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            _log.warning(
                f"LaTeX parsing timed out after {timeout}s for "
                f"'{self.file.name}'. "
                "Returning partial document with raw text fallback."
            )
            fallback = DoclingDocument(name=self.file.stem)
            fallback.add_text(label=DocItemLabel.TEXT, text=self.latex_text)
            return self._normalize_document_text(fallback)

        if error_container:
            _log.error(f"Error during LaTeX parsing: {error_container[0]}")
            return self._normalize_document_text(doc)

        if result_container:
            return self._normalize_document_text(result_container[0])

        return self._normalize_document_text(doc)

    @staticmethod
    def _normalize_document_text(doc: DoclingDocument) -> DoclingDocument:
        for item in doc.texts:
            if item.content_layer == ContentLayer.INVISIBLE or item.label in {
                DocItemLabel.FORMULA,
                DocItemLabel.CODE,
            }:
                continue
            item.text = _SOURCE_LAYOUT_WHITESPACE.sub(" ", item.text)
            item.orig = _SOURCE_LAYOUT_WHITESPACE.sub(" ", item.orig)

        for table in doc.tables:
            for cell in table.data.table_cells:
                cell.text = _SOURCE_LAYOUT_WHITESPACE.sub(" ", cell.text)

        return doc

    def _process_nodes(
        self,
        nodes,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
        heading_level: int | None = None,
    ):
        if nodes is None:
            return

        text_buffer: list[str | InlineFormula] = []

        def flush_text_buffer():
            if text_buffer:
                if any(isinstance(part, InlineFormula) for part in text_buffer):
                    if heading_level is not None:
                        paragraph = doc.add_heading(
                            parent=parent,
                            text="",
                            level=heading_level,
                            formatting=formatting,
                        )
                    else:
                        paragraph = doc.add_text(
                            parent=parent,
                            label=text_label or DocItemLabel.TEXT,
                            text="",
                            formatting=formatting,
                        )
                    inline_group = doc.add_inline_group(parent=paragraph)
                    parts: list[str | InlineFormula] = []
                    for part in text_buffer:
                        if (
                            isinstance(part, str)
                            and parts
                            and isinstance(parts[-1], str)
                        ):
                            parts[-1] += part
                        else:
                            parts.append(part)
                    if isinstance(parts[0], str):
                        parts[0] = parts[0].lstrip()
                    if isinstance(parts[-1], str):
                        parts[-1] = parts[-1].rstrip()
                    text_run = ""
                    for part in parts:
                        if isinstance(part, InlineFormula):
                            if text_run:
                                doc.add_text(
                                    parent=inline_group,
                                    label=DocItemLabel.TEXT,
                                    text=text_run,
                                )
                                text_run = ""
                            doc.add_formula(parent=inline_group, text=part.text)
                        else:
                            text_run += part
                    if text_run:
                        doc.add_text(
                            parent=inline_group,
                            label=DocItemLabel.TEXT,
                            text=text_run,
                        )
                else:
                    combined_text = "".join(text_buffer).strip()
                    if combined_text:
                        if heading_level is not None:
                            doc.add_heading(
                                parent=parent,
                                text=combined_text,
                                level=heading_level,
                                formatting=formatting,
                            )
                        else:
                            doc.add_text(
                                parent=parent,
                                label=text_label or DocItemLabel.TEXT,
                                text=combined_text,
                                formatting=formatting,
                            )
                text_buffer.clear()

        idx = 0
        while idx < len(nodes):
            node = nodes[idx]
            consumed_following = 0
            try:
                if isinstance(node, LatexCharsNode):
                    self._process_chars_node(
                        node,
                        doc,
                        parent,
                        formatting,
                        text_label,
                        text_buffer,
                        flush_text_buffer,
                    )

                elif isinstance(node, InlineFormula):
                    text_buffer.append(node)

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
                            flush_text_buffer,
                        )
                    else:
                        consumed_following = self._process_macro_node_inline(
                            node,
                            doc,
                            parent,
                            formatting,
                            text_label,
                            text_buffer,
                            flush_text_buffer,
                            nodes[idx + 1 :],
                        )

                elif isinstance(node, LatexEnvironmentNode):
                    if node.envname == "math":
                        math_text = self._clean_math(
                            node.latex_verbatim(), node.envname
                        )
                        if math_text:
                            text_buffer.append(InlineFormula(math_text))
                    else:
                        flush_text_buffer()
                        self._process_environment(
                            node, doc, parent, formatting, text_label
                        )

                elif isinstance(node, LatexMathNode):
                    self._process_math_node(
                        node, doc, parent, text_buffer, flush_text_buffer
                    )

                elif isinstance(node, LatexGroupNode):
                    self._process_group_node(
                        node,
                        doc,
                        parent,
                        formatting,
                        text_label,
                        text_buffer,
                        flush_text_buffer,
                    )

            except Exception as e:
                _log.warning(f"Failed to process node {type(node).__name__}: {e}")
            idx += 1 + consumed_following

        flush_text_buffer()
