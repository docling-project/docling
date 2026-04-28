import logging
import os
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem
from docling_core.types.doc.document import Formatting
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.backend.latex.handlers.environments import EnvironmentHandlerMixin
from docling.backend.latex.handlers.macros import MacroHandlerMixin
from docling.backend.latex.handlers.math import MathHandlerMixin
from docling.backend.latex.utils.encoding import decode_latex_content
from docling.backend.latex.utils.table import TableHelperMixin
from docling.backend.latex.utils.text import TextHelperMixin
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


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
        if options is None:
            options = LatexBackendOptions()
        super().__init__(in_doc, path_or_stream, options)
        self.labels: dict[str, bool] = {}
        self._custom_macros: dict[str, str] = {}
        self._custom_macro_num_args: dict[str, int] = {}
        self._input_stack: set[str] = set()
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
        import re

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

        walker = LatexWalker(preprocessed_text, tolerant_parsing=True)

        try:
            nodes, _pos, _len = walker.get_latex_nodes()
        except Exception as e:
            _log.warning(f"LaTeX parsing failed: {e}. Using fallback text extraction.")
            doc.add_text(label=DocItemLabel.TEXT, text=self.latex_text)
            return doc

        workers = max(1, (os.cpu_count() or 2) - 1)
        self._tikz_executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self._tikz_futures = []  # type: ignore

        try:
            self._extract_custom_macros(nodes)
            self._extract_preamble_metadata(nodes, doc)

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

    def convert(self) -> DoclingDocument:
        doc = DoclingDocument(name=self.file.stem)
        timeout: float | None = getattr(self.options, "parse_timeout", None)

        if timeout is None:
            return self._do_parse_and_process(doc)

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
            return fallback

        if error_container:
            _log.error(f"Error during LaTeX parsing: {error_container[0]}")
            return doc

        if result_container:
            return result_container[0]

        return doc

    def _process_nodes(
        self,
        nodes,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        if nodes is None:
            return

        text_buffer: list[str] = []

        def flush_text_buffer():
            if text_buffer:
                combined_text = "".join(text_buffer).strip()
                if combined_text:
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

                elif isinstance(node, LatexMacroNode):
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
                    flush_text_buffer()
                    self._process_environment(node, doc, parent, formatting, text_label)

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
