import logging
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ImageRef,
    NodeItem,
    TableCell,
    TableData,
    TextItem,
)
from docling_core.types.doc.document import Formatting
from PIL import Image
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class LatexDocumentBackend(DeclarativeDocumentBackend):
    def __init__(
        self,
        in_doc: InputDocument,
        path_or_stream: Union[BytesIO, Path],
        options: LatexBackendOptions = LatexBackendOptions(),
    ):
        super().__init__(in_doc, path_or_stream, options)
        self.latex_text = ""
        self.labels: dict[str, bool] = {}

        if isinstance(self.path_or_stream, BytesIO):
            self.latex_text = self.path_or_stream.getvalue().decode("utf-8")
        elif isinstance(self.path_or_stream, Path):
            with open(self.path_or_stream, encoding="utf-8") as f:
                self.latex_text = f.read()

        self.latex_text = self._preprocess_latex(self.latex_text)

    def is_valid(self) -> bool:
        return bool(self.latex_text.strip())

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.LATEX}

    def _preprocess_latex(self, text: str) -> str:
        """Clean up common LaTeX issues before parsing"""

        text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
        return text

    def convert(self) -> DoclingDocument:
        doc = DoclingDocument(name=self.file.stem)
        walker = LatexWalker(self.latex_text)

        try:
            nodes, pos, len_ = walker.get_latex_nodes()
        except Exception as e:
            _log.warning(f"LaTeX parsing failed: {e}. Using fallback text extraction.")
            doc.add_text(label=DocItemLabel.TEXT, text=self.latex_text)
            return doc

        try:
            doc_node = None
            if nodes:
                for node in nodes:
                    if (
                        isinstance(node, LatexEnvironmentNode)
                        and node.envname == "document"
                    ):
                        doc_node = node
                        break

            if doc_node:
                self._process_nodes(doc_node.nodelist, doc)
            else:
                self._process_nodes(nodes, doc)

        except Exception as e:
            _log.error(f"Error processing LaTeX nodes: {e}")

        return doc

    def _process_nodes(
        self, nodes, doc: DoclingDocument, parent: Optional[NodeItem] = None
    ):
        if nodes is None:
            return

        text_buffer: list[str] = []

        def flush_text_buffer():
            if text_buffer:
                combined_text = "".join(text_buffer).strip()
                if combined_text:
                    doc.add_text(
                        parent=parent, label=DocItemLabel.TEXT, text=combined_text
                    )
                text_buffer.clear()

        for node in nodes:
            if isinstance(node, LatexCharsNode):
                text = node.chars

                if "\n\n" in text:
                    flush_text_buffer()

                    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for part in parts:
                        doc.add_text(
                            parent=parent, label=DocItemLabel.PARAGRAPH, text=part
                        )
                else:
                    text_buffer.append(text)

            elif isinstance(node, LatexMacroNode):
                if node.macroname in ["%", "$", "&", "#", "_", "{", "}"]:
                    text_buffer.append(node.macroname)
                elif node.macroname == " ":
                    text_buffer.append(" ")
                else:
                    flush_text_buffer()
                    self._process_macro(node, doc, parent)

            elif isinstance(node, LatexEnvironmentNode):
                flush_text_buffer()
                self._process_environment(node, doc, parent)

            elif isinstance(node, LatexMathNode):
                flush_text_buffer()
                math_text = self._clean_math(node.latex_verbatim(), "inline")
                doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)

            elif isinstance(node, LatexGroupNode):
                self._process_nodes(node.nodelist, doc, parent)

        flush_text_buffer()

    def _process_macro(  # noqa: C901
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem] = None,
    ):
        """Process LaTeX macro nodes"""

        if node.macroname in [
            "part",
            "chapter",
            "section",
            "subsection",
            "subsubsection",
        ]:
            title = self._extract_macro_arg(node)
            if title:
                level = self._get_heading_level(node.macroname)
                doc.add_heading(parent=parent, text=title, level=level)

        elif node.macroname == "title":
            title = self._extract_macro_arg(node)
            if title:
                doc.add_text(parent=parent, label=DocItemLabel.TITLE, text=title)

        elif node.macroname == "author":
            pass

        elif node.macroname in ["date", "thanks", "maketitle"]:
            pass

        elif node.macroname in ["textbf", "textit", "emph", "texttt", "underline"]:
            text = self._extract_macro_arg(node)
            if text:
                formatting = Formatting()
                if node.macroname == "textbf":
                    formatting.bold = True
                elif node.macroname in ["textit", "emph"]:
                    formatting.italic = True
                elif node.macroname == "underline":
                    formatting.underline = True

                doc.add_text(
                    parent=parent,
                    label=DocItemLabel.TEXT,
                    text=text,
                    formatting=formatting,
                )

        elif node.macroname in ["cite", "citep", "citet", "ref", "eqref"]:
            ref_arg = self._extract_macro_arg(node)
            if ref_arg:
                ref_text = f"[{ref_arg}]"
                doc.add_text(parent=parent, label=DocItemLabel.REFERENCE, text=ref_text)

        elif node.macroname == "url":
            url_text = self._extract_macro_arg(node)
            if url_text:
                doc.add_text(parent=parent, label=DocItemLabel.REFERENCE, text=url_text)

        elif node.macroname == "label":
            label_text = self._extract_macro_arg(node)
            if label_text:
                self.labels[label_text] = True

        elif node.macroname == "caption":
            caption_text = self._extract_macro_arg(node)
            if caption_text:
                doc.add_text(
                    parent=parent, label=DocItemLabel.CAPTION, text=caption_text
                )

        elif node.macroname in ["footnote", "marginpar"]:
            footnote_text = self._extract_macro_arg(node)
            if footnote_text:
                doc.add_text(
                    parent=parent, label=DocItemLabel.FOOTNOTE, text=footnote_text
                )

        elif node.macroname == "includegraphics":
            img_path = self._extract_macro_arg(node)
            if img_path:
                image = None
                try:
                    if isinstance(self.path_or_stream, Path):
                        img_full_path = self.path_or_stream.parent / img_path
                        if img_full_path.exists():
                            pil_image = Image.open(img_full_path)
                            dpi = pil_image.info.get("dpi", (72, 72))
                            if isinstance(dpi, tuple):
                                dpi = dpi[0]
                            image = ImageRef.from_pil(image=pil_image, dpi=int(dpi))
                            _log.debug(
                                f"Loaded image {img_path}: {pil_image.size}, DPI={dpi}"
                            )
                except Exception as e:
                    _log.debug(f"Could not load image {img_path}: {e}")

                caption = doc.add_text(
                    label=DocItemLabel.CAPTION, text=f"Image: {img_path}"
                )

                doc.add_picture(
                    parent=parent,
                    caption=caption,
                    image=image,  # Will be None if image couldn't be loaded
                )

        elif node.macroname == "\\":
            pass

        elif node.macroname in [
            "documentclass",
            "usepackage",
            "geometry",
            "hypersetup",
            "lstset",
            "bibliographystyle",
            "newcommand",
            "renewcommand",
        ]:
            pass

        elif node.macroname == "item":
            pass

        else:
            _log.debug(f"Skipping unknown macro: {node.macroname}")

    def _process_environment(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem] = None,
    ):
        """Process LaTeX environment nodes"""

        if node.envname == "document":
            self._process_nodes(node.nodelist, doc, parent)

        elif node.envname == "abstract":
            doc.add_heading(parent=parent, text="Abstract", level=1)
            self._process_nodes(node.nodelist, doc, parent)

        elif node.envname.replace("*", "") in [
            "equation",
            "align",
            "gather",
            "multline",
            "flalign",
            "alignat",
            "displaymath",
            "eqnarray",
        ]:
            math_text = self._clean_math(node.latex_verbatim(), node.envname)
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)

        elif node.envname == "math":
            math_text = self._clean_math(node.latex_verbatim(), node.envname)
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)

        elif node.envname in ["itemize", "enumerate", "description"]:
            self._process_list(node, doc, parent)

        elif node.envname == "tabular":
            table_data = self._parse_table(node)
            if table_data:
                doc.add_table(parent=parent, data=table_data)

        elif node.envname in ["table", "table*"]:
            self._process_nodes(node.nodelist, doc, parent)

        elif node.envname in ["figure", "figure*"]:
            self._process_nodes(node.nodelist, doc, parent)

        elif node.envname in ["verbatim", "lstlisting", "minted"]:
            code_text = self._extract_verbatim_content(
                node.latex_verbatim(), node.envname
            )
            doc.add_text(parent=parent, label=DocItemLabel.CODE, text=code_text)

        elif node.envname == "thebibliography":
            doc.add_heading(parent=parent, text="References", level=1)
            self._process_bibliography(node, doc, parent)

        elif node.envname in ["filecontents", "filecontents*"]:
            pass

        else:
            self._process_nodes(node.nodelist, doc, parent)

    def _process_list(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem] = None,
    ):
        """Process itemize/enumerate environments"""

        list_group = doc.add_group(parent=parent, name="list", label=GroupLabel.LIST)

        items = []
        current_item: list = []

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
            item_text = self._nodes_to_text(item_nodes)
            if item_text:
                doc.add_text(
                    parent=list_group, label=DocItemLabel.LIST_ITEM, text=item_text
                )

    def _parse_table(self, node: LatexEnvironmentNode) -> Optional[TableData]:
        """Parse tabular environment into TableData"""
        try:
            table_text = node.latex_verbatim()

            content = re.search(
                r"\\begin\{tabular\}.*?\{.*?\}(.*?)\\end\{tabular\}",
                table_text,
                re.DOTALL,
            )

            if not content:
                return None

            table_content = content.group(1).strip()

            rows = [r.strip() for r in table_content.split(r"\\") if r.strip()]

            table_rows = []
            for row in rows:
                row = row.replace(r"\hline", "").replace(r"\cline", "").strip()

                if not row:
                    continue

                cells = []
                for c in re.split(r"(?<!\\)&", row):
                    cell_text = c.strip()

                    cell_text = (
                        cell_text.replace(r"\%", "%")
                        .replace(r"\$", "$")
                        .replace(r"\&", "&")
                    )
                    cell_text = cell_text.replace(r"\#", "#").replace(r"\_", "_")
                    cell_text = cell_text.replace(r"\{", "{").replace(r"\}", "}")
                    cells.append(cell_text)
                table_rows.append(cells)

            if not table_rows:
                return None

            num_rows = len(table_rows)
            num_cols = max(len(row) for row in table_rows) if table_rows else 0

            grid = []
            for i, row in enumerate(table_rows):
                grid_row = []
                for j in range(num_cols):
                    cell_text = row[j] if j < len(row) else ""
                    grid_row.append(
                        TableCell(
                            text=cell_text,
                            start_row_offset_idx=i,
                            end_row_offset_idx=i + 1,
                            start_col_offset_idx=j,
                            end_col_offset_idx=j + 1,
                        )
                    )
                grid.append(grid_row)

            flat_cells = [cell for row in grid for cell in row]

            return TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=flat_cells
            )

        except Exception as e:
            _log.warning(f"Failed to parse table: {e}")
            return None

    def _extract_verbatim_content(self, latex_str: str, env_name: str) -> str:
        """Extract content from verbatim environments"""

        pattern = rf"\\begin\{{{env_name}\}}.*?(.*?)\\end\{{{env_name}\}}"
        match = re.search(pattern, latex_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return latex_str

    def _process_bibliography(
        self,
        node: LatexEnvironmentNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem] = None,
    ):
        """Process bibliography environment"""

        bib_group = doc.add_group(
            parent=parent, name="bibliography", label=GroupLabel.LIST
        )

        for n in node.nodelist:
            if isinstance(n, LatexMacroNode) and n.macroname == "bibitem":
                bib_text = self._nodes_to_text([n])
                if bib_text:
                    doc.add_text(
                        parent=bib_group, label=DocItemLabel.LIST_ITEM, text=bib_text
                    )

    def _nodes_to_text(self, nodes) -> str:
        """Convert a list of nodes to plain text"""
        text_parts = []

        for node in nodes:
            if isinstance(node, LatexCharsNode):
                text_parts.append(node.chars)

            elif isinstance(node, LatexGroupNode):
                text_parts.append(self._nodes_to_text(node.nodelist))

            elif isinstance(node, LatexMacroNode):
                if node.macroname in [
                    "textbf",
                    "textit",
                    "emph",
                    "texttt",
                    "underline",
                ]:
                    text = self._extract_macro_arg(node)
                    if text:
                        text_parts.append(text)
                elif node.macroname in ["cite", "citep", "citet", "ref", "eqref"]:
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

                else:
                    arg_text = self._extract_macro_arg(node)
                    if arg_text:
                        text_parts.append(arg_text)

            elif isinstance(node, LatexMathNode):
                text_parts.append(node.latex_verbatim())

            elif isinstance(node, LatexEnvironmentNode):
                if node.envname in ["equation", "align", "gather"]:
                    text_parts.append(node.latex_verbatim())
                else:
                    text_parts.append(self._nodes_to_text(node.nodelist))

        result = "".join(text_parts)
        result = re.sub(r" +", " ", result)
        result = re.sub(r"\n\n+", "\n\n", result)
        return result.strip()

    def _extract_macro_arg(self, node: LatexMacroNode) -> str:
        """Extract text from macro argument"""
        if node.nodeargd and node.nodeargd.argnlist:
            arg = node.nodeargd.argnlist[-1]
            if arg:
                if hasattr(arg, "nodelist"):
                    return self._nodes_to_text(arg.nodelist)
                return arg.latex_verbatim().strip("{} ")
        return ""

    def _clean_math(self, latex_str: str, env_name: str) -> str:
        """Clean math expressions for better readability"""

        envs_to_strip = [
            "equation",
            "equation*",
            "displaymath",
            "math",
            "eqnarray",
            "eqnarray*",
        ]

        if env_name in envs_to_strip:
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

        return latex_str.strip()

    def _get_heading_level(self, macroname: str) -> int:
        """Get heading level for sectioning commands"""
        levels = {
            "part": 1,
            "chapter": 1,
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "paragraph": 4,
        }
        return levels.get(macroname, 1)
