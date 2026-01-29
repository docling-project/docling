import logging
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import pypdfium2
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
from pylatexenc.latex2text import LatexNodes2Text

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
        self._custom_macros: dict[str, str] = {}

        if isinstance(self.path_or_stream, BytesIO):
            self.latex_text = self.path_or_stream.getvalue().decode("utf-8")
        elif isinstance(self.path_or_stream, Path):
            with open(self.path_or_stream, encoding="utf-8") as f:
                self.latex_text = f.read()

    def is_valid(self) -> bool:
        return bool(self.latex_text.strip())

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    @classmethod
    def supported_formats(cls) -> set[InputFormat]:
        return {InputFormat.LATEX}

    def _preprocess_custom_macros(self, latex_text: str) -> str:
        """Pre-process LaTeX to expand common problematic macros before parsing"""
        # Common equation shortcuts that cause parsing issues
        latex_text = re.sub(r'\\be\b', r'\\begin{equation}', latex_text)
        latex_text = re.sub(r'\\ee\b', r'\\end{equation}', latex_text)
        latex_text = re.sub(r'\\bea\b', r'\\begin{eqnarray}', latex_text)
        latex_text = re.sub(r'\\eea\b', r'\\end{eqnarray}', latex_text)
        latex_text = re.sub(r'\\beq\b', r'\\begin{equation}', latex_text)
        latex_text = re.sub(r'\\eeq\b', r'\\end{equation}', latex_text)

        return latex_text

    def convert(self) -> DoclingDocument:
        doc = DoclingDocument(name=self.file.stem)

        # Pre-process: expand common custom equation macros
        preprocessed_text = self._preprocess_custom_macros(self.latex_text)

        walker = LatexWalker(preprocessed_text, tolerant_parsing=True)

        try:
            nodes, pos, len_ = walker.get_latex_nodes()
        except Exception as e:
            _log.warning(f"LaTeX parsing failed: {e}. Using fallback text extraction.")
            doc.add_text(label=DocItemLabel.TEXT, text=self.latex_text)
            return doc

        try:
            # First pass: Extract custom macros from ALL nodes (including preamble)
            # This must happen before finding the document environment
            self._extract_custom_macros(nodes)

            doc_node = self._find_document_env(nodes)

            if doc_node:
                self._process_nodes(doc_node.nodelist, doc)
            else:
                self._process_nodes(nodes, doc)

        except Exception as e:
            _log.error(f"Error processing LaTeX nodes: {e}")

        return doc

    def _extract_custom_macros(self, nodes, depth: int = 0):
        """Extract custom macro definitions from the document"""
        if nodes is None or depth > 5:
            return

        for node in nodes:
            if isinstance(node, LatexMacroNode) and node.macroname == "newcommand":
                # Extract macro name and definition
                # pylatexenc parses \newcommand{\macroname}{definition} as:
                # argnlist[0]: None (optional * variant)
                # argnlist[1]: {\macroname} - the macro name
                # argnlist[2]: None (optional number of arguments)
                # argnlist[3]: None (optional default value)
                # argnlist[4]: {definition} - the definition (or last non-None element)
                if node.nodeargd and node.nodeargd.argnlist:
                    argnlist = node.nodeargd.argnlist
                    
                    # Find the name argument (typically at index 1)
                    name_arg = argnlist[1] if len(argnlist) > 1 else None
                    
                    # Find the definition argument (last non-None argument)
                    def_arg = None
                    for arg in reversed(argnlist):
                        if arg is not None:
                            def_arg = arg
                            break

                    if name_arg and def_arg and name_arg is not def_arg:
                        # Extract macro name from the first argument
                        # The macro name comes as raw latex like "{\myterm}" or "\myterm"
                        macro_name_raw = name_arg.latex_verbatim()

                        # Clean up: remove braces, spaces, and leading backslash
                        # This handles both {\myterm} and \myterm formats
                        macro_name = macro_name_raw.strip("{} \n\t")
                        
                        # Remove leading backslash if present
                        if macro_name.startswith("\\"):
                            macro_name = macro_name[1:]

                        # Extract definition text
                        if hasattr(def_arg, "nodelist"):
                            macro_def = self._nodes_to_text(def_arg.nodelist)
                        else:
                            macro_def = def_arg.latex_verbatim().strip("{} ")

                        if macro_name:  # Only register if we got a valid name
                            self._custom_macros[macro_name] = macro_def
                            _log.debug(f"Registered custom macro: \\{macro_name} -> '{macro_def}'")

            # Recursively search in nested structures
            if hasattr(node, "nodelist") and node.nodelist:
                self._extract_custom_macros(node.nodelist, depth + 1)
            if hasattr(node, "nodeargd") and node.nodeargd:
                argnlist = getattr(node.nodeargd, "argnlist", None)
                if argnlist:
                    for arg in argnlist:
                        if hasattr(arg, "nodelist") and arg.nodelist:
                            self._extract_custom_macros(arg.nodelist, depth + 1)

    def _find_document_env(self, nodes, depth: int = 0):
        """Recursively search for document environment"""
        if nodes is None or depth > 5:
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

    def _process_nodes(
            self,
            nodes,
            doc: DoclingDocument,
            parent: Optional[NodeItem] = None,
            formatting: Optional[Formatting] = None,
            text_label: Optional[DocItemLabel] = None,
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

        for node in nodes:
            if isinstance(node, LatexCharsNode):
                text = node.chars

                if "\n\n" in text:
                    flush_text_buffer()

                    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for part in parts:
                        doc.add_text(
                            parent=parent,
                            label=text_label or DocItemLabel.PARAGRAPH,
                            text=part,
                            formatting=formatting,
                        )
                else:
                    text_buffer.append(text)

            elif isinstance(node, LatexMacroNode):
                if node.macroname in ["%", "$", "&", "#", "_", "{", "}", "~"]:
                    if node.macroname == "~":
                        text_buffer.append(" ")  # Non-breaking space
                    else:
                        text_buffer.append(node.macroname)
                elif node.macroname == " ":
                    text_buffer.append(" ")
                # Handle inline formatting macros - keep in buffer
                elif node.macroname in ["textbf", "textit", "emph", "texttt", "underline"]:
                    formatted_text = self._extract_macro_arg(node)
                    if formatted_text:
                        text_buffer.append(formatted_text)
                # Handle custom macros - expand and keep in buffer
                elif node.macroname in self._custom_macros:
                    expansion = self._custom_macros[node.macroname]
                    _log.debug(f"Expanding custom macro \\{node.macroname} -> '{expansion}'")
                    text_buffer.append(expansion)
                # Handle citations and references inline to avoid line breaks
                elif node.macroname in ["cite", "citep", "citet", "ref", "eqref"]:
                    ref_arg = self._extract_macro_arg(node)
                    if ref_arg:
                        text_buffer.append(f"[{ref_arg}]")
                # Handle URLs inline
                elif node.macroname == "url":
                    url_text = self._extract_macro_arg(node)
                    if url_text:
                        text_buffer.append(url_text)
                else:
                    flush_text_buffer()
                    self._process_macro(node, doc, parent, formatting, text_label)

            elif isinstance(node, LatexEnvironmentNode):
                flush_text_buffer()
                self._process_environment(node, doc, parent, formatting, text_label)

            elif isinstance(node, LatexMathNode):
                is_display = getattr(node, "displaytype", None) == "display"

                if not is_display:
                    math_verbatim = node.latex_verbatim()
                    is_display = math_verbatim.startswith(("$$", "\\[")) or math_verbatim.startswith(
                        ("\\begin{equation}", "\\begin{align}", "\\begin{gather}", "\\begin{displaymath}"))

                if is_display:
                    flush_text_buffer()
                    math_text = self._clean_math(node.latex_verbatim(), "display")
                    doc.add_text(
                        parent=parent, label=DocItemLabel.FORMULA, text=math_text
                    )
                else:
                    # Inline math: keep in buffer to avoid splitting paragraphs
                    text_buffer.append(node.latex_verbatim())

            elif isinstance(node, LatexGroupNode):
                self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        flush_text_buffer()

    def _process_macro(  # noqa: C901
            self,
            node: LatexMacroNode,
            doc: DoclingDocument,
            parent: Optional[NodeItem] = None,
            formatting: Optional[Formatting] = None,
            text_label: Optional[DocItemLabel] = None,
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

        elif node.macroname in ["textsc", "textsf", "textrm", "textnormal", "mbox"]:
            # Similar recursion
            if node.nodeargd and node.nodeargd.argnlist:
                arg = node.nodeargd.argnlist[-1]
                if hasattr(arg, "nodelist"):
                    self._process_nodes(arg.nodelist, doc, parent, formatting, text_label)


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
            # Store labels for potential cross-referencing
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
                            suffix = img_full_path.suffix.lower()
                            if suffix == ".pdf":
                                pdf = pypdfium2.PdfDocument(img_full_path)
                                page = pdf[0]
                                pil_image = page.render(scale=2).to_pil()
                                page.close()
                                pdf.close()
                                dpi = 144
                                _log.debug(
                                    f"Rendered PDF image {img_path}: {pil_image.size}"
                                )
                            else:
                                pil_image = Image.open(img_full_path)
                                dpi = pil_image.info.get("dpi", (72, 72))
                                if isinstance(dpi, tuple):
                                    dpi = dpi[0]
                                _log.debug(
                                    f"Loaded image {img_path}: {pil_image.size}, DPI={dpi}"
                                )
                            image = ImageRef.from_pil(image=pil_image, dpi=int(dpi))
                except Exception as e:
                    _log.debug(f"Could not load image {img_path}: {e}")

                caption_node = None
                # Check for caption in parent figure environment if we want to link explicitly
                # But Docling add_picture logic handles caption?
                # The existing code added caption then picture.

                caption = doc.add_text(
                    label=DocItemLabel.CAPTION, text=f"Image: {img_path}"
                )

                doc.add_picture(
                    parent=parent,
                    caption=caption,
                    image=image,
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
            "def",
            "let",
            "edef",
            "gdef",
            "xdef",
            "newenvironment",
            "renewenvironment",
            "DeclareMathOperator",
            "DeclareMathSymbol",
            "setlength",
            "setcounter",
            "addtolength",
            "color",
            "definecolor",
            "colorlet",
            "AtBeginDocument",
            "AtEndDocument",
            "newlength",
            "newcounter",
            "newif",
            "providecommand",
            "DeclareOption",
            "RequirePackage",
            "ProvidesPackage",
            "LoadClass",
            "makeatletter",
            "makeatother",
            "NeedsTeXFormat",
            "ProvidesClass",
            "DeclareRobustCommand",
        ]:
            pass

        elif node.macroname in ["input", "include"]:
            filepath = self._extract_macro_arg(node)
            if filepath and isinstance(self.path_or_stream, Path):
                input_path = self.path_or_stream.parent / filepath
                if not input_path.suffix:
                    input_path = input_path.with_suffix(".tex")
                if input_path.exists():
                    try:
                        content = input_path.read_text(encoding="utf-8")
                        sub_walker = LatexWalker(content, tolerant_parsing=True)
                        sub_nodes, _, _ = sub_walker.get_latex_nodes()
                        self._process_nodes(sub_nodes, doc, parent, formatting, text_label)
                        _log.debug(f"Loaded input file: {input_path}")
                    except Exception as e:
                        _log.debug(f"Failed to load input file {filepath}: {e}")

        elif node.macroname in ["&", "%", "$", "#", "_", "{", "}"]:
            # Escaped symbols: \& -> &
            doc.add_text(parent=parent, text=node.macroname, formatting=formatting,
                         label=(text_label or DocItemLabel.TEXT))

        elif node.macroname in ["'", '"', "^", "`", "~", "=", ".", "c", "d", "b", "H", "k", "r", "t", "u", "v"]:
            # Accents and diacritics
            try:
                text = LatexNodes2Text().nodelist_to_text([node])
                doc.add_text(parent=parent, text=text, formatting=formatting, label=(text_label or DocItemLabel.TEXT))
            except Exception:
                # Fallback handled by generic handler if we don't catch it,
                # but we just continue
                pass

        elif node.macroname == "href":
            # \href{url}{text}
            if node.nodeargd and len(node.nodeargd.argnlist) >= 2:
                # url_arg = node.nodeargd.argnlist[0]
                text_arg = node.nodeargd.argnlist[1]

                # We process the text content.
                # Ideally we would mark it as a link, but Docling TextItem doesn't have URL field?
                # We prioritize content preservation.
                if hasattr(text_arg, "nodelist"):
                    self._process_nodes(text_arg.nodelist, doc, parent, formatting, text_label)

        elif node.macroname in ["newline", "hfill", "break", "centering"]:
            if node.macroname == "newline":
                doc.add_text(parent=parent, text="\n", formatting=formatting, label=(text_label or DocItemLabel.TEXT))

        elif node.macroname in ["bf", "it", "rm", "sc", "sf", "sl", "tt", "cal", "em",
                                "tiny", "scriptsize", "footnotesize", "small",
                                "large", "Large", "LARGE", "huge", "Huge"]:
            # Legacy formatting and size switches - ignore to preserve content flow (prevent "Unknown macro" skip)
            pass

        elif node.macroname in ["textcolor", "colorbox"]:
            # \textcolor{color}{text} - process only the text content (last argument)
            if node.nodeargd and node.nodeargd.argnlist:
                # Find the last non-None argument (the text content)
                for arg in reversed(node.nodeargd.argnlist):
                    if arg is not None and hasattr(arg, "nodelist"):
                        self._process_nodes(arg.nodelist, doc, parent, formatting, text_label)
                        break

        elif node.macroname == "item":
            pass

        else:
            # Unknown macro - try to extract content from arguments
            if node.nodeargd and node.nodeargd.argnlist:
                processed_any = False
                for arg in node.nodeargd.argnlist:
                    if hasattr(arg, "nodelist"):
                        self._process_nodes(arg.nodelist, doc, parent, formatting, text_label)
                        processed_any = True

                if processed_any:
                    _log.debug(f"Processed content of unknown macro: {node.macroname}")
                else:
                    _log.debug(f"Skipping unknown macro: {node.macroname}")
            else:
                _log.debug(f"Skipping unknown macro: {node.macroname}")

    def _process_environment(
            self,
            node: LatexEnvironmentNode,
            doc: DoclingDocument,
            parent: Optional[NodeItem] = None,
            formatting: Optional[Formatting] = None,
            text_label: Optional[DocItemLabel] = None,
    ):
        """Process LaTeX environment nodes"""

        if node.envname == "document":
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname == "abstract":
            doc.add_heading(parent=parent, text="Abstract", level=1)
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

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
            self._process_list(node, doc, parent, formatting, text_label)

        elif node.envname == "tabular":
            table_data = self._parse_table(node)
            if table_data:
                doc.add_table(parent=parent, data=table_data)

        elif node.envname in ["table", "table*"]:
            self._process_nodes(node.nodelist, doc, parent, formatting, text_label)

        elif node.envname in ["figure", "figure*"]:
            # Process figure environment with proper grouping
            self._process_figure(node, doc, parent, formatting, text_label)

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

    def _process_figure(
            self,
            node: LatexEnvironmentNode,
            doc: DoclingDocument,
            parent: Optional[NodeItem] = None,
            formatting: Optional[Formatting] = None,
            text_label: Optional[DocItemLabel] = None,
    ):
        """Process figure environment with proper grouping"""
        # Create a group for the figure to contain images and captions together
        figure_group = doc.add_group(
            parent=parent, name="figure", label=GroupLabel.SECTION
        )

        # Process all nodes within the figure
        self._process_nodes(node.nodelist, doc, figure_group, formatting, text_label)

    def _process_list(
            self,
            node: LatexEnvironmentNode,
            doc: DoclingDocument,
            parent: Optional[NodeItem] = None,
            formatting: Optional[Formatting] = None,
            text_label: Optional[DocItemLabel] = None,
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
                    # Handle optional argument for description lists or similar
                    # But we are just collecting nodes here.
                    # The content of item is in following nodes.
                    # Does item macro have arguments? \item[label]
                    # We should include the item macro itself or its argument?
                    # The user issue 1.3 says "You skip \item in _process_macro, then re-handle it manually in _process_list... Nested lists break".
                    # Here we are collecting nodes between items.
                    # We should probably process the item arguments (label) if present.
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
                text_label=DocItemLabel.LIST_ITEM
            )

    def _parse_table(self, node: LatexEnvironmentNode) -> Optional[TableData]:
        """Parse tabular environment into TableData using AST"""

        rows = []
        current_row = []
        current_cell_nodes = []

        def finish_cell():
            text = self._nodes_to_text(current_cell_nodes).strip()
            # Handle empty cells or just spacing?
            # Standard Latex table cell.
            # Docling TableCell expects text.
            # We can rely on default spans (1,1).
            # Initialize offsets to 0, they are updated later.
            current_row.append(TableCell(
                text=text,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0
            ))
            current_cell_nodes.clear()

        def finish_row():
            finish_cell()  # Finish the last cell of the row
            if current_row:
                rows.append(current_row[:])  # Copy
            current_row.clear()

        for n in node.nodelist:
            if isinstance(n, LatexMacroNode):
                if n.macroname == "\\":  # Row break
                    finish_row()
                elif n.macroname in ["hline", "cline", "toprule", "midrule", "bottomrule"]:
                    # Ignore rule lines for data extraction
                    pass
                elif n.macroname == "&":  # Cell break (if parsed as macro)
                    finish_cell()
                elif n.macroname in ["%", "$", "#", "_", "{", "}"]:
                    # Escaped characters - add to current cell
                    current_cell_nodes.append(n)
                else:
                    current_cell_nodes.append(n)

            elif isinstance(n, LatexCharsNode):
                text = n.chars
                if "&" in text:
                    # This happens if `&` is parsed as char.
                    # Split text by `&`.
                    parts = text.split("&")
                    for i, part in enumerate(parts):
                        if part:
                            # Add text node for the part
                            current_cell_nodes.append(LatexCharsNode(chars=part))

                        if i < len(parts) - 1:
                            finish_cell()
                else:
                    current_cell_nodes.append(n)

            else:
                # Other nodes (Groups, etc).
                # Check if it is `&` (specials).
                if hasattr(n, 'specials_chars') and n.specials_chars == '&':
                    finish_cell()
                else:
                    current_cell_nodes.append(n)

        finish_row()

        if not rows:
            return None

        # Calculate dimensions and build grid
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0

        # Pad rows to uniform length
        flat_cells = []
        for i, row in enumerate(rows):
            for j in range(num_cols):
                if j < len(row):
                    cell = row[j]
                else:
                    cell = TableCell(text="")

                # Update cell offsets (required by TableData?)
                cell.start_row_offset_idx = i
                cell.end_row_offset_idx = i + 1
                cell.start_col_offset_idx = j
                cell.end_col_offset_idx = j + 1

                flat_cells.append(cell)

        return TableData(
            num_rows=num_rows, num_cols=num_cols, table_cells=flat_cells
        )

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
            formatting: Optional[Formatting] = None,
    ):
        """Process bibliography environment"""

        bib_group = doc.add_group(
            parent=parent, name="bibliography", label=GroupLabel.LIST
        )

        items = []
        current_item: list = []
        current_key = ""

        # Pre-process to group by bibitem
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
            # We can optionally add the key as text prefix if desired,
            # but usually the renderer handles numbering/labels.
            # However, for robustness, we might want to ensure the key is visible if it's manual.
            # For now, just process the content.
            # If we want to emulate [key], we can prepend a text node?
            # Or assume Docling logic handles it? Docling logic is generic.
            # I'll prepend the key if it exists.

            # Create a localized group or just add items?
            # Using _process_nodes with LIST_ITEM label.

            if key:
                # Add key as separate text or part of first item?
                # Simply processing nodes will add text.
                # I'll add the key manually first.
                doc.add_text(
                    parent=bib_group,
                    label=DocItemLabel.LIST_ITEM,
                    text=f"[{key}] ",
                    formatting=formatting
                )

            self._process_nodes(
                item_nodes,
                doc,
                bib_group,
                formatting,
                text_label=DocItemLabel.LIST_ITEM
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
                    text_parts.append(" ")  # Non-breaking space becomes regular space
                elif node.macroname == "item":
                    if node.nodeargd and node.nodeargd.argnlist:
                        arg = node.nodeargd.argnlist[0]
                        if arg:
                            opt_text = arg.latex_verbatim().strip("[] ")
                            text_parts.append(f"{opt_text}: ")
                elif node.macroname in ["%", "$", "&", "#", "_", "{", "}"]:
                    # Escaped characters
                    text_parts.append(node.macroname)
                # Handle custom macros in _nodes_to_text as well
                elif node.macroname in self._custom_macros:
                    expansion = self._custom_macros[node.macroname]
                    text_parts.append(expansion)
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