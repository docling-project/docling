import logging
import re
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Union

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
from pylatexenc.latex2text import LatexNodes2Text
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
        self._custom_macros: dict[str, str] = {}

        if isinstance(self.path_or_stream, BytesIO):
            raw_bytes = self.path_or_stream.getvalue()

            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    self.latex_text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not self.latex_text:
                _log.warning("Failed to decode LaTeX content, using replacement mode")
                self.latex_text = raw_bytes.decode("utf-8", errors="replace")
        elif isinstance(self.path_or_stream, Path):
            # Try multiple encodings for file
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    with open(self.path_or_stream, encoding=encoding) as f:
                        self.latex_text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                except FileNotFoundError:
                    _log.error(f"LaTeX file not found: {self.path_or_stream}")
                    break
                except OSError as e:
                    _log.error(f"Error reading LaTeX file: {e}")
                    break

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
        latex_text = re.sub(r"\\be\b", r"\\begin{equation}", latex_text)
        latex_text = re.sub(r"\\ee\b", r"\\end{equation}", latex_text)
        latex_text = re.sub(r"\\bea\b", r"\\begin{eqnarray}", latex_text)
        latex_text = re.sub(r"\\eea\b", r"\\end{eqnarray}", latex_text)
        latex_text = re.sub(r"\\beq\b", r"\\begin{equation}", latex_text)
        latex_text = re.sub(r"\\eeq\b", r"\\end{equation}", latex_text)

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

                        # Extract definition as raw LaTeX (for use in math expansion)
                        if hasattr(def_arg, "nodelist"):
                            # Get raw LaTeX content for proper math expansion
                            macro_def = def_arg.latex_verbatim()
                            # Only strip outermost braces if they wrap the entire content
                            if macro_def.startswith("{") and macro_def.endswith("}"):
                                macro_def = macro_def[1:-1]

                        if macro_name:  # Only register if we got a valid name
                            self._custom_macros[macro_name] = macro_def
                            _log.debug(
                                f"Registered custom macro: \\{macro_name} -> '{macro_def}'"
                            )

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
            # Split by paragraph breaks, keeping any content before first break
            parts = text.split("\n\n")

            # First part goes into current buffer (e.g., "." before a paragraph break)
            first_part = parts[0].strip()
            if first_part:
                text_buffer.append(first_part)

            # Flush buffer (now includes content before the break)
            flush_fn()

            # Remaining parts are separate paragraphs
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

    def _process_macro_node_inline(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem],
        formatting: Optional[Formatting],
        text_label: Optional[DocItemLabel],
        text_buffer: List[str],
        flush_fn: Callable[[], None],
    ):
        if node.macroname in ["%", "$", "&", "#", "_", "{", "}", "~"]:
            if node.macroname == "~":
                text_buffer.append(" ")  # Non-breaking space
            else:
                text_buffer.append(node.macroname)
        elif node.macroname == " ":
            text_buffer.append(" ")
        # Handle inline formatting macros - keep in buffer
        elif node.macroname in [
            "textbf",
            "textit",
            "emph",
            "texttt",
            "underline",
        ]:
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
        # Skip formatting switches that take arguments we don't want to output
        elif node.macroname in ["color", "definecolor", "colorlet"]:
            pass  # Ignore color commands entirely
        else:
            # Check if this is a structural macro that needs special handling
            structural_macros = {
                "section",
                "subsection",
                "subsubsection",
                "chapter",
                "part",
                "paragraph",
                "subparagraph",
                "caption",
                "label",
                "includegraphics",
                "bibliography",
                "title",
                "author",
                "maketitle",
                "footnote",
                "marginpar",
                "textsc",
                "textsf",
                "textrm",
                "textnormal",
                "mbox",
                "href",
                "newline",
                "hfill",
                "break",
                "centering",
                "textcolor",
                "colorbox",
                "item",
                "input",
                "include",
            }
            if node.macroname in structural_macros:
                # Structural macro - flush buffer and process with _process_macro
                flush_fn()
                self._process_macro(node, doc, parent, formatting, text_label)
            elif node.nodeargd and node.nodeargd.argnlist:
                # Unknown macro with arguments - extract all args as inline text
                inline_text = self._extract_all_macro_args_inline(node)
                if inline_text:
                    text_buffer.append(inline_text)
                else:
                    _log.debug(
                        f"Skipping unknown macro with no extractable content: {node.macroname}"
                    )
            else:
                _log.debug(
                    f"Skipping unknown macro without arguments: {node.macroname}"
                )

    def _process_math_node(
        self,
        node: LatexMathNode,
        doc: DoclingDocument,
        parent: Optional[NodeItem],
        text_buffer: List[str],
        flush_fn: Callable[[], None],
    ):
        is_display = getattr(node, "displaytype", None) == "display"

        if not is_display:
            math_verbatim = node.latex_verbatim()
            is_display = math_verbatim.startswith(
                (
                    "$$",
                    "\\[",
                    "\\begin{equation}",
                    "\\begin{align}",
                    "\\begin{gather}",
                    "\\begin{displaymath}",
                )
            )

        if is_display:
            flush_fn()
            math_text = self._clean_math(node.latex_verbatim(), "display")
            doc.add_text(parent=parent, label=DocItemLabel.FORMULA, text=math_text)
        else:
            # Expand custom macros in inline math for KaTeX compatibility
            text_buffer.append(self._expand_macros(node.latex_verbatim()))

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
                    self._process_macro_node_inline(
                        node,
                        doc,
                        parent,
                        formatting,
                        text_label,
                        text_buffer,
                        flush_text_buffer,
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
                continue  # Continue with next node

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
                    self._process_nodes(
                        arg.nodelist, doc, parent, formatting, text_label
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
                        self._process_nodes(
                            sub_nodes, doc, parent, formatting, text_label
                        )
                        _log.debug(f"Loaded input file: {input_path}")
                    except Exception as e:
                        _log.debug(f"Failed to load input file {filepath}: {e}")

        elif node.macroname in ["&", "%", "$", "#", "_", "{", "}"]:
            # Escaped symbols: \& -> &
            doc.add_text(
                parent=parent,
                text=node.macroname,
                formatting=formatting,
                label=(text_label or DocItemLabel.TEXT),
            )

        elif node.macroname in [
            "'",
            '"',
            "^",
            "`",
            "~",
            "=",
            ".",
            "c",
            "d",
            "b",
            "H",
            "k",
            "r",
            "t",
            "u",
            "v",
        ]:
            # Accents and diacritics
            try:
                text = LatexNodes2Text().nodelist_to_text([node])
                doc.add_text(
                    parent=parent,
                    text=text,
                    formatting=formatting,
                    label=(text_label or DocItemLabel.TEXT),
                )
            except Exception:
                pass

        elif node.macroname == "href":
            # \href{url}{text}
            if node.nodeargd and len(node.nodeargd.argnlist) >= 2:
                # url_arg = node.nodeargd.argnlist[0]
                text_arg = node.nodeargd.argnlist[1]

                if hasattr(text_arg, "nodelist"):
                    self._process_nodes(
                        text_arg.nodelist, doc, parent, formatting, text_label
                    )

        elif node.macroname in ["newline", "hfill", "break", "centering"]:
            if node.macroname == "newline":
                doc.add_text(
                    parent=parent,
                    text="\n",
                    formatting=formatting,
                    label=(text_label or DocItemLabel.TEXT),
                )

        elif node.macroname in [
            "bf",
            "it",
            "rm",
            "sc",
            "sf",
            "sl",
            "tt",
            "cal",
            "em",
            "tiny",
            "scriptsize",
            "footnotesize",
            "small",
            "large",
            "Large",
            "LARGE",
            "huge",
            "Huge",
            "color",  # \color{red} - ignore color switch
        ]:
            # Legacy formatting and size switches - ignore to preserve content flow (prevent "Unknown macro" skip)
            pass

        elif node.macroname in ["textcolor", "colorbox"]:
            # \textcolor{color}{text} - process only the text content (last argument)
            if node.nodeargd and node.nodeargd.argnlist:
                # Find the last non-None argument (the text content)
                for arg in reversed(node.nodeargd.argnlist):
                    if arg is not None and hasattr(arg, "nodelist"):
                        self._process_nodes(
                            arg.nodelist, doc, parent, formatting, text_label
                        )
                        break

        elif node.macroname == "item":
            pass

        else:
            # Unknown macro - try to extract content from arguments
            if node.nodeargd and node.nodeargd.argnlist:
                processed_any = False
                for arg in node.nodeargd.argnlist:
                    if hasattr(arg, "nodelist"):
                        self._process_nodes(
                            arg.nodelist, doc, parent, formatting, text_label
                        )
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

    def _process_table_macro_node(
        self,
        n: LatexMacroNode,
        source_latex: str,
        current_cell_nodes: List,
        finish_cell_fn: Callable[..., None],
        finish_row_fn: Callable[[], None],
        parse_brace_args_fn: Callable[[str], List[str]],
    ):
        if n.macroname == "\\":  # Row break
            finish_row_fn()

        elif n.macroname == "multicolumn":
            # \multicolumn{num_cols}{alignment}{content}
            # Extract from source using node position
            if hasattr(n, "pos") and n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_cols = int(args[0])
                    except (ValueError, TypeError):
                        num_cols = 1
                    content_text = args[2]  # Skip alignment arg[1]
                    if content_text:
                        current_cell_nodes.append(LatexCharsNode(chars=content_text))
                    finish_cell_fn(col_span=num_cols)
                else:
                    # Fallback
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname == "multirow":
            # \multirow{num_rows}{width}{content}
            if hasattr(n, "pos") and n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_rows = int(args[0])
                    except (ValueError, TypeError):
                        num_rows = 1
                    content_text = args[2]  # Skip width arg[1]
                    if content_text:
                        current_cell_nodes.append(LatexCharsNode(chars=content_text))
                    finish_cell_fn(row_span=num_rows)
                else:
                    # Fallback
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname in [
            "hline",
            "cline",
            "toprule",
            "midrule",
            "bottomrule",
            "cmidrule",
            "specialrule",
        ]:
            # Ignore rule lines for data extraction
            pass

        elif n.macroname in [
            "rule",
            "vspace",
            "hspace",
            "vskip",
            "hskip",
            "smallskip",
            "medskip",
            "bigskip",
            "strut",
            "phantom",
            "hphantom",
            "vphantom",
            "noalign",
        ]:
            # Ignore formatting commands - don't add to cell content
            pass

        elif n.macroname == "&":  # Cell break (if parsed as macro)
            finish_cell_fn()

        elif n.macroname in ["%", "$", "#", "_", "{", "}"]:
            # Escaped characters - add to current cell
            current_cell_nodes.append(n)

        else:
            current_cell_nodes.append(n)

    def _parse_table(self, node: LatexEnvironmentNode) -> Optional[TableData]:
        """Parse tabular environment into TableData with multicolumn/multirow support"""

        rows = []
        current_row = []
        current_cell_nodes: list = []

        # Get source latex for parsing multicolumn/multirow
        # These macros don't have their args parsed by pylatexenc by default
        source_latex = node.latex_verbatim()

        def parse_brace_args(text: str) -> list:
            """Parse {arg1}{arg2}{arg3} from text, return list of args"""
            args = []
            i = 0
            while i < len(text):
                if text[i] == "{":
                    depth = 1
                    start = i + 1
                    i += 1
                    while i < len(text) and depth > 0:
                        if text[i] == "{":
                            depth += 1
                        elif text[i] == "}":
                            depth -= 1
                        i += 1
                    args.append(text[start : i - 1])
                else:
                    i += 1
            return args

        def finish_cell(col_span: int = 1, row_span: int = 1):
            """Finish current cell with optional spanning"""
            text = self._nodes_to_text(current_cell_nodes).strip()
            cell = TableCell(
                text=text,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
            )
            # Store span info temporarily (will be set properly later)
            cell._col_span = col_span  # type: ignore[attr-defined]
            cell._row_span = row_span  # type: ignore[attr-defined]
            current_row.append(cell)
            current_cell_nodes.clear()

            # Add placeholder cells for column span
            for _ in range(col_span - 1):
                placeholder = TableCell(
                    text="",
                    start_row_offset_idx=0,
                    end_row_offset_idx=0,
                    start_col_offset_idx=0,
                    end_col_offset_idx=0,
                )
                placeholder._is_placeholder = True  # type: ignore[attr-defined]
                current_row.append(placeholder)

        def finish_row():
            """Finish current row and handle multirow placeholders"""
            if current_cell_nodes:
                finish_cell()  # Finish the last cell of the row
            if current_row:
                rows.append(current_row[:])  # Copy
            current_row.clear()

        # Guard against None nodelist
        if node.nodelist is None:
            return None

        for n in node.nodelist:
            if isinstance(n, LatexMacroNode):
                self._process_table_macro_node(
                    n,
                    source_latex,
                    current_cell_nodes,
                    finish_cell,
                    finish_row,
                    parse_brace_args,
                )

            elif isinstance(n, LatexCharsNode):
                text = n.chars
                if "&" in text:
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
                if hasattr(n, "specials_chars") and n.specials_chars == "&":
                    finish_cell()
                else:
                    current_cell_nodes.append(n)

        finish_row()

        if not rows:
            return None

        # Calculate dimensions
        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0

        # Build flat cell list with proper indices and spans
        flat_cells = []
        for i, row in enumerate(rows):
            for j in range(num_cols):
                if j < len(row):
                    cell = row[j]
                    # Skip placeholder cells created by multicolumn
                    if getattr(cell, "_is_placeholder", False):
                        continue
                else:
                    cell = TableCell(
                        text="",
                        start_row_offset_idx=0,
                        end_row_offset_idx=0,
                        start_col_offset_idx=0,
                        end_col_offset_idx=0,
                    )

                # Set row/col indices
                cell.start_row_offset_idx = i
                cell.start_col_offset_idx = j

                # Apply spans if stored
                col_span = getattr(cell, "_col_span", 1)
                row_span = getattr(cell, "_row_span", 1)
                cell.end_row_offset_idx = i + row_span
                cell.end_col_offset_idx = j + col_span

                flat_cells.append(cell)

        return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=flat_cells)

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

        # Pre process to group by bibitem
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
                elif node.macroname in ["%", "$", "&", "#", "_", "{", "}"]:
                    # Escaped characters
                    text_parts.append(node.macroname)
                # Handle custom macros in _nodes_to_text as well
                elif node.macroname in self._custom_macros:
                    expansion = self._custom_macros[node.macroname]
                    text_parts.append(expansion)
                else:
                    # Unknown macro - extract all arguments inline
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
                # Expand custom macros in math for KaTeX compatibility
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

    def _extract_macro_arg(self, node: LatexMacroNode) -> str:
        """Extract text from macro argument (last argument only)"""
        if node.nodeargd and node.nodeargd.argnlist:
            arg = node.nodeargd.argnlist[-1]
            if arg:
                if hasattr(arg, "nodelist"):
                    return self._nodes_to_text(arg.nodelist)
                return arg.latex_verbatim().strip("{} ")
        return ""

    def _extract_macro_arg_by_index(self, node: LatexMacroNode, index: int) -> str:
        """Extract text from macro argument by index (0-based)"""
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if arg:
                    if hasattr(arg, "nodelist"):
                        return self._nodes_to_text(arg.nodelist)
                    return arg.latex_verbatim().strip("{} ")
        return ""

    def _extract_macro_arg_nodes(self, node: LatexMacroNode, index: int) -> list:
        """Extract node list from macro argument by index (0-based)"""
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if arg and hasattr(arg, "nodelist"):
                    return arg.nodelist
        return []

    def _extract_all_macro_args_inline(self, node: LatexMacroNode) -> str:
        """Extract all macro arguments as inline text, concatenated with spaces."""
        if not node.nodeargd or not node.nodeargd.argnlist:
            return ""

        parts = []
        for arg in node.nodeargd.argnlist:
            if arg is not None:
                if hasattr(arg, "nodelist"):
                    text = self._nodes_to_text(arg.nodelist)
                    if text:
                        parts.append(text)
                else:
                    text = arg.latex_verbatim().strip("{} ")
                    if text:
                        parts.append(text)

        return " ".join(parts)

    def _is_text_only_group(self, node: LatexGroupNode) -> bool:
        """Check if a group contains only text-like content (no structural elements)."""
        if not node.nodelist:
            return True

        # Macros that indicate structural content
        structural_macros = {
            "section",
            "subsection",
            "subsubsection",
            "chapter",
            "part",
            "caption",
            "label",
            "includegraphics",
            "bibliography",
            "title",
            "author",
            "maketitle",
            "footnote",
            "marginpar",
        }

        for n in node.nodelist:
            if isinstance(n, LatexEnvironmentNode):
                # Environments are usually structural
                return False
            elif isinstance(n, LatexMacroNode):
                if n.macroname in structural_macros:
                    return False
            elif isinstance(n, LatexGroupNode):
                # Recursively check nested groups
                if not self._is_text_only_group(n):
                    return False

        return True

    def _expand_macros(self, latex_str: str) -> str:
        """Expand custom macros in LaTeX string for KaTeX/MathJax compatibility"""
        for macro_name, macro_def in self._custom_macros.items():
            # Replace \macroname with its definition (word boundary to avoid partial matches)
            # Use lambda to avoid backslash interpretation in replacement string
            latex_str = re.sub(
                rf"\\{re.escape(macro_name)}(?![a-zA-Z])",
                lambda m: macro_def,
                latex_str,
            )
        return latex_str

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

        # Expand custom macros for KaTeX/MathJax compatibility
        latex_str = self._expand_macros(latex_str)

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
