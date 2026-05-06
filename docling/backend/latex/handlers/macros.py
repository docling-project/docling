import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path
    from typing import Any

import pypdfium2
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    ImageRef,
    NodeItem,
)
from PIL import Image
from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexWalker,
    LatexWalkerParseError,
)

from docling.backend.latex.constants import (
    MACROS_ACCENTS,
    MACROS_CITATION,
    MACROS_COLOR,
    MACROS_COLOR_INLINE,
    MACROS_ESCAPED,
    MACROS_HEADING,
    MACROS_IGNORED,
    MACROS_INLINE_VERBATIM,
    MACROS_LEGACY_FORMATTING,
    MACROS_NEWCOMMAND,
    MACROS_PREAMBLE_METADATA,
    MACROS_SPACING,
    MACROS_STRUCTURAL,
    MACROS_TEXT_FORMATTING,
    MACROS_TEXT_STYLE,
)

_log = logging.getLogger(__name__)


class MacroHandlerMixin:
    if TYPE_CHECKING:
        path_or_stream: "BytesIO | Path"
        _input_stack: set[str]
        _custom_macros: dict[str, str]
        _custom_macro_num_args: dict[str, int]
        labels: dict[str, bool]

        def _process_nodes(
            self,
            nodes: "Any",
            doc: "Any",
            parent: "Any" = ...,
            formatting: "Any" = ...,
            text_label: "Any" = ...,
        ) -> None: ...
        def _nodes_to_text(self, nodes: "Any") -> str: ...

    def _preprocess_custom_macros(self, latex_text: str) -> str:
        latex_text = re.sub(r"\\be\b", r"\\begin{equation}", latex_text)
        latex_text = re.sub(r"\\ee\b", r"\\end{equation}", latex_text)
        latex_text = re.sub(r"\\bea\b", r"\\begin{eqnarray}", latex_text)
        latex_text = re.sub(r"\\eea\b", r"\\end{eqnarray}", latex_text)
        latex_text = re.sub(r"\\beq\b", r"\\begin{equation}", latex_text)
        latex_text = re.sub(r"\\eeq\b", r"\\end{equation}", latex_text)

        return latex_text

    def _extract_custom_macros(self, nodes, depth: int = 0):
        if nodes is None or depth > 10:
            return

        for node in nodes:
            if isinstance(node, LatexMacroNode) and node.macroname in MACROS_NEWCOMMAND:
                if node.nodeargd and node.nodeargd.argnlist:
                    argnlist = node.nodeargd.argnlist

                    name_arg = argnlist[1] if len(argnlist) > 1 else None
                    num_args_arg = argnlist[2] if len(argnlist) > 2 else None

                    def_arg = None
                    for arg in reversed(argnlist):
                        if arg is not None:
                            def_arg = arg
                            break

                    if name_arg and def_arg and name_arg is not def_arg:
                        macro_name_raw = name_arg.latex_verbatim()
                        macro_name = macro_name_raw.strip("{} \n\t")
                        if macro_name.startswith("\\"):
                            macro_name = macro_name[1:]

                        macro_def = ""
                        if hasattr(def_arg, "nodelist"):
                            macro_def = def_arg.latex_verbatim()
                            if macro_def.startswith("{") and macro_def.endswith("}"):
                                macro_def = macro_def[1:-1]
                        else:
                            macro_def = def_arg.latex_verbatim().strip("{} ")

                        if macro_name:
                            self._custom_macros[macro_name] = macro_def
                            self._custom_macro_num_args[macro_name] = (
                                self._parse_custom_macro_num_args(num_args_arg)
                            )
                            _log.debug(
                                f"Registered custom macro: \\{macro_name} -> '{macro_def}'"
                            )

            if hasattr(node, "nodelist") and node.nodelist:
                self._extract_custom_macros(node.nodelist, depth + 1)
            if hasattr(node, "nodeargd") and node.nodeargd:
                argnlist = getattr(node.nodeargd, "argnlist", None)
                if argnlist:
                    for arg in argnlist:
                        if hasattr(arg, "nodelist") and arg.nodelist:
                            self._extract_custom_macros(arg.nodelist, depth + 1)

    def _extract_preamble_metadata(self, nodes, doc: DoclingDocument, depth: int = 0):
        if nodes is None or depth > 10:
            return

        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                return

            if (
                isinstance(node, LatexMacroNode)
                and node.macroname in MACROS_PREAMBLE_METADATA
            ):
                text = self._extract_macro_arg(node)
                if text:
                    if node.macroname == "title":
                        doc.add_text(label=DocItemLabel.TITLE, text=text)
                    else:
                        doc.add_text(label=DocItemLabel.TEXT, text=text)

            if hasattr(node, "nodelist") and node.nodelist:
                self._extract_preamble_metadata(node.nodelist, doc, depth + 1)
            if hasattr(node, "nodeargd") and node.nodeargd:
                argnlist = getattr(node.nodeargd, "argnlist", None)
                if argnlist:
                    for arg in argnlist:
                        if hasattr(arg, "nodelist") and arg.nodelist:
                            self._extract_preamble_metadata(
                                arg.nodelist, doc, depth + 1
                            )

    def _process_macro_node_inline(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: list[str],
        flush_fn: Callable[[], None],
        following_nodes=None,
    ) -> int:
        if node.macroname in MACROS_INLINE_VERBATIM:
            if node.macroname == "~":
                text_buffer.append(" ")
            else:
                text_buffer.append(node.macroname)
        elif node.macroname == " ":
            text_buffer.append(" ")
        elif node.macroname in MACROS_TEXT_FORMATTING:
            formatted_text = self._extract_macro_arg(node)
            if formatted_text:
                text_buffer.append(formatted_text)
        elif node.macroname in self._custom_macros:
            expansion, consumed = self._expand_custom_macro_invocation(
                node, following_nodes or []
            )
            if expansion:
                _log.debug(
                    f"Expanding custom macro \\{node.macroname} -> '{expansion}'"
                )
                if self._custom_macro_num_args.get(node.macroname, 0) > 0:
                    text_buffer.append(self._parse_latex_fragment_to_text(expansion))
                else:
                    text_buffer.append(expansion)
            return consumed
        elif node.macroname in MACROS_CITATION:
            ref_arg = self._extract_macro_arg(node)
            if ref_arg:
                text_buffer.append(f"[{ref_arg}]")
        elif node.macroname == "url":
            url_text = self._extract_macro_arg(node)
            if url_text:
                text_buffer.append(url_text)
        elif node.macroname in MACROS_COLOR:
            pass
        elif node.macroname in MACROS_TEXT_STYLE:
            formatted_text = self._extract_macro_arg(node)
            if formatted_text:
                text_buffer.append(formatted_text)
        elif node.macroname in MACROS_COLOR_INLINE:
            # Skip the color argument; the text content is always the last arg
            if node.nodeargd and node.nodeargd.argnlist:
                text_arg = node.nodeargd.argnlist[-1]
                if text_arg is not None and hasattr(text_arg, "nodelist"):
                    text = self._nodes_to_text(text_arg.nodelist)
                    if text:
                        text_buffer.append(text)
        else:
            if node.macroname in MACROS_STRUCTURAL:
                flush_fn()
                self._process_macro(node, doc, parent, formatting, text_label)
            elif node.macroname in MACROS_SPACING or node.macroname in MACROS_IGNORED:
                # Spacing and ignored commands are silently discarded along with
                # their arguments (e.g. \vspace{-1mm} should not emit "-1mm")
                _log.debug(
                    f"Discarding spacing/ignored macro and its arguments: {node.macroname}"
                )
            elif node.nodeargd and node.nodeargd.argnlist:
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
        return 0

    def _process_macro(  # noqa: C901
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        text_label: DocItemLabel | None = None,
    ):
        if node.macroname in MACROS_HEADING:
            title = self._extract_macro_arg(node)
            if title:
                level = self._get_heading_level(node.macroname)
                doc.add_heading(parent=parent, text=title, level=level)

        elif node.macroname == "title":
            title = self._extract_macro_arg(node)
            if title:
                doc.add_text(parent=parent, label=DocItemLabel.TITLE, text=title)

        elif node.macroname in ["author", "date"]:
            meta_text = self._extract_macro_arg(node)
            if meta_text:
                doc.add_text(parent=parent, label=DocItemLabel.TEXT, text=meta_text)

        elif node.macroname in ["thanks", "maketitle"]:
            pass

        elif node.macroname in MACROS_TEXT_STYLE:
            if node.nodeargd and node.nodeargd.argnlist:
                arg = node.nodeargd.argnlist[-1]
                if hasattr(arg, "nodelist"):
                    self._process_nodes(
                        arg.nodelist, doc, parent, formatting, text_label
                    )

        elif node.macroname in MACROS_CITATION:
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
                        base_dir = self.path_or_stream.parent.resolve()
                        img_full_path = self.path_or_stream.parent / img_path
                        try:
                            if not img_full_path.resolve().is_relative_to(base_dir):
                                _log.warning(
                                    f"Path traversal attempt blocked for image: {img_path}"
                                )
                                raise ValueError("Path traversal not allowed")
                        except ValueError:
                            _log.warning(
                                f"Invalid path for image (different drive or traversal): {img_path}"
                            )
                            raise
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

        elif node.macroname in MACROS_IGNORED:
            pass

        elif node.macroname in ["input", "include"]:
            from pylatexenc.latexwalker import LatexWalker

            filepath = self._extract_macro_arg(node)
            if filepath and isinstance(self.path_or_stream, Path):
                base_dir = self.path_or_stream.parent.resolve()
                input_path = self.path_or_stream.parent / filepath
                if not input_path.suffix:
                    input_path = input_path.with_suffix(".tex")

                try:
                    if not input_path.resolve().is_relative_to(base_dir):
                        _log.warning(
                            f"Path traversal attempt blocked for input file: {filepath}"
                        )
                        return
                except ValueError:
                    _log.warning(
                        f"Invalid path for input file (different drive or traversal): {filepath}"
                    )
                    return

                resolved = str(input_path.resolve())
                if resolved in self._input_stack:
                    _log.warning(f"Circular \\input detected: {filepath}")
                elif len(self._input_stack) >= 10:
                    _log.warning(
                        f"\\input depth limit (10) reached, skipping: {filepath}"
                    )
                elif input_path.exists():
                    self._input_stack.add(resolved)
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
                    finally:
                        self._input_stack.discard(resolved)

        elif node.macroname in MACROS_ESCAPED:
            doc.add_text(
                parent=parent,
                text=node.macroname,
                formatting=formatting,
                label=(text_label or DocItemLabel.TEXT),
            )

        elif node.macroname in MACROS_ACCENTS:
            try:
                from pylatexenc.latex2text import LatexNodes2Text

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
            if node.nodeargd and len(node.nodeargd.argnlist) >= 2:
                url_arg = node.nodeargd.argnlist[0]
                text_arg = node.nodeargd.argnlist[1]

                url_text = ""
                if url_arg is not None:
                    if hasattr(url_arg, "nodelist"):
                        url_text = self._nodes_to_text(url_arg.nodelist)
                    else:
                        url_text = url_arg.latex_verbatim().strip("{} ")

                display_text = ""
                if text_arg is not None:
                    if hasattr(text_arg, "nodelist"):
                        display_text = self._nodes_to_text(text_arg.nodelist)
                    else:
                        display_text = text_arg.latex_verbatim().strip("{} ")

                if url_text and display_text:
                    link_text = f"[{display_text}]({url_text})"
                elif url_text:
                    link_text = url_text
                elif display_text:
                    link_text = display_text
                else:
                    link_text = ""

                if link_text:
                    doc.add_text(
                        parent=parent,
                        label=DocItemLabel.REFERENCE,
                        text=link_text,
                        formatting=formatting,
                    )

        elif node.macroname in MACROS_SPACING:
            if node.macroname == "newline":
                doc.add_text(
                    parent=parent,
                    text="\n",
                    formatting=formatting,
                    label=(text_label or DocItemLabel.TEXT),
                )

        elif node.macroname in MACROS_LEGACY_FORMATTING:
            pass

        elif node.macroname in ["textcolor", "colorbox"]:
            if node.nodeargd and node.nodeargd.argnlist:
                for arg in reversed(node.nodeargd.argnlist):
                    if arg is not None and hasattr(arg, "nodelist"):
                        self._process_nodes(
                            arg.nodelist, doc, parent, formatting, text_label
                        )
                        break

        elif node.macroname == "item":
            pass

        else:
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

    def _extract_macro_arg(self, node: LatexMacroNode) -> str:
        if node.nodeargd and node.nodeargd.argnlist:
            arg = node.nodeargd.argnlist[-1]
            if arg:
                if hasattr(arg, "nodelist"):
                    return self._nodes_to_text(arg.nodelist)
                return arg.latex_verbatim().strip("{} ")
        return ""

    def _extract_macro_arg_by_index(self, node: LatexMacroNode, index: int) -> str:
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if arg:
                    if hasattr(arg, "nodelist"):
                        return self._nodes_to_text(arg.nodelist)
                    return arg.latex_verbatim().strip("{} ")
        return ""

    def _extract_macro_arg_nodes(self, node: LatexMacroNode, index: int) -> list:
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if arg and hasattr(arg, "nodelist"):
                    return arg.nodelist
        return []

    def _extract_all_macro_args_inline(self, node: LatexMacroNode) -> str:
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

    def _expand_macros(self, latex_str: str) -> str:
        for macro_name, macro_def in self._custom_macros.items():
            if self._custom_macro_num_args.get(macro_name, 0) > 0:
                continue
            latex_str = re.sub(
                rf"\\{re.escape(macro_name)}(?![a-zA-Z])",
                lambda m: macro_def,
                latex_str,
            )
        return latex_str

    def _parse_custom_macro_num_args(self, num_args_arg) -> int:
        if num_args_arg is None:
            return 0

        raw = num_args_arg.latex_verbatim().strip("{}[] \n\t")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _extract_custom_macro_invocation_args(
        self, following_nodes, expected_arg_count: int
    ) -> tuple[list[str], int]:
        if expected_arg_count <= 0:
            return [], 0

        arg_values: list[str] = []
        consumed = 0

        for next_node in following_nodes:
            if len(arg_values) >= expected_arg_count:
                break

            if isinstance(next_node, LatexCharsNode) and not next_node.chars.strip():
                consumed += 1
                continue

            if isinstance(next_node, LatexGroupNode):
                arg_values.append(self._nodes_to_text(next_node.nodelist or []))
                consumed += 1
                continue

            break

        return arg_values, consumed

    def _render_custom_macro_expansion(
        self, macro_name: str, arg_values: list[str]
    ) -> str:
        expansion = self._custom_macros[macro_name]
        for idx in range(len(arg_values), 0, -1):
            expansion = expansion.replace(f"#{idx}", arg_values[idx - 1])
        return expansion

    def _parse_latex_fragment_to_text(self, latex_fragment: str) -> str:
        try:
            walker = LatexWalker(latex_fragment, tolerant_parsing=True)
            parsed_nodes, _, _ = walker.get_latex_nodes()
        except LatexWalkerParseError:
            return latex_fragment

        return self._nodes_to_text(parsed_nodes)

    def _expand_custom_macro_invocation(
        self, node: LatexMacroNode, following_nodes
    ) -> tuple[str, int]:
        expected_arg_count = self._custom_macro_num_args.get(node.macroname, 0)
        if expected_arg_count <= 0:
            return self._custom_macros[node.macroname], 0

        arg_values, consumed = self._extract_custom_macro_invocation_args(
            following_nodes, expected_arg_count
        )
        if len(arg_values) < expected_arg_count:
            return self._custom_macros[node.macroname], 0

        return self._render_custom_macro_expansion(node.macroname, arg_values), consumed

    def _get_heading_level(self, macroname: str) -> int:
        levels = {
            "part": 1,
            "chapter": 1,
            "section": 1,
            "subsection": 2,
            "subsubsection": 3,
            "paragraph": 4,
            "subparagraph": 5,
        }
        return levels.get(macroname, 1)
