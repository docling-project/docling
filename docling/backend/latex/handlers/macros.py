# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    Formatting,
    ImageRef,
    NodeItem,
)
from PIL import Image

from docling.backend.latex.constants import (
    MACROS_CITATION,
    MACROS_COLOR,
    MACROS_COLOR_INLINE,
    MACROS_ESCAPED,
    MACROS_HEADING,
    MACROS_IGNORED,
    MACROS_INLINE_VERBATIM,
    MACROS_LEGACY_FORMATTING,
    MACROS_METADATA,
    MACROS_NEWCOMMAND,
    MACROS_SPACING,
    MACROS_STRUCTURAL,
    MACROS_TEXT_FORMATTING,
    MACROS_TEXT_STYLE,
)
from docling.backend.latex.handlers.math import InlineFormula
from docling.backend.latex.utils.encoding import decode_latex_content
from docling.backend.latex.utils.latex_context import LATEX_CONTEXT_DB
from docling.backend.latex.utils.text import iter_child_nodelists
from docling.backend.utils.image import normalize_image_for_png

if TYPE_CHECKING:
    from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexGroupNode,
        LatexMacroNode,
        LatexWalker,
        LatexWalkerParseError,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__

_log = logging.getLogger(__name__)

# pypdfium2 ships with the PDF extras, not with format-latex. Only \includegraphics
# of a .pdf file needs it, so guard the import and report it on that path instead
# of breaking every LaTeX conversion at import time.
# See https://github.com/docling-project/docling/issues/3613.
_PYPDFIUM2_AVAILABLE: bool = False
_PYPDFIUM2_IMPORT_ERROR: ImportError | None = None
try:  # pragma: no cover - import-time guard
    import pypdfium2

    _PYPDFIUM2_AVAILABLE = True
except ImportError as e:  # pragma: no cover - import-time guard
    _PYPDFIUM2_IMPORT_ERROR = e

_PYPDFIUM2_INSTALL_HINT = (
    "The 'pypdfium2' package is required to embed PDF images from LaTeX "
    "sources. Install it with `pip install 'docling-slim[format-pdf-pypdfium2]'`."
)


def _skip_latex_space(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _latex_group_end(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    index = start
    while index < len(text):
        if (
            text[index] == "\\"
            and index + 1 < len(text)
            and text[index + 1] in "\\{}[]%"
        ):
            index += 2
            continue
        if text[index] == "%":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return len(text)


def _strip_latex_comments(text: str) -> str:
    command_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@"
    parts: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text) and text[index + 1] in "\\%":
            parts.append(text[index : index + 2])
            index += 2
            continue
        if text[index] == "%":
            newline = text.find("\n", index)
            next_index = len(text) if newline < 0 else newline + 1
            while next_index < len(text) and text[next_index] == "%":
                newline = text.find("\n", next_index)
                next_index = len(text) if newline < 0 else newline + 1
            if next_index < len(text) and text[next_index] in command_letters:
                word_start = index
                while word_start > 0 and text[word_start - 1] in command_letters:
                    word_start -= 1
                slash = word_start - 1
                if slash >= 0 and text[slash] == "\\":
                    while slash > 0 and text[slash - 1] == "\\":
                        slash -= 1
                    if (word_start - slash) % 2 == 1:
                        # A space ends the command without taking its argument.
                        parts.append(" ")
            index = next_index
            continue
        parts.append(text[index])
        index += 1
    return "".join(parts)


def _simple_declaration_end(text: str, command: str, cursor: int) -> int | None:
    cursor = _skip_latex_space(text, cursor)
    if command == "newif":
        if cursor >= len(text) or text[cursor] != "\\":
            return None
        cursor += 1
        while cursor < len(text) and text[cursor].isalpha():
            cursor += 1
        return cursor

    group_count = 9 if command == "newtheoremstyle" else 1
    for _ in range(group_count):
        cursor = _skip_latex_space(text, cursor)
        if cursor >= len(text) or text[cursor] != "{":
            return None
        cursor = _latex_group_end(text, cursor, "{", "}")
    if command == "newcounter":
        optional = _skip_latex_space(text, cursor)
        if optional < len(text) and text[optional] == "[":
            cursor = _latex_group_end(text, optional, "[", "]")
    return cursor


class MacroHandlerMixin:
    if TYPE_CHECKING:
        _source_dir: Path | None
        _input_stack: set[str]
        _custom_macros: dict[str, str]
        _custom_macro_num_args: dict[str, int]
        _masked_macro_spans: list[tuple[int, int]]
        _suppress_image_path_caption: bool
        labels: dict[str, bool]

        def _process_nodes(
            self,
            nodes: Any,
            doc: Any,
            parent: Any = ...,
            formatting: Any = ...,
            text_label: Any = ...,
            heading_level: int | None = ...,
        ) -> None: ...
        def _nodes_to_text(self, nodes: Any) -> str: ...
        def _char_macro_to_text(self, node: Any) -> str | None: ...
        def _item_term(self, node: Any) -> str: ...

    def _preprocess_custom_macros(self, latex_text: str) -> str:
        """Expand legacy shortcuts without changing macro definitions."""
        self._masked_macro_spans = self._macro_definition_spans(latex_text)
        shortcuts: dict[str, str] = {}
        environment_alias = re.compile(r"\\(?:begin|end)\{[^{}]+\}|\$\$")
        fuzz_assignment = re.compile(
            r"\\[vh]fuzz\s*=\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*"
            r"(?:pt|pc|in|bp|cm|mm|dd|cc|sp|em|ex)\b"
        )

        def replace(segment: str) -> str:
            if shortcuts:
                names = "|".join(
                    re.escape(name) for name in sorted(shortcuts, key=len, reverse=True)
                )
                pattern = re.compile(r"\\(?:" + names + r")(?![A-Za-z@])")
                segment = pattern.sub(
                    lambda match: shortcuts[match.group()[1:]], segment
                )
            return fuzz_assignment.sub("", segment)

        pieces: list[str] = []
        cursor = 0
        for start, end in self._masked_macro_spans:
            pieces.append(replace(latex_text[cursor:start]))
            definition = latex_text[start:end]
            if definition.startswith(
                (r"\newcommand", r"\renewcommand", r"\providecommand")
            ):
                definition_nodes, _, _ = LatexWalker(
                    definition, tolerant_parsing=True, latex_context=LATEX_CONTEXT_DB
                ).get_latex_nodes()
                self._extract_custom_macros(definition_nodes)
            elif match := re.fullmatch(
                r"\\(?:def|edef|gdef|xdef)\s*\\([A-Za-z@]+)\s*\{(.*)\}",
                definition,
                re.DOTALL,
            ):
                # A source definition can replace a built-in accent macro.
                # Register zero-argument definitions before the main parse so
                # the parser does not consume a following math delimiter.
                self._custom_macros[match.group(1)] = _strip_latex_comments(
                    match.group(2)
                )
                self._custom_macro_num_args[match.group(1)] = 0
            for name, expansion in self._custom_macros.items():
                body = expansion.strip()
                if self._custom_macro_num_args.get(
                    name
                ) == 0 and environment_alias.fullmatch(body):
                    shortcuts[name] = body
                else:
                    shortcuts.pop(name, None)
            pieces.append(re.sub(r"[^\n]", " ", definition))
            cursor = end
        pieces.append(replace(latex_text[cursor:]))
        return "".join(pieces)

    @staticmethod
    def _macro_definition_spans(text: str) -> list[tuple[int, int]]:
        """Find balanced definitions that may contain legacy shortcut names."""
        pattern = re.compile(
            r"\\(?P<command>newcommand|renewcommand|providecommand|"
            r"newenvironment|newtheoremstyle|newtheorem|newlength|newcounter|"
            r"newif|def|edef|gdef|xdef)\b"
        )

        verbatim_start = re.compile(r"\\begin\{(verbatim\*?|lstlisting|minted)\}")
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(text):
            if text[index] == "%":
                newline = text.find("\n", index)
                index = len(text) if newline < 0 else newline + 1
                continue
            if text[index] != "\\":
                index += 1
                continue
            if index + 1 < len(text) and text[index + 1] in "\\%":
                index += 2
                continue
            if match := verbatim_start.match(text, index):
                end = text.find(r"\end{" + match.group(1) + "}", match.end())
                index = len(text) if end < 0 else end + len(match.group(0))
                continue
            match = pattern.match(text, index)
            if match is None:
                index += 1
                continue
            cursor = match.end()
            command = match.group("command")
            if command in {"newif", "newlength", "newcounter", "newtheoremstyle"}:
                end = _simple_declaration_end(text, command, cursor)
                if end is not None:
                    spans.append((match.start(), end))
                index = end if end is not None else match.end()
                continue
            if command in MACROS_NEWCOMMAND | {"newenvironment", "newtheorem"}:
                cursor = _skip_latex_space(text, cursor)
                if cursor < len(text) and text[cursor] == "*":
                    cursor += 1
                cursor = _skip_latex_space(text, cursor)
                if cursor < len(text) and text[cursor] == "{":
                    cursor = _latex_group_end(text, cursor, "{", "}")
                elif cursor < len(text) and text[cursor] == "\\":
                    cursor += 1
                    while cursor < len(text) and text[cursor].isalpha():
                        cursor += 1
                option_count = 2 if command != "newtheorem" else 1
                for _ in range(option_count):
                    cursor = _skip_latex_space(text, cursor)
                    if cursor < len(text) and text[cursor] == "[":
                        cursor = _latex_group_end(text, cursor, "[", "]")
            else:
                while cursor < len(text) and text[cursor] != "{":
                    cursor += 1
            cursor = _skip_latex_space(text, cursor)
            if cursor < len(text) and text[cursor] == "{":
                end = _latex_group_end(text, cursor, "{", "}")
                if command == "newenvironment":
                    second = _skip_latex_space(text, end)
                    if second < len(text) and text[second] == "{":
                        end = _latex_group_end(text, second, "{", "}")
                elif command == "newtheorem":
                    second = _skip_latex_space(text, end)
                    if second < len(text) and text[second] == "[":
                        end = _latex_group_end(text, second, "[", "]")
                spans.append((match.start(), end))
                index = end
            else:
                index = match.end()
        return spans

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
                        if isinstance(def_arg, LatexGroupNode):
                            macro_def = def_arg.latex_verbatim()
                            if macro_def.startswith("{") and macro_def.endswith("}"):
                                macro_def = macro_def[1:-1]
                        else:
                            macro_def = def_arg.latex_verbatim().strip("{} ")

                        if macro_name:
                            self._custom_macros[macro_name] = _strip_latex_comments(
                                macro_def
                            )
                            self._custom_macro_num_args[macro_name] = (
                                self._parse_custom_macro_num_args(num_args_arg)
                            )
                            _log.debug(
                                f"Registered custom macro: \\{macro_name} -> '{macro_def}'"
                            )

            for child_nodes in iter_child_nodelists(node):
                self._extract_custom_macros(child_nodes, depth + 1)

    def _process_item_macro_inline(
        self,
        node: LatexMacroNode,
        text_buffer: list[str | InlineFormula],
        following_nodes=None,
    ) -> None:
        term = self._item_term(node)
        if term:
            text_buffer.append(f"{term}: ")
            if following_nodes and isinstance(following_nodes[0], LatexCharsNode):
                following_nodes[0].chars = following_nodes[0].chars.lstrip()

    def _append_custom_math_expansion(
        self,
        expansion: str,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: list[str | InlineFormula],
        flush_fn: Callable[[], None],
    ) -> bool:
        if r"\ensuremath" not in expansion and "$" not in expansion:
            return False
        try:
            parsed_nodes, _, _ = LatexWalker(
                expansion,
                tolerant_parsing=True,
                latex_context=LATEX_CONTEXT_DB,
            ).get_latex_nodes()
        except LatexWalkerParseError:
            return False
        if not self._contains_math(parsed_nodes):
            return False
        self._append_inline_nodes(
            parsed_nodes, doc, parent, formatting, text_label, text_buffer, flush_fn
        )
        return True

    def _append_ensuremath(
        self, node: LatexMacroNode, text_buffer: list[str | InlineFormula]
    ) -> None:
        if not node.nodeargd or not node.nodeargd.argnlist:
            return
        argument = node.nodeargd.argnlist[-1]
        if isinstance(argument, LatexGroupNode):
            math_text = argument.latex_verbatim()[1:-1].strip()
            if math_text:
                text_buffer.append(InlineFormula(self._expand_macros(math_text)))

    def _process_macro_node_inline(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None,
        text_label: DocItemLabel | None,
        text_buffer: list[str | InlineFormula],
        flush_fn: Callable[[], None],
        following_nodes=None,
    ) -> int:
        char_text = self._char_macro_to_text(node)
        if char_text is not None:
            text_buffer.append(char_text)
        elif node.macroname in MACROS_INLINE_VERBATIM:
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
        elif node.macroname == "ensuremath":
            self._append_ensuremath(node, text_buffer)
        elif node.macroname in self._custom_macros:
            expansion, consumed = self._expand_custom_macro_invocation(
                node, following_nodes or []
            )
            if expansion:
                _log.debug(
                    f"Expanding custom macro \\{node.macroname} -> '{expansion}'"
                )
                if self._append_custom_math_expansion(
                    expansion,
                    doc,
                    parent,
                    formatting,
                    text_label,
                    text_buffer,
                    flush_fn,
                ):
                    if node.latex_verbatim()[-1].isspace():
                        text_buffer.append(" ")
                    return consumed
                if self._custom_macro_num_args.get(node.macroname, 0) > 0:
                    text_buffer.append(self._parse_latex_fragment_to_text(expansion))
                else:
                    text_buffer.append(expansion)
            return consumed
        elif node.macroname in MACROS_CITATION:
            ref_arg = self._extract_macro_arg(node)
            if ref_arg:
                text_buffer.append(f"[{ref_arg}]")
        elif node.macroname in ("url", "href"):
            link_text = (
                self._href_link_text(node)
                if node.macroname == "href"
                else self._extract_macro_arg(node)
            )
            if link_text:
                text_buffer.append(link_text)
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
                if isinstance(text_arg, LatexGroupNode):
                    text = self._nodes_to_text(text_arg.nodelist)
                    if text:
                        text_buffer.append(text)
        elif node.macroname == "item":
            flush_fn()
            self._process_item_macro_inline(node, text_buffer, following_nodes)
        elif node.macroname == "label":
            self._process_macro(node, doc, parent, formatting, text_label)
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
            if node.nodeargd is not None and node.nodeargd.argnlist:
                argument = node.nodeargd.argnlist[-1]
                if isinstance(argument, LatexGroupNode):
                    self._process_nodes(
                        argument.nodelist,
                        doc,
                        parent,
                        formatting,
                        heading_level=self._get_heading_level(node.macroname),
                    )

        elif node.macroname in MACROS_METADATA:
            self._metadata_handler.process_macro(node, doc, parent, formatting)

        elif node.macroname == "maketitle":
            pass

        elif node.macroname in MACROS_TEXT_STYLE:
            if node.nodeargd and node.nodeargd.argnlist:
                arg = node.nodeargd.argnlist[-1]
                if isinstance(arg, LatexGroupNode):
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

        elif node.macroname in {"caption", "captionof"}:
            self._add_caption(node, doc, parent, formatting)

        elif node.macroname == "marginpar":
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
                    if self._source_dir is not None:
                        base_dir = self._source_dir.resolve()
                        img_full_path = self._source_dir / img_path
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
                                if not _PYPDFIUM2_AVAILABLE:
                                    raise ImportError(
                                        _PYPDFIUM2_INSTALL_HINT
                                    ) from _PYPDFIUM2_IMPORT_ERROR
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
                            image = ImageRef.from_pil(
                                image=normalize_image_for_png(pil_image), dpi=int(dpi)
                            )
                except Exception as e:
                    _log.debug(f"Could not load image {img_path}: {e}")

                caption = None
                if not self._suppress_image_path_caption:
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
            if filepath and self._source_dir is not None:
                base_dir = self._source_dir.resolve()
                input_path = self._source_dir / filepath
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
                        content = decode_latex_content(input_path)
                        sub_walker = LatexWalker(
                            content,
                            tolerant_parsing=True,
                            latex_context=LATEX_CONTEXT_DB,
                        )
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
                    if isinstance(arg, LatexGroupNode):
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
                    if isinstance(arg, LatexGroupNode):
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

    def _add_caption(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None,
        formatting: Formatting | None = None,
    ) -> None:
        if node.nodeargd is None or not node.nodeargd.argnlist:
            return
        argument = node.nodeargd.argnlist[-1]
        if not isinstance(argument, LatexGroupNode):
            return
        self._process_nodes(
            argument.nodelist,
            doc,
            parent,
            formatting,
            DocItemLabel.CAPTION,
        )

    def _extract_macro_arg(self, node: LatexMacroNode) -> str:
        if node.nodeargd and node.nodeargd.argnlist:
            arg = node.nodeargd.argnlist[-1]
            if arg:
                if isinstance(arg, LatexGroupNode):
                    return self._nodes_to_text(arg.nodelist)
                return arg.latex_verbatim().strip("{} ")
        return ""

    def _extract_macro_arg_by_index(self, node: LatexMacroNode, index: int) -> str:
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if arg:
                    if isinstance(arg, LatexGroupNode):
                        return self._nodes_to_text(arg.nodelist)
                    return arg.latex_verbatim().strip("{} ")
        return ""

    def _href_link_text(self, node: LatexMacroNode) -> str:
        """``[text](url)`` for ``\\href[options]{url}{text}``, or whichever part is set."""
        args = node.nodeargd.argnlist if node.nodeargd else []
        if not args or len(args) < 2:
            return ""
        url_arg = args[-2]
        # Taken verbatim: characters such as _ and # are not markup in a URL.
        url = url_arg.latex_verbatim().strip("{} ") if url_arg is not None else ""
        text = self._extract_macro_arg(node)
        if url and text:
            return f"[{text}]({url})"
        return url or text

    def _extract_macro_arg_nodes(self, node: LatexMacroNode, index: int) -> list:
        if node.nodeargd and node.nodeargd.argnlist:
            if 0 <= index < len(node.nodeargd.argnlist):
                arg = node.nodeargd.argnlist[index]
                if isinstance(arg, LatexGroupNode):
                    return arg.nodelist
        return []

    def _extract_all_macro_args_inline(self, node: LatexMacroNode) -> str:
        if not node.nodeargd or not node.nodeargd.argnlist:
            return ""

        parts = []
        for arg in node.nodeargd.argnlist:
            if arg is not None:
                if isinstance(arg, LatexGroupNode):
                    text = self._nodes_to_text(arg.nodelist)
                    if text:
                        parts.append(text)
                else:
                    text = arg.latex_verbatim().strip("{} ")
                    if text:
                        parts.append(text)

        return " ".join(parts)

    def _expand_macros(self, latex_str: str) -> str:
        for _ in range(10):
            previous = latex_str
            for macro_name, macro_def in self._custom_macros.items():
                if self._custom_macro_num_args.get(macro_name, 0) > 0:
                    continue
                if "\\" + macro_name not in latex_str:
                    continue
                latex_str = re.sub(
                    rf"\\{re.escape(macro_name)}(?![a-zA-Z])",
                    lambda m: macro_def,
                    latex_str,
                )
            if latex_str == previous:
                break
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
            walker = LatexWalker(
                latex_fragment, tolerant_parsing=True, latex_context=LATEX_CONTEXT_DB
            )
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
