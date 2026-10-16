# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse LaTeX front matter as rich document text with source roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem
from docling_core.types.doc.common.meta import BaseMeta
from docling_core.types.doc.document import Formatting

from docling.backend.latex.constants import MACROS_METADATA
from docling.backend.latex.handlers.math import InlineFormula
from docling.backend.latex.utils.text import iter_child_nodelists

if TYPE_CHECKING:
    from pylatexenc.latexwalker import LatexMacroNode

    from docling.backend.latex.backend import LatexDocumentBackend

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexGroupNode,
        LatexMacroNode,
        LatexSpecialsNode,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


_ROLE_ALIASES = {"affil": "affiliation"}
_FOOTNOTE_ROLES = {"thanks", "footnote", "altaffiliation"}
_SPLIT_ROLES = {"author", "affiliation", "institute", "address"}


class LatexMetadataHandler:
    """Keep front-matter boundaries, inline math, and source command roles."""

    def __init__(self, backend: LatexDocumentBackend) -> None:
        self._backend = backend

    def process_preamble(self, nodes: list, doc: DoclingDocument) -> None:
        """Read metadata before the document environment."""
        self._process_preamble_nodes(nodes, doc, depth=0)

    def _process_preamble_nodes(
        self, nodes: list, doc: DoclingDocument, depth: int
    ) -> None:
        if depth > 10:
            return
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                break
            if isinstance(node, LatexMacroNode) and node.macroname in MACROS_METADATA:
                self.process_macro(node, doc)
                continue
            for children in iter_child_nodelists(node):
                self._process_preamble_nodes(children, doc, depth + 1)

    def process_macro(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
    ) -> None:
        """Add one or more rich text items for a metadata command."""
        if node.nodeargd is None or not node.nodeargd.argnlist:
            return
        argument = node.nodeargd.argnlist[-1]
        if not isinstance(argument, LatexGroupNode):
            return

        role = _ROLE_ALIASES.get(node.macroname, node.macroname)
        label = (
            DocItemLabel.TITLE
            if role == "title"
            else DocItemLabel.FOOTNOTE
            if role in _FOOTNOTE_ROLES
            else DocItemLabel.TEXT
        )
        nodes = self._unwrap_parboxes(argument.nodelist or [])
        if role == "altaffiliation" and len(node.nodeargd.argnlist) > 1:
            optional = node.nodeargd.argnlist[0]
            if isinstance(optional, LatexGroupNode) and optional.nodelist:
                nodes = [*optional.nodelist, LatexCharsNode(chars=" "), *nodes]
        segments = self._split_on_and(nodes) if role in _SPLIT_ROLES else [nodes]
        parent_ref = parent.self_ref if parent is not None else doc.body.self_ref

        for segment in segments:
            before = len(doc.texts)
            self._backend._process_nodes(
                self._replace_inst_markers(segment), doc, parent, formatting, label
            )
            for item in doc.texts[before:]:
                if item.label != label or item.parent.cref != parent_ref:
                    continue
                if (
                    item.meta is not None
                    and "latex__classification" in item.meta.get_custom_part()
                ):
                    continue
                meta = item.meta or BaseMeta()
                meta.set_custom_field("latex", "classification", role)
                item.meta = meta

    def _replace_inst_markers(self, nodes: list) -> list:
        """Represent affiliation markers as inline superscript formulas."""
        result: list = []
        for node in nodes:
            if (
                isinstance(node, LatexMacroNode)
                and node.macroname == "inst"
                and node.nodeargd is not None
                and node.nodeargd.argnlist
            ):
                argument = node.nodeargd.argnlist[0]
                if isinstance(argument, LatexGroupNode):
                    marker = argument.latex_verbatim()[1:-1]
                    if marker:
                        formula = self._backend._clean_math(
                            f"$^{{{marker}}}$", "inline"
                        )
                        result.append(InlineFormula(formula))
                    continue
            result.append(node)
        return result

    @staticmethod
    def _split_on_and(nodes: list) -> list[list]:
        segments: list[list] = [[]]
        for node in nodes:
            if isinstance(node, LatexMacroNode) and node.macroname == "and":
                segments.append([])
            else:
                segments[-1].append(node)
        return [segment for segment in segments if segment]

    @staticmethod
    def _unwrap_parboxes(nodes: list) -> list:
        unwrapped: list = []
        for node in nodes:
            if isinstance(node, LatexSpecialsNode) and node.specials_chars == "~":
                unwrapped.append(LatexCharsNode(chars=" "))
                continue
            if isinstance(node, LatexMacroNode) and node.macroname == "parbox":
                if node.nodeargd is not None and node.nodeargd.argnlist:
                    argument = node.nodeargd.argnlist[-1]
                    if isinstance(argument, LatexGroupNode):
                        unwrapped.extend(argument.nodelist or [])
                        continue
            unwrapped.append(node)
        return unwrapped
