# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Parse LaTeX front matter as rich document text with source roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    GroupItem,
    GroupLabel,
    NodeItem,
)
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
_AFFILIATION_ROLES = {"affiliation", "altaffiliation", "institute", "address"}


class LatexMetadataHandler:
    """Keep front-matter boundaries, inline math, and source command roles."""

    def __init__(self, backend: LatexDocumentBackend) -> None:
        self._backend = backend
        self._metadata_groups: dict[str, GroupItem] = {}
        self._role_groups: dict[tuple[str, str], GroupItem] = {}

    def process_preamble(self, nodes: list, doc: DoclingDocument) -> None:
        """Read metadata before the document environment."""
        self._metadata_groups.clear()
        self._role_groups.clear()
        self._process_preamble_nodes(nodes, doc, depth=0)

    def _parent_for_role(
        self, doc: DoclingDocument, parent: NodeItem | None, role: str
    ) -> GroupItem:
        parent_ref = parent.self_ref if parent is not None else doc.body.self_ref
        if parent_ref not in self._metadata_groups:
            self._metadata_groups[parent_ref] = doc.add_group(
                parent=parent, label=GroupLabel.UNSPECIFIED, name="metadata"
            )
        metadata_group = self._metadata_groups[parent_ref]
        group_name = (
            "authors"
            if role == "author"
            else "affiliations"
            if role in _AFFILIATION_ROLES
            else None
        )
        if group_name is None:
            return metadata_group
        key = (parent_ref, group_name)
        if key not in self._role_groups:
            self._role_groups[key] = doc.add_group(
                parent=metadata_group,
                label=GroupLabel.UNSPECIFIED,
                name=group_name,
            )
        return self._role_groups[key]

    def _process_preamble_nodes(
        self, nodes: list, doc: DoclingDocument, depth: int
    ) -> None:
        if depth > 10:
            return
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.envname == "document":
                break
            if isinstance(node, LatexMacroNode) and node.macroname in MACROS_METADATA:
                self.process_macro(node, doc, in_preamble=True)
                continue
            for children in iter_child_nodelists(node):
                self._process_preamble_nodes(children, doc, depth + 1)

    def process_macro(
        self,
        node: LatexMacroNode,
        doc: DoclingDocument,
        parent: NodeItem | None = None,
        formatting: Formatting | None = None,
        *,
        in_preamble: bool = False,
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
        role_parent = (
            parent
            if role == "footnote" and not in_preamble
            else self._parent_for_role(doc, parent, role)
        )
        for segment in segments:
            self._backend._process_nodes(
                self._replace_inst_markers(segment),
                doc,
                role_parent,
                formatting,
                label,
            )

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
