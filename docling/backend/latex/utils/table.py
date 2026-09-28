# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from docling_core.types.doc.document import TableCell, TableData

from docling.backend.latex.constants import (
    MACROS_ESCAPED,
    TABLE_MACROS_IGNORE,
    TABLE_MACROS_RULE,
)

if TYPE_CHECKING:
    from typing import Any

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexMacroNode,
        LatexWalker,
        LatexWalkerParseError,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


class TableHelperMixin:
    if TYPE_CHECKING:

        def _nodes_to_text(self, nodes: Any) -> str: ...

    @staticmethod
    def _consumed_end(
        macro_pos: int, args_end: int, source_latex: str, source_start: int
    ) -> int:
        """End of a multirow/multicolumn invocation in document coordinates.

        Extends over the insignificant whitespace that may sit between the
        macro arguments and the following ``&`` or ``\\`` so it cannot
        become an empty phantom cell.
        """
        end = macro_pos + args_end
        rel = end - source_start
        while rel < len(source_latex) and source_latex[rel] in " \t\n":
            end += 1
            rel += 1
        return end

    def _process_table_macro_node(
        self,
        n: LatexMacroNode,
        source_latex: str,
        source_start: int,
        consumed_until: List[int],
        suppress_cell_break: List[bool],
        current_cell_nodes: List,
        finish_cell_fn: Callable[..., None],
        finish_row_fn: Callable[[], None],
        parse_brace_args_fn: Callable[..., tuple[List[str], int]],
    ):
        if n.macroname == "\\":  # Row break
            finish_row_fn()

        elif n.macroname == "multicolumn":
            if hasattr(n, "pos") and n.pos is not None:
                # n.pos is document-global while source_latex starts at
                # source_start; slice with the document-local offset.
                remaining = source_latex[n.pos - source_start :]
                args, args_end = parse_brace_args_fn(remaining, 3)
                if len(args) >= 3:
                    consumed_until[0] = max(
                        consumed_until[0],
                        self._consumed_end(n.pos, args_end, source_latex, source_start),
                    )
                    try:
                        num_cols = int(args[0])
                    except (ValueError, TypeError):
                        num_cols = 1
                    content_text = args[2]
                    if content_text:
                        try:
                            w = LatexWalker(content_text, tolerant_parsing=True)
                            parsed, _, _ = w.get_latex_nodes()
                            current_cell_nodes.extend(parsed)
                        except LatexWalkerParseError:
                            current_cell_nodes.append(
                                LatexCharsNode(chars=content_text)
                            )
                    finish_cell_fn(col_span=num_cols)
                    # The invocation already closed its cell; a following &
                    # must not emit an empty one for it.
                    suppress_cell_break[0] = True
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname == "multirow":
            if hasattr(n, "pos") and n.pos is not None:
                # Same document-global vs node-local coordinate handling as
                # for multicolumn above.
                remaining = source_latex[n.pos - source_start :]
                args, args_end = parse_brace_args_fn(remaining, 3)
                if len(args) >= 3:
                    consumed_until[0] = max(
                        consumed_until[0],
                        self._consumed_end(n.pos, args_end, source_latex, source_start),
                    )
                    try:
                        num_rows = int(args[0])
                    except (ValueError, TypeError):
                        num_rows = 1
                    content_text = args[2]
                    if content_text:
                        try:
                            w = LatexWalker(content_text, tolerant_parsing=True)
                            parsed, _, _ = w.get_latex_nodes()
                            current_cell_nodes.extend(parsed)
                        except LatexWalkerParseError:
                            current_cell_nodes.append(
                                LatexCharsNode(chars=content_text)
                            )
                    finish_cell_fn(row_span=num_rows)
                    suppress_cell_break[0] = True
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname in TABLE_MACROS_RULE:
            pass
        elif n.macroname in TABLE_MACROS_IGNORE:
            pass
        elif n.macroname == "&":  # Cell break
            if suppress_cell_break[0]:
                suppress_cell_break[0] = False
            else:
                finish_cell_fn()
        elif n.macroname in MACROS_ESCAPED:
            current_cell_nodes.append(n)
        else:
            current_cell_nodes.append(n)

    def _add_chars_node(
        self,
        n: LatexCharsNode,
        current_cell_nodes: List,
        suppress_cell_break: List[bool],
        finish_cell_fn: Callable[..., None],
    ) -> None:
        """Add a chars node, splitting on ``&`` cell breaks it may hold."""
        if "&" in n.chars:
            parts = n.chars.split("&")
            for i, part in enumerate(parts):
                if part:
                    current_cell_nodes.append(LatexCharsNode(chars=part))
                if i < len(parts) - 1:
                    if suppress_cell_break[0]:
                        suppress_cell_break[0] = False
                    else:
                        finish_cell_fn()
        else:
            current_cell_nodes.append(n)

    def _parse_table(self, node: LatexEnvironmentNode) -> TableData | None:
        rows = []
        current_row = []
        current_cell_nodes: list = []

        source_latex = node.latex_verbatim()
        # source_latex starts at the node itself, while pylatexenc node
        # positions are document-global; remember the base for offset math.
        source_start = getattr(node, "pos", 0) or 0
        # Document-global position up to which nodes were consumed as
        # multirow/multicolumn arguments; walkers re-yield those braces as
        # standalone group nodes, which must not become phantom cells.
        consumed_until = [0]
        # One pending cell break is swallowed right after a multirow/
        # multicolumn invocation finished its own cell.
        suppress_cell_break = [False]

        def parse_brace_args(
            text: str, max_args: int | None = None
        ) -> tuple[list, int]:
            # Stop after max_args groups: the source continues past the
            # macro's own arguments, and greedily consuming further braces
            # (e.g. those of \\end{tabular}) would mark the rest of the
            # table as consumed.
            args = []
            i = 0
            last_end = 0
            while i < len(text):
                if max_args is not None and len(args) >= max_args:
                    break
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
                    last_end = i
                else:
                    i += 1
            return args, last_end

        def finish_cell(col_span: int = 1, row_span: int = 1):
            text = self._nodes_to_text(current_cell_nodes).strip()
            cell = TableCell(
                text=text,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
            )
            cell._col_span = col_span  # type: ignore[attr-defined]
            cell._row_span = row_span  # type: ignore[attr-defined]
            current_row.append(cell)
            current_cell_nodes.clear()

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
            # A buffer holding only whitespace is inter-token spacing, not a
            # cell: finishing it would append an empty trailing cell (and, at
            # the end of the table, an extra empty row).
            if current_cell_nodes:
                if self._nodes_to_text(current_cell_nodes).strip():
                    finish_cell()
                else:
                    current_cell_nodes.clear()
            if current_row:
                rows.append(current_row[:])
            current_row.clear()
            suppress_cell_break[0] = False

        if node.nodelist is None:
            return None

        for n in node.nodelist:
            n_pos = getattr(n, "pos", None)
            if n_pos is not None and n_pos < consumed_until[0]:
                # Braces already consumed as multirow/multicolumn arguments;
                # pylatexenc re-yields them as group nodes.
                continue
            if isinstance(n, LatexMacroNode):
                self._process_table_macro_node(
                    n,
                    source_latex,
                    source_start,
                    consumed_until,
                    suppress_cell_break,
                    current_cell_nodes,
                    finish_cell,
                    finish_row,
                    parse_brace_args,
                )
            elif isinstance(n, LatexCharsNode):
                self._add_chars_node(
                    n, current_cell_nodes, suppress_cell_break, finish_cell
                )
            else:
                if hasattr(n, "specials_chars") and n.specials_chars == "&":
                    if suppress_cell_break[0]:
                        suppress_cell_break[0] = False
                    else:
                        finish_cell()
                else:
                    current_cell_nodes.append(n)

        finish_row()

        if not rows:
            return None

        num_rows = len(rows)
        num_cols = max(len(row) for row in rows) if rows else 0

        flat_cells = []
        for i, row in enumerate(rows):
            for j in range(num_cols):
                if j < len(row):
                    cell = row[j]
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

                cell.start_row_offset_idx = i
                cell.start_col_offset_idx = j

                col_span = getattr(cell, "_col_span", 1)
                row_span = getattr(cell, "_row_span", 1)
                cell.end_row_offset_idx = i + row_span
                cell.end_col_offset_idx = j + col_span

                flat_cells.append(cell)

        return TableData(num_rows=num_rows, num_cols=num_cols, table_cells=flat_cells)
