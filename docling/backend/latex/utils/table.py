# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from docling_core.types.doc.document import (
    DoclingDocument,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
)

from docling.backend.latex.constants import (
    MACROS_ESCAPED,
    TABLE_MACROS_IGNORE,
    TABLE_MACROS_RULE,
)
from docling.backend.latex.utils.latex_context import LATEX_CONTEXT_DB
from docling.backend.utils.table_spans import (
    MAX_COLSPAN,
    MAX_ROWSPAN,
    clamp_span,
    table_width,
)

try:  # pragma: no cover - import-time guard
    from pylatexenc.latexwalker import (
        LatexCharsNode,
        LatexEnvironmentNode,
        LatexMacroNode,
        LatexSpecialsNode,
        LatexWalker,
        LatexWalkerParseError,
    )
except ImportError:
    pass  # guarded by LatexDocumentBackend.__init__


@dataclass
class ParsedTable:
    data: TableData
    rich_cells: list[tuple[int, list[Any]]]


class TableHelperMixin:
    if TYPE_CHECKING:

        def _nodes_to_text(self, nodes: Any) -> str: ...
        def _contains_math(self, nodes: list) -> bool: ...
        def _process_nodes(
            self, nodes: Any, doc: DoclingDocument, parent: Any
        ) -> None: ...

    def _process_table_macro_node(
        self,
        n: LatexMacroNode,
        source_latex: str,
        current_cell_nodes: list[Any],
        finish_cell_fn: Callable[..., None],
        finish_row_fn: Callable[[], None],
        parse_brace_args_fn: Callable[[str], list[str]],
    ):
        if n.macroname == "\\":  # Row break
            finish_row_fn()

        elif n.macroname == "multicolumn":
            if n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_cols = clamp_span(int(args[0]), MAX_COLSPAN)
                    except (ValueError, TypeError):
                        num_cols = 1
                    content_text = args[2]
                    if content_text:
                        try:
                            w = LatexWalker(
                                content_text,
                                tolerant_parsing=True,
                                latex_context=LATEX_CONTEXT_DB,
                            )
                            parsed, _, _ = w.get_latex_nodes()
                            current_cell_nodes.extend(parsed)
                        except LatexWalkerParseError:
                            current_cell_nodes.append(
                                LatexCharsNode(chars=content_text)
                            )
                    finish_cell_fn(col_span=num_cols)
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname == "multirow":
            if n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_rows = clamp_span(int(args[0]), MAX_ROWSPAN)
                    except (ValueError, TypeError):
                        num_rows = 1
                    content_text = args[2]
                    if content_text:
                        try:
                            w = LatexWalker(
                                content_text,
                                tolerant_parsing=True,
                                latex_context=LATEX_CONTEXT_DB,
                            )
                            parsed, _, _ = w.get_latex_nodes()
                            current_cell_nodes.extend(parsed)
                        except LatexWalkerParseError:
                            current_cell_nodes.append(
                                LatexCharsNode(chars=content_text)
                            )
                    finish_cell_fn(row_span=num_rows)
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname in TABLE_MACROS_RULE:
            pass
        elif n.macroname == "caption":
            pass
        elif n.macroname in TABLE_MACROS_IGNORE:
            pass
        elif n.macroname == "&":  # Cell break
            finish_cell_fn()
        elif n.macroname in MACROS_ESCAPED:
            current_cell_nodes.append(n)
        else:
            current_cell_nodes.append(n)

    def _parse_table(self, node: LatexEnvironmentNode) -> ParsedTable | None:
        # None marks a column that a \multicolumn cell before it covers.
        rows: list[list[tuple[TableCell, int, int, list[Any] | None] | None]] = []
        current_row: list[tuple[TableCell, int, int, list[Any] | None] | None] = []
        current_cell_nodes: list[Any] = []

        source_latex = node.latex_verbatim()

        def parse_brace_args(text: str) -> list:
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
            rich_nodes = (
                current_cell_nodes.copy()
                if self._contains_math(current_cell_nodes)
                else None
            )
            text = (
                ""
                if rich_nodes is not None
                else self._nodes_to_text(current_cell_nodes).strip()
            )
            cell = TableCell(
                text=text,
                start_row_offset_idx=0,
                end_row_offset_idx=0,
                start_col_offset_idx=0,
                end_col_offset_idx=0,
            )
            current_row.append((cell, col_span, row_span, rich_nodes))
            current_cell_nodes.clear()

            for _ in range(col_span - 1):
                current_row.append(None)

        def finish_row():
            if current_cell_nodes:
                finish_cell()
            if current_row:
                rows.append(current_row[:])
            current_row.clear()

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
                            current_cell_nodes.append(LatexCharsNode(chars=part))
                        if i < len(parts) - 1:
                            finish_cell()
                else:
                    current_cell_nodes.append(n)
            else:
                if isinstance(n, LatexSpecialsNode) and n.specials_chars == "&":
                    finish_cell()
                else:
                    current_cell_nodes.append(n)

        finish_row()

        if not rows:
            return None

        num_rows = len(rows)
        # Columns to the right of the last cell that starts in them hold no
        # cell of their own, so a span wider than the table does not widen it.
        num_cols = min(
            max(len(row) for row in rows),
            table_width(
                [
                    [(entry[1], entry[2]) for entry in row if entry is not None]
                    for row in rows
                ]
            ),
        )

        flat_cells = []
        rich_cells: list[tuple[int, list[Any]]] = []
        for i, row in enumerate(rows):
            for j in range(num_cols):
                if j < len(row):
                    entry = row[j]
                    if entry is None:
                        continue
                    cell, col_span, row_span, rich_nodes = entry
                else:
                    cell = TableCell(
                        text="",
                        start_row_offset_idx=0,
                        end_row_offset_idx=0,
                        start_col_offset_idx=0,
                        end_col_offset_idx=0,
                    )
                    col_span, row_span = 1, 1
                    rich_nodes = None

                cell.start_row_offset_idx = i
                cell.start_col_offset_idx = j

                cell.end_row_offset_idx = min(i + row_span, num_rows)
                cell.end_col_offset_idx = min(j + col_span, num_cols)

                if rich_nodes is not None:
                    rich_cells.append((len(flat_cells), rich_nodes))
                flat_cells.append(cell)

        return ParsedTable(
            data=TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=flat_cells
            ),
            rich_cells=rich_cells,
        )

    def _add_rich_table_cells(
        self, parsed: ParsedTable, doc: DoclingDocument, table: TableItem
    ) -> None:
        for index, nodes in parsed.rich_cells:
            cell_group = doc.add_group(parent=table)
            self._process_nodes(nodes, doc, parent=cell_group)
            cell = table.data.table_cells[index]
            table.data.table_cells[index] = RichTableCell(
                **cell.model_dump(exclude={"text"}),
                text="",
                ref=cell_group.get_ref(),
            )
