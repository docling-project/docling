from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from typing import Any

from docling_core.types.doc.document import TableCell, TableData
from pylatexenc.latexwalker import LatexCharsNode, LatexEnvironmentNode, LatexMacroNode

from docling.backend.latex.constants import (
    MACROS_ESCAPED,
    TABLE_MACROS_IGNORE,
    TABLE_MACROS_RULE,
)


class TableHelperMixin:
    if TYPE_CHECKING:

        def _nodes_to_text(self, nodes: "Any") -> str: ...

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
            if hasattr(n, "pos") and n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_cols = int(args[0])
                    except (ValueError, TypeError):
                        num_cols = 1
                    content_text = args[2]
                    if content_text:
                        current_cell_nodes.append(LatexCharsNode(chars=content_text))
                    finish_cell_fn(col_span=num_cols)
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname == "multirow":
            if hasattr(n, "pos") and n.pos is not None:
                remaining = source_latex[n.pos :]
                args = parse_brace_args_fn(remaining)
                if len(args) >= 3:
                    try:
                        num_rows = int(args[0])
                    except (ValueError, TypeError):
                        num_rows = 1
                    content_text = args[2]
                    if content_text:
                        current_cell_nodes.append(LatexCharsNode(chars=content_text))
                    finish_cell_fn(row_span=num_rows)
                else:
                    current_cell_nodes.append(n)
            else:
                current_cell_nodes.append(n)

        elif n.macroname in TABLE_MACROS_RULE:
            pass
        elif n.macroname in TABLE_MACROS_IGNORE:
            pass
        elif n.macroname == "&":  # Cell break
            finish_cell_fn()
        elif n.macroname in MACROS_ESCAPED:
            current_cell_nodes.append(n)
        else:
            current_cell_nodes.append(n)

    def _parse_table(self, node: LatexEnvironmentNode) -> TableData | None:
        rows = []
        current_row = []
        current_cell_nodes: list = []

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
                if hasattr(n, "specials_chars") and n.specials_chars == "&":
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
