# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Merging of tables that continue across a page break.

The layout and table-structure models work one page at a time, so a table that runs over a
page break comes out as one ``TableItem`` per page. Fragments after the first often start
with a row flagged ``column_header`` that is really the tail of a record cut by the break,
or with a repeat of the original header.

:class:`TableContinuationModel` runs after the reading-order model and joins such fragments
into one ``TableItem`` whose ``prov`` has one entry per page. A fragment is treated as the
continuation of the table right before it in reading order when:

- nothing else in the body sits between the two tables;
- it starts on the page right after the one the previous fragment ends on;
- it has no caption of its own, or only a "continued" caption;
- it has the same number of columns and the same orientation;
- the left edges of its columns line up with those of the previous fragment.

Cell bounding boxes cover the cell's text, so column widths and right edges change with the
content on each page. The left edge of left-aligned text does not, which makes it the
alignment signal. Each merged cell keeps its bounding box in the coordinates of the page it
came from.

When merging, a repeated header is dropped, and header flags on rows that do not repeat the
original header are cleared. With ``repair_split_rows``, a first row whose leading column is
empty, following a last row whose leading column is filled, is joined back into that row.

The core (:meth:`TableContinuationModel.merge_continued_tables`) works on a bare
``DoclingDocument``, so it can also be applied to documents that were already converted.
"""

import itertools
import logging
import re
from dataclasses import dataclass, field

from docling_core.types.doc import DoclingDocument, NodeItem, TableCell, TableItem
from docling_core.types.doc.document import RichTableCell

from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableContinuationOptions

_log = logging.getLogger(__name__)

# Captions that mark a fragment as the continuation of an earlier table.
_CONTINUED_CAPTION = re.compile(
    r"\b(continued|cont\.|contd\.?|vervolg|fortsetzung|suite|continuación|continuação)\b",
    re.IGNORECASE,
)


@dataclass
class TableMergeDecision:
    """Why a table fragment was or was not merged into the table before it."""

    first: str  # self_ref of the table merged into
    second: str  # self_ref of the candidate continuation
    merged: bool
    reason: str
    details: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _grid_texts(table: TableItem) -> list[list[str]]:
    grid = [
        ["" for _ in range(table.data.num_cols)] for _ in range(table.data.num_rows)
    ]
    for cell in table.data.table_cells:
        grid[cell.start_row_offset_idx][cell.start_col_offset_idx] = cell.text
    return grid


def _column_left_edges(table: TableItem, page_width: float) -> list[float | None]:
    """Left edge of each column as a fraction of page width, from its single-column cells."""
    edges: list[float | None] = [None] * table.data.num_cols
    for cell in table.data.table_cells:
        if cell.bbox is None or cell.col_span != 1 or not cell.text.strip():
            continue
        col = cell.start_col_offset_idx
        left = cell.bbox.l / page_width
        current = edges[col]
        edges[col] = left if current is None else min(current, left)
    return edges


def _header_rows(table: TableItem) -> int:
    """Number of leading rows whose cells are all column headers."""
    rows = 0
    for r in range(table.data.num_rows):
        cells = [c for c in table.data.table_cells if c.start_row_offset_idx == r]
        if cells and all(c.column_header for c in cells):
            rows += 1
        else:
            break
    return rows


def _join_split_row(
    a: TableItem,
    b_cells: list[TableCell],
    body_a: list[list[str]],
    grid_b: list[list[str]],
    first_b: int,
) -> bool:
    """Append the first continuation row to the last row of ``a`` if it is that record's tail.

    The row is a tail when its leading column is empty while the leading column of the last
    body row of ``a`` is filled. Returns whether the row was joined.
    """
    if not body_a or first_b >= len(grid_b):
        return False
    tail, last = grid_b[first_b], body_a[-1]
    if tail[0].strip() or not last[0].strip() or not any(t.strip() for t in tail):
        return False
    tail_cells = [c for c in b_cells if c.start_row_offset_idx == first_b]
    if any(c.row_span != 1 for c in tail_cells):
        return False
    last_row = a.data.num_rows - 1
    by_col = {
        c.start_col_offset_idx: c
        for c in a.data.table_cells
        if c.start_row_offset_idx == last_row
    }
    for c in tail_cells:
        target = by_col.get(c.start_col_offset_idx)
        if c.text.strip() and target is not None:
            target.text = f"{target.text.rstrip()} {c.text.strip()}".strip()
    return True


def _body_sequence(document: DoclingDocument) -> list[NodeItem]:
    """Body items in reading order, without the captions and footnotes of tables."""
    items: list[NodeItem] = []
    table_children: set[str] = set()
    for item, _level in document.iterate_items():
        if isinstance(item, TableItem):
            table_children.update(
                ref.cref for ref in [*item.children, *item.captions, *item.footnotes]
            )
        if item.self_ref in table_children:
            continue
        items.append(item)
    return items


class TableContinuationModel:
    """Merges tables that continue across a page break in an assembled ``DoclingDocument``.

    Intended to run right after the reading-order model. It accepts a ``ConversionResult``
    and returns the (in-place modified) ``DoclingDocument``. The core logic is exposed via
    :meth:`merge_continued_tables`, which works on a bare ``DoclingDocument``.
    """

    def __init__(self, options: TableContinuationOptions):
        self.options = options

    def __call__(self, conv_res: ConversionResult) -> DoclingDocument:
        document = conv_res.document
        if not self.options.enabled:
            return document
        self.merge_continued_tables(document)
        return document

    def merge_continued_tables(
        self, document: DoclingDocument
    ) -> list[TableMergeDecision]:
        """Merge every table that continues the table right before it. Edits ``document`` in place."""
        decisions: list[TableMergeDecision] = []
        to_delete: list[NodeItem] = []
        sequence = _body_sequence(document)
        current: TableItem | None = None
        for prev, item in itertools.pairwise(sequence):
            if not isinstance(item, TableItem):
                current = None
                continue
            anchor = current if current is not None else prev
            if not isinstance(anchor, TableItem):
                continue
            ok, reason = self._check(document, anchor, item)
            details: list[str] = []
            if ok:
                details = self._merge(document, anchor, item)
                to_delete.append(item)
                current = anchor
            else:
                current = None
            decision = TableMergeDecision(
                anchor.self_ref, item.self_ref, ok, reason, details
            )
            _log.debug("Table continuation: %s", decision)
            decisions.append(decision)
        if to_delete:
            document.delete_items(node_items=to_delete)
        return decisions

    def _check(
        self, document: DoclingDocument, a: TableItem, b: TableItem
    ) -> tuple[bool, str]:
        if not a.prov or not b.prov:
            return False, "missing provenance"
        if b.prov[0].page_no != a.prov[-1].page_no + 1:
            return (
                False,
                f"not on consecutive pages ({a.prov[-1].page_no} -> {b.prov[0].page_no})",
            )
        if any(isinstance(c, RichTableCell) for c in b.data.table_cells):
            return False, "continuation has rich cells"
        attached = {ref.cref for ref in [*b.captions, *b.footnotes]}
        if any(ref.cref not in attached for ref in b.children):
            return False, "continuation has children other than captions and footnotes"
        if b.captions:
            caption = " ".join(ref.resolve(document).text for ref in b.captions)
            if not _CONTINUED_CAPTION.search(caption):
                return False, f"continuation has its own caption: {caption[:60]!r}"
        if a.data.num_cols != b.data.num_cols:
            return (
                False,
                f"column count differs ({a.data.num_cols} vs {b.data.num_cols})",
            )
        if a.data.orientation != b.data.orientation:
            return False, "orientation differs"
        page_a = document.pages.get(a.prov[-1].page_no)
        page_b = document.pages.get(b.prov[0].page_no)
        if page_a is None or page_b is None:
            return False, "missing page size"
        edges_a = _column_left_edges(a, page_a.size.width)
        edges_b = _column_left_edges(b, page_b.size.width)
        shifts = [
            abs(x - y)
            for x, y in zip(edges_a, edges_b)
            if x is not None and y is not None
        ]
        if len(shifts) < max(2, a.data.num_cols // 2):
            return False, "too few columns with geometry to compare"
        worst = max(shifts)
        if worst > self.options.max_column_shift:
            return (
                False,
                f"columns misaligned (largest left-edge shift {worst:.3f} of page width)",
            )
        return (
            True,
            f"columns aligned (largest left-edge shift {worst:.3f} of page width)",
        )

    def _merge(
        self, document: DoclingDocument, a: TableItem, b: TableItem
    ) -> list[str]:
        details: list[str] = []
        grid_a, grid_b = _grid_texts(a), _grid_texts(b)
        head_a, head_b = _header_rows(a), _header_rows(b)
        skip_b = 0

        if (
            head_a
            and head_b == head_a
            and [list(map(_norm, r)) for r in grid_b[:head_b]]
            == [list(map(_norm, r)) for r in grid_a[:head_a]]
        ):
            skip_b = head_b
            details.append(f"dropped {head_b} repeated header row(s)")
        elif head_b:
            details.append(
                f"cleared header flag on {head_b} row(s) that do not repeat the header"
            )

        b_cells = [c for c in b.data.table_cells if c.start_row_offset_idx >= skip_b]
        for c in b_cells:
            c.column_header = False

        first_b = skip_b
        joined = self.options.repair_split_rows and _join_split_row(
            a, b_cells, grid_a[head_a:], grid_b, first_b
        )
        if joined:
            b_cells = [c for c in b_cells if c.start_row_offset_idx != first_b]
            details.append("joined the split record across the page break")

        offset = a.data.num_rows - (first_b + (1 if joined else 0))
        for c in b_cells:
            c.start_row_offset_idx += offset
            c.end_row_offset_idx += offset
        a.data.table_cells.extend(b_cells)
        a.data.num_rows = max(
            (c.end_row_offset_idx for c in a.data.table_cells), default=0
        )
        a.prov.extend(b.prov)

        # The fragment is deleted with its children, so move its footnotes to the merged
        # table first. A "continued" caption has nothing to add and goes with the fragment.
        for ref in b.footnotes:
            footnote = ref.resolve(document)
            b.children = [c for c in b.children if c.cref != ref.cref]
            footnote.parent = a.get_ref()
            a.children.append(ref)
            a.footnotes.append(ref)
        b.footnotes = []
        return details
