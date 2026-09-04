"""Reconcile predicted table rows with rules drawn in the PDF vector layer.

TableFormer predicts row boundaries from a fixed-resolution rendering of the
table, where a thin drawn rule is sub-pixel and dense wrapped rows are
visually ambiguous. On ruled tables this can cut a wrapped cell one text line
too early, so the tail of one row is matched into the head of the next
(issue #4028). The drawn rules, however, mark the true boundaries exactly.

This module re-bins the word cells of a predicted table into the bands
delimited by those rules. It only re-assigns cells that already belong to the
table, so text is conserved by construction — nothing is re-extracted from
page regions. Whenever the rules do not corroborate the predicted structure
(no rules, band/row mismatch, spanning cells), it declines and leaves the
prediction untouched, keeping borderless and scanned tables unaffected.
"""

from __future__ import annotations

import bisect
import logging
from statistics import median

from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import TextCell

from docling.datamodel.base_models import Table

_log = logging.getLogger(__name__)

# A rule must span at least this fraction of the table width to count as a
# row boundary; shorter strokes are underlines or cell decorations.
_MIN_RULE_SPAN_FRAC = 0.7
# Rules closer together than this are one double-drawn boundary (pt).
_RULE_MERGE_TOL = 2.0
# Rules this close to the table's top/bottom edge duplicate the outer border.
_EDGE_MARGIN = 3.0


def reconcile_table_rows_with_rules(
    table: Table,
    rules: list[BoundingBox],
    word_cells: list[TextCell],
) -> bool:
    """Re-bin ``table``'s words into the row bands drawn by ``rules``.

    ``rules`` are horizontal shape lines in page coordinates (top-left
    origin), as returned by ``PdfPageBackend.get_shape_lines``. ``word_cells``
    are the text cells that were fed to cell matching, unscaled.

    Returns ``True`` when reconciliation was applied and ``False`` when it
    declined; declining never modifies the table.
    """
    table_bbox = table.cluster.bbox
    if table.num_rows < 2 or table.num_cols < 1 or not rules or not word_cells:
        return False

    for cell in table.table_cells:
        if cell.bbox is None:
            return False
        if (cell.end_row_offset_idx - cell.start_row_offset_idx) != 1:
            return False
        if (cell.end_col_offset_idx - cell.start_col_offset_idx) != 1:
            return False

    boundaries = _row_boundaries(table_bbox, rules)
    if len(boundaries) - 1 != table.num_rows:
        _log.debug(
            "Rule reconciliation declined: %d ruled bands vs %d predicted rows",
            len(boundaries) - 1,
            table.num_rows,
        )
        return False

    column_ranges = _column_ranges(table)
    assigned: dict[tuple[int, int], list[TextCell]] = {}
    for word in word_cells:
        if not word.text.strip():
            continue
        word_bbox = word.rect.to_bounding_box()
        y_center = (word_bbox.t + word_bbox.b) / 2
        x_center = (word_bbox.l + word_bbox.r) / 2
        if not (table_bbox.t <= y_center <= table_bbox.b):
            continue
        if not (table_bbox.l <= x_center <= table_bbox.r):
            continue
        row = min(
            max(bisect.bisect_right(boundaries, y_center) - 1, 0),
            table.num_rows - 1,
        )
        col = _nearest_column(column_ranges, x_center)
        assigned.setdefault((row, col), []).append(word)

    for cell in table.table_cells:
        assert cell.bbox is not None  # guaranteed by the validation loop above
        key = (cell.start_row_offset_idx, cell.start_col_offset_idx)
        words = _reading_order(assigned.get(key, []))
        new_text = " ".join(word.text.strip() for word in words)
        if new_text != (cell.text or ""):
            cell.text = new_text
        if words:
            cell.bbox = _union(
                [word.rect.to_bounding_box() for word in words],
                coord_origin=cell.bbox.coord_origin,
            )
    return True


def _row_boundaries(table_bbox: BoundingBox, rules: list[BoundingBox]) -> list[float]:
    """Band boundaries: table edges plus the interior rules that span it.

    The table's own top and bottom edges are admitted as implicit boundaries
    because a row cut by a page break has a drawn rule on one side only.
    """
    width = table_bbox.r - table_bbox.l
    interior: list[float] = []
    for y in sorted(
        (rule.t + rule.b) / 2
        for rule in rules
        if min(rule.r, table_bbox.r) - max(rule.l, table_bbox.l)
        >= _MIN_RULE_SPAN_FRAC * width
        and table_bbox.t + _EDGE_MARGIN
        < (rule.t + rule.b) / 2
        < table_bbox.b - _EDGE_MARGIN
    ):
        if not interior or y - interior[-1] > _RULE_MERGE_TOL:
            interior.append(y)
    return [table_bbox.t, *interior, table_bbox.b]


def _column_ranges(table: Table) -> list[tuple[int, float, float]]:
    ranges: dict[int, tuple[float, float]] = {}
    for cell in table.table_cells:
        col = cell.start_col_offset_idx
        assert cell.bbox is not None
        left, right = ranges.get(col, (cell.bbox.l, cell.bbox.r))
        ranges[col] = (min(left, cell.bbox.l), max(right, cell.bbox.r))
    return [(col, left, right) for col, (left, right) in sorted(ranges.items())]


def _nearest_column(
    column_ranges: list[tuple[int, float, float]], x_center: float
) -> int:
    best_col, best_distance = column_ranges[0][0], float("inf")
    for col, left, right in column_ranges:
        if left <= x_center <= right:
            return col
        distance = min(abs(x_center - left), abs(x_center - right))
        if distance < best_distance:
            best_col, best_distance = col, distance
    return best_col


def _reading_order(words: list[TextCell]) -> list[TextCell]:
    """Sort words top-to-bottom by text line, left-to-right within a line."""
    if not words:
        return []
    boxes = {id(word): word.rect.to_bounding_box() for word in words}
    line_step = median(boxes[id(word)].b - boxes[id(word)].t for word in words) / 2
    ordered = sorted(
        words, key=lambda word: (boxes[id(word)].t + boxes[id(word)].b) / 2
    )
    lines: list[list[TextCell]] = []
    line_center = float("-inf")
    for word in ordered:
        center = (boxes[id(word)].t + boxes[id(word)].b) / 2
        if center - line_center > line_step:
            lines.append([])
        lines[-1].append(word)
        line_center = center
    return [
        word
        for line in lines
        for word in sorted(line, key=lambda word: boxes[id(word)].l)
    ]


def _union(boxes: list[BoundingBox], coord_origin) -> BoundingBox:
    return BoundingBox(
        l=min(box.l for box in boxes),
        t=min(box.t for box in boxes),
        r=max(box.r for box in boxes),
        b=max(box.b for box in boxes),
        coord_origin=coord_origin,
    )
