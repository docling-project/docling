# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keys read from detected table cells, and grid context."""

from __future__ import annotations

from collections import defaultdict

from docling_core.types.doc import BoundingBox, TableCell

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.keying.candidates import siblings
from docling.models.stages.form_field.keying.geometry import (
    band_index,
    gap,
    lettered,
    span_overlap,
)
from docling.models.stages.form_field.keying.rules import printed_cell, rules_of
from docling.models.stages.form_field.keying.types import (
    Candidate,
    Label,
    Scope,
    TableSlot,
    Value,
)


def table_fields(
    found: list[Cluster],
    table_cells: dict[int, list[TableCell]],
    rule_boxes: list[BoundingBox],
    page_height: float,
    values: list[Value],
    labels: list[Label],
) -> tuple[list[Candidate], dict[int, TableSlot]]:
    """The grid cell of each value inside a detected table, and its key.

    A value is keyed only by lettered text of its own cell. A row caption or a
    column header elsewhere in the table never keys a value: the table's
    headers already carry that association. Cell text is used whole; codes
    and units without letters never key a value, and neither does cell text
    that repeats the value itself (a filled field rendered into the page).
    Appends the cell labels it uses to ``labels``.

    A value inside one detected cell sits in that cell. Otherwise it sits in
    the row and column whose bands it overlaps most; a band is the extent of
    the single-span cells of that row or column, because detected cell boxes
    cover their text rather than the printed cell. A value between bands
    (text at the top of its printed cell, the box below) takes the nearest
    band only when every row, or every column, of the grid has one: a missing
    band would hand its values to a neighbour, so such a value gets no cell,
    and no key.

    Printed rules correct the bands where the detected grid differs from the
    printed table. When rules close a printed cell around the value and that
    cell holds detected text, the value belongs with that text: if the bands
    point elsewhere, the value moves to the cell of the text in its printed
    cell (a caption before a code), and its key follows. When the printed cell
    holds no text and misses the band of the value's column, the detected grid
    lost that column: the value gets no cell. Without a closed printed cell the
    bands decide.
    """
    fields: list[Candidate] = []
    slots: dict[int, TableSlot] = {}
    rules = rules_of(rule_boxes)
    frames = {region.id: region.bbox for region in found}
    for table_id, detected in sorted(table_cells.items()):
        for i, v in enumerate(values):
            if v.scope.table == table_id and v.scope.cell is not None:
                cell = detected[v.scope.cell]
                slots[i] = TableSlot(
                    table_id,
                    (cell.start_row_offset_idx, cell.end_row_offset_idx),
                    (cell.start_col_offset_idx, cell.end_col_offset_idx),
                )
        members = [
            i
            for i, v in enumerate(values)
            if v.scope.table == table_id and not v.scope.eligible
        ]
        cells = [
            (cell, cell.bbox.to_top_left_origin(page_height))
            for cell in detected
            if cell.bbox is not None
        ]
        rows: dict[int, list[BoundingBox]] = defaultdict(list)
        columns: dict[int, list[BoundingBox]] = defaultdict(list)
        for cell, box in cells:
            if cell.end_row_offset_idx - cell.start_row_offset_idx == 1:
                rows[cell.start_row_offset_idx].append(box)
            if cell.end_col_offset_idx - cell.start_col_offset_idx == 1:
                columns[cell.start_col_offset_idx].append(box)
        if not members or not rows or not columns:
            continue
        row_bands = {
            r: (min(b.t for b in bs), max(b.b for b in bs)) for r, bs in rows.items()
        }
        column_bands = {
            c: (min(b.l for b in bs), max(b.r for b in bs)) for c, bs in columns.items()
        }
        grid_rows = max(cell.end_row_offset_idx for cell in detected)
        grid_columns = max(cell.end_col_offset_idx for cell in detected)

        def at(row: int, column: int) -> list[int]:
            return [
                k
                for k, (cell, _) in enumerate(cells)
                if cell.start_row_offset_idx <= row < cell.end_row_offset_idx
                and cell.start_col_offset_idx <= column < cell.end_col_offset_idx
            ]

        indices: dict[int, int] = {}

        def label_of(k: int) -> int:
            if k not in indices:
                cell, box = cells[k]
                indices[k] = len(labels)
                labels.append(
                    Label(
                        " ".join(cell.text.split()),
                        box,
                        frozenset(),
                        Scope(table_id),
                        None,
                    )
                )
            return indices[k]

        for i in members:
            box = values[i].bbox
            row = band_index(box.t, box.b, row_bands)
            column = band_index(box.l, box.r, column_bands)
            printed = (
                None
                if not rules or table_id not in frames
                else printed_cell(box, rules, frames[table_id])
            )
            by_print = lost_column = False
            if printed is not None:
                held = [
                    k
                    for k, (cell, cell_box) in enumerate(cells)
                    if cell.text.strip()
                    and cell_box.intersection_over_self(printed) >= 0.5
                ]
                if held and not set(held) & set(at(row, column)):
                    k = min(
                        held,
                        key=lambda k: (
                            not lettered(cells[k][0].text),
                            gap(cells[k][1], box),
                            k,
                        ),
                    )
                    row = cells[k][0].start_row_offset_idx
                    column = cells[k][0].start_col_offset_idx
                    by_print = True
                elif not held:
                    lost_column = not any(
                        span_overlap(printed.l, printed.r, b.l, b.r) > 0
                        for b in columns[column]
                    )
            home = at(row, column)
            own = next(
                (
                    k
                    for k in home
                    if lettered(cells[k][0].text)
                    and "".join(cells[k][0].text.split())
                    != "".join((values[i].native.widget_text or "").split())
                ),
                None,
            )
            # A cell found through its printed text needs no band check; the
            # text's column may be one no single-span cell lies in.
            placed = by_print or (
                (
                    span_overlap(box.t, box.b, *row_bands[row]) > 0
                    or len(row_bands) == grid_rows
                )
                and (
                    span_overlap(box.l, box.r, *column_bands[column]) > 0
                    or len(column_bands) == grid_columns
                )
            )
            if placed and not lost_column:
                spans = cells[own if own is not None else home[0]][0] if home else None
                slots[i] = TableSlot(
                    table_id,
                    (row, row + 1)
                    if spans is None
                    else (spans.start_row_offset_idx, spans.end_row_offset_idx),
                    (column, column + 1)
                    if spans is None
                    else (spans.start_col_offset_idx, spans.end_col_offset_idx),
                    None if own is None else label_of(own),
                )
                if own is not None:
                    fields.append(Candidate((i,), label_of(own), "table_cell", {}))
    return fields, slots


def add_context(
    chosen: list[Candidate],
    proposed: list[Candidate],
    values: list[Value],
    null_cost: float,
) -> None:
    """Keep the aligned caption across the other axis as context in value grids.

    A value with like-sized siblings in both its row and its column sits in a
    grid: its key is one axis' caption, and the cheapest aligned caption along
    the other axis (a column header over a row-keyed amount) is kept as
    context. A header heads a line of values, so a sibling along that line
    must see it too; the caption of the previous option in a list does not
    qualify, nor does text too costly to key a value on its own (page
    headers). Context is informative only; it never changes the key.
    """
    grid = {
        i
        for i, v in enumerate(values)
        if any(siblings(v, u, "left") for u in values if u is not v)
        and any(siblings(v, u, "up") for u in values if u is not v)
    }
    across = {"left": ("up", "down"), "right": ("up", "down")}
    for c in chosen:
        if c.side is None or c.members[0] not in grid:
            continue
        sides = across.get(c.side, ("left", "right"))
        value = values[c.members[0]]
        options = [
            o
            for o in proposed
            if o.members == c.members
            and o.side in sides
            and not o.features.get("misaligned")
            and o.cost < null_cost
            and o.label != c.label
            and any(
                other.label == o.label
                and other.side == o.side
                and other.members != c.members
                and siblings(values[other.members[0]], value, o.side)
                for other in proposed
            )
        ]
        if options:
            c.context = min(options, key=lambda o: (o.cost, o.label)).label
