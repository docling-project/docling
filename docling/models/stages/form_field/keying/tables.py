# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Keys read from detected table cells, and grid context."""

from __future__ import annotations

from collections import defaultdict

from docling_core.types.doc import BoundingBox

from docling.models.stages.form_field.keying.candidates import siblings
from docling.models.stages.form_field.keying.geometry import band_index, lettered
from docling.models.stages.form_field.keying.types import (
    Candidate,
    Label,
    Scope,
    Snapshot,
    Value,
)


def table_fields(
    snapshot: Snapshot, values: list[Value], labels: list[Label]
) -> list[Candidate]:
    """Keys for values inside detected tables, read from the table's own cells.

    A value sits in the row and column whose bands it overlaps most; a band
    is the extent of the single-span cells of that row or column, because
    detected cell boxes cover their text rather than the printed cell. The
    key is the lettered text of the value's own cell; otherwise the first
    lettered cell to its left, the row caption. The first column header above
    is kept as context, and is the key only when there is no row caption. Cell
    text is used whole; codes and units without letters never key a value.
    Appends the cell labels it uses to ``labels``.
    """
    fields: list[Candidate] = []
    for table_id, structure in sorted(snapshot.tables.table_map.items()):
        members = [
            i
            for i, v in enumerate(values)
            if v.scope.table == table_id and not v.scope.eligible
        ]
        cells = [
            (cell, cell.bbox.to_top_left_origin(snapshot.size.height))
            for cell in structure.table_cells
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
                        "table_cell",
                    )
                )
            return indices[k]

        for i in members:
            box = values[i].bbox
            row = band_index(box.t, box.b, row_bands)
            column = band_index(box.l, box.r, column_bands)
            home = at(row, column)
            own = next((k for k in home if lettered(cells[k][0].text)), None)
            caption = next(
                (
                    k
                    for c in range(column - 1, -1, -1)
                    for k in at(row, c)
                    if k not in home and lettered(cells[k][0].text)
                ),
                None,
            )
            header = next(
                (
                    k
                    for r in range(row - 1, -1, -1)
                    for k in at(r, column)
                    if k not in home
                    and cells[k][0].column_header
                    and cells[k][0].text.strip()
                ),
                None,
            )
            key = next((k for k in (own, caption, header) if k is not None), None)
            if key is None:
                continue
            context = header if header is not None and header != key else None
            fields.append(
                Candidate(
                    (i,),
                    label_of(key),
                    "table_cell",
                    {},
                    context=None if context is None else label_of(context),
                )
            )
    return fields


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
