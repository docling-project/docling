# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Values and labels of one page: widgets, text atoms and rebuilt captions."""

from __future__ import annotations

import statistics
from collections import defaultdict

from docling_core.types.doc import BoundingBox

from docling.models.stages.form_field.keying.geometry import (
    contains,
    gap,
    lettered,
    overlap,
    span_overlap,
)
from docling.models.stages.form_field.keying.types import (
    Label,
    Region,
    Scope,
    Snapshot,
    Value,
)

# Pages with more text atoms than this (the development forms have at most
# 401 labels) skip caption joining and abstain from free-form keying.
MAX_LABELS = 3000


def regions(snapshot: Snapshot) -> list[Region]:
    found: dict[int, Region] = {}
    pending = list(snapshot.layout)
    while pending:
        region = pending.pop()
        if region.id not in found:
            found[region.id] = region
            pending.extend(region.children)
    return sorted(found.values(), key=lambda r: r.id)


def scope_of(
    bbox: BoundingBox, snapshot: Snapshot, found: list[Region] | None = None
) -> Scope:
    tables = [
        r
        for r in (regions(snapshot) if found is None else found)
        if r.label in {"table", "document_index"} and overlap(r.bbox, bbox) > 0
    ]
    if not tables:
        return Scope()
    # Any intersection excludes a box from the outside pool. A local exception
    # requires complete containment in one unambiguous detected cell.
    table = min(tables, key=lambda r: (r.bbox.area(), r.id))
    if not contains(table.bbox, bbox) or any(
        not contains(other.bbox, table.bbox) for other in tables if other.id != table.id
    ):
        return Scope(table.id)
    structure = snapshot.tables.table_map.get(table.id)
    cells = (
        []
        if structure is None
        else [
            i
            for i, cell in enumerate(structure.table_cells)
            if cell.bbox is not None and contains(cell.bbox, bbox)
        ]
    )
    return Scope(table.id, cells[0] if len(cells) == 1 else None)


def inputs(snapshot: Snapshot) -> tuple[list[Value], list[Label], float]:
    found = regions(snapshot)
    values = []
    seen_indices: set[int] = set()
    for native in snapshot.widgets:
        if native.index in seen_indices:
            raise ValueError(f"Duplicate native widget index: {native.index}")
        seen_indices.add(native.index)
        bbox = native.rect.to_bounding_box().to_top_left_origin(snapshot.size.height)
        if (
            bbox.width <= 0
            or bbox.height <= 0
            or (
                native.widget_field_type == "/Btn"
                and native.widget_field_flags & (1 << 16)
            )
        ):
            continue
        values.append(Value(native, bbox, scope_of(bbox, snapshot, found)))

    # Atom identity is source-backed and shared by all overlapping span choices.
    atoms: dict[tuple, int] = {}
    labels: dict[tuple[frozenset[int], Scope], Label] = {}
    heights = []
    for region in found:
        if region.label in {
            "form",
            "key_value_region",
            "table",
            "document_index",
            "picture",
        }:
            continue
        by_scope: dict[Scope, list[tuple[int, str, BoundingBox]]] = defaultdict(list)
        for cell in region.cells:
            text = cell.text.strip()
            box = cell.rect.to_bounding_box().to_top_left_origin(snapshot.size.height)
            if not text or box.area() <= 0:
                continue
            scope = scope_of(box, snapshot, found)
            if not scope.eligible:
                continue
            if any(
                contains(v.bbox, box)
                and "".join(text.split())
                == "".join((v.native.widget_text or "").split())
                for v in values
            ):
                continue
            atom = atoms.setdefault(
                (cell.index, tuple(box.as_tuple()), text), len(atoms)
            )
            heights.append(box.height)
            by_scope[scope].append((atom, text, box))
        for scope, cells in by_scope.items():
            for bundle in [cells, *[[cell] for cell in cells]]:
                ids = frozenset(c[0] for c in bundle)
                bbox = BoundingBox.enclosing_bbox([c[2] for c in bundle])
                if scope_of(bbox, snapshot, found) != scope:
                    continue
                labels[ids, scope] = Label(
                    " ".join(c[1] for c in bundle),
                    bbox,
                    ids,
                    scope,
                    region.label,
                    len(bundle) < len(cells),
                )
    h = statistics.median(heights) if heights else 1.0
    if len(labels) > MAX_LABELS:
        # Joining captions compares every pair of blocks; a page this far
        # beyond the development forms abstains in assign() anyway.
        return values, list(labels.values()), h
    # Rebuild whole captions the layout split into pieces: first the pieces of
    # one text line, then the lines of one caption. Joined captions compete
    # with their pieces; they never replace them.
    blocks = lines(
        [label for label in labels.values() if not label.fragment], values, h
    )
    for label in [*blocks, *stacks(blocks, values, h)]:
        labels.setdefault((label.atoms, label.scope), label)
    return values, list(labels.values()), h


def joined(parts: list[Label]) -> Label:
    return Label(
        " ".join(part.text for part in parts),
        BoundingBox.enclosing_bbox([part.bbox for part in parts]),
        frozenset().union(*(part.atoms for part in parts)),
        parts[0].scope,
        parts[0].role,
        stack=True,
    )


def nested(a: frozenset[int], b: frozenset[int]) -> bool:
    return a <= b or b <= a


def lines(blocks: list[Label], values: list[Value], h: float) -> list[Label]:
    """Text lines the layout split into side-by-side blocks, joined back.

    Two blocks of one line join when they are each other's nearest neighbour,
    have the same height, are under a text line apart and have no value
    between them. Pieces of one line share the lines just above and below it;
    blocks with different neighbouring lines, or that each caption their own
    value directly above or below (sub-captions over separate boxes), are
    separate cells and stay apart. Returns the blocks with every joined run
    replaced by its line.
    """

    def boxes_at(b: Label) -> frozenset[int]:
        return frozenset(
            i
            for i, v in enumerate(values)
            if v.scope == b.scope
            and span_overlap(b.bbox.l, b.bbox.r, v.bbox.l, v.bbox.r)
            >= 0.5 * min(b.bbox.width, v.bbox.width)
            and gap(b.bbox, v.bbox) <= h
        )

    def around(b: Label) -> frozenset[int]:
        return frozenset(
            k
            for k, other in enumerate(blocks)
            if other.scope == b.scope
            and span_overlap(b.bbox.l, b.bbox.r, other.bbox.l, other.bbox.r) > 0
            and 0 < max(other.bbox.t - b.bbox.b, b.bbox.t - other.bbox.b) <= 0.75 * h
        )

    below = [boxes_at(b) for b in blocks]
    neighbours = [around(b) for b in blocks]
    nearest: dict[int, int] = {}
    for i, a in enumerate(blocks):
        right = [
            j
            for j, b in enumerate(blocks)
            if j != i
            and b.scope == a.scope
            and min(a.bbox.height, b.bbox.height)
            >= 0.8 * max(a.bbox.height, b.bbox.height)
            and span_overlap(a.bbox.t, a.bbox.b, b.bbox.t, b.bbox.b)
            >= 0.8 * min(a.bbox.height, b.bbox.height)
            and -0.25 * h <= b.bbox.l - a.bbox.r <= h
        ]
        if right:
            nearest[i] = min(right, key=lambda j: (blocks[j].bbox.l, j))
    links: dict[int, int] = {}
    for i, j in nearest.items():
        a, b = blocks[i].bbox, blocks[j].bbox
        between = BoundingBox(
            l=min(a.r, b.l), t=max(a.t, b.t), r=max(a.r, b.l), b=min(a.b, b.b)
        )
        if (
            [k for k, n in nearest.items() if n == j] == [i]
            and nested(below[i], below[j])
            and neighbours[i] == neighbours[j]
            and not any(overlap(v.bbox, between) > 0 for v in values)
        ):
            links[i] = j
    result = []
    for start, block in enumerate(blocks):
        if start in links.values():
            continue
        run = [start]
        while run[-1] in links:
            run.append(links[run[-1]])
        result.append(block if len(run) == 1 else joined([blocks[k] for k in run]))
    return result


def stacks(blocks: list[Label], values: list[Value], h: float) -> list[Label]:
    """Captions the layout split into one block per line, joined back.

    Two single-line blocks of the same line height (one font size, so never a
    heading over a caption) join when they are each other's only neighbour
    across a gap of under three quarters of a text line, share a left edge or
    a centre, and the upper block aligns with no value the lower one misses: a
    caption ends at its value's row, so a line below a caption that already
    sits beside its value starts another caption. A heading over several
    sub-captions has several
    neighbours and never joins one of them; a line with a value inside it is an
    inline caption of its own. A caption merely touching its value's edge
    still joins. Chains stop at four blocks.
    """
    joinable = [
        b
        for b in blocks
        if lettered(b.text)
        and b.bbox.height <= 1.5 * h
        and b.role not in {"section_header", "page_header", "page_footer", "title"}
        and not any(
            overlap(b.bbox, v.bbox) > 0
            and span_overlap(b.bbox.t, b.bbox.b, v.bbox.t, v.bbox.b)
            >= 0.5 * min(b.bbox.height, v.bbox.height)
            for v in values
        )
    ]

    def adjacent(upper: BoundingBox, lower: BoundingBox) -> bool:
        return (
            lower.t > upper.t
            and -0.25 * h <= lower.t - upper.b <= 0.75 * h
            and span_overlap(upper.l, upper.r, lower.l, lower.r)
            >= 0.5 * min(upper.width, lower.width)
            and min(upper.height, lower.height)
            >= 0.95 * max(upper.height, lower.height)
        )

    rows = [
        frozenset(
            i
            for i, v in enumerate(values)
            if v.scope == b.scope
            and span_overlap(b.bbox.t, b.bbox.b, v.bbox.t, v.bbox.b)
            >= 0.5 * min(b.bbox.height, v.bbox.height)
        )
        for b in joinable
    ]
    below: dict[int, int] = {}
    for i, upper in enumerate(joinable):
        under = [
            j
            for j, lower in enumerate(joinable)
            if lower.scope == upper.scope and adjacent(upper.bbox, lower.bbox)
        ]
        if len(under) != 1:
            continue
        (j,) = under
        u, lower = upper.bbox, joinable[j].bbox
        over = [
            k
            for k, other in enumerate(joinable)
            if other.scope == upper.scope and adjacent(other.bbox, lower)
        ]
        if (
            over == [i]
            and (abs(lower.l - u.l) <= h or abs(lower.l + lower.r - u.l - u.r) <= 2 * h)
            and rows[i] <= rows[j]
        ):
            below[i] = j
    captions = []
    for start in range(len(joinable)):
        chain = [joinable[start]]
        index = start
        while index in below and len(chain) < 4:
            index = below[index]
            chain.append(joinable[index])
            captions.append(joined(chain))
    return captions
