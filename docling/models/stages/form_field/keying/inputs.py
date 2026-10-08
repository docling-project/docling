# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Values and candidate keys of one page: widgets, text atoms and joined key text."""

from __future__ import annotations

import statistics
from collections import defaultdict

from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import PdfWidget

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.keying.geometry import (
    contains,
    gap,
    lettered,
    overlap,
    span_overlap,
)
from docling.models.stages.form_field.keying.types import (
    KeyText,
    Scope,
    Value,
    is_skipped,
)

# Pages with more text atoms than this skip key joining and abstain from
# free-form keying.
MAX_KEY_TEXTS = 3000
# Layout regions whose cells never become keys.
NOT_KEYS = {
    DocItemLabel.FORM,
    DocItemLabel.KEY_VALUE_REGION,
    DocItemLabel.TABLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.PICTURE,
}


def regions(clusters: list[Cluster]) -> list[Cluster]:
    """Every layout cluster once, children included, ordered by id."""
    found: dict[int, Cluster] = {}
    pending = list(clusters)
    while pending:
        region = pending.pop()
        if region.id not in found:
            found[region.id] = region
            pending.extend(region.children)
    return sorted(found.values(), key=lambda r: r.id)


def scope_of(
    bbox: BoundingBox, found: list[Cluster], table_cells: dict[int, list[TableCell]]
) -> Scope:
    """The detected table, and its cell, that a top-left box belongs to.

    ``found`` is the flat list of layout clusters (see ``regions``) and
    ``table_cells`` the detected cells of each table cluster, by cluster id.
    """
    tables = [
        r
        for r in found
        if r.label in {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
        and overlap(r.bbox, bbox) > 0
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
    cells = [
        i
        for i, cell in enumerate(table_cells.get(table.id, []))
        if cell.bbox is not None and contains(cell.bbox, bbox)
    ]
    return Scope(table.id, cells[0] if len(cells) == 1 else None)


def paints_value(box: BoundingBox, text: str, values: list[Value]) -> bool:
    """Whether the text is a filled value its widget paints into the page.

    Such text sits inside the widget's box and equals the widget's value; it
    is the value itself, never a key.
    """
    return any(
        contains(v.bbox, box)
        and "".join(text.split()) == "".join((v.native.widget_text or "").split())
        for v in values
    )


def inputs(
    widgets: list[PdfWidget],
    found: list[Cluster],
    table_cells: dict[int, list[TableCell]],
    page_height: float,
) -> tuple[
    list[Value],
    list[KeyText],
    float,
    dict[int, set[tuple[int, int]]],
    set[tuple[int, int]],
]:
    """Values, key texts, median text height, atom sources, and painted source cells."""
    values = []
    for native in widgets:
        bbox = native.rect.to_bounding_box().to_top_left_origin(page_height)
        if is_skipped(native, bbox):
            continue
        values.append(Value(native, bbox, scope_of(bbox, found, table_cells)))

    # Atom identity is source-backed and shared by all overlapping span choices.
    atoms: dict[tuple, int] = {}
    sources: dict[int, set[tuple[int, int]]] = defaultdict(set)
    key_texts: dict[tuple[frozenset[int], Scope], KeyText] = {}
    heights = []
    painted_cells: set[tuple[int, int]] = set()
    for region in found:
        if region.label in NOT_KEYS:
            continue
        by_scope: dict[Scope, list[tuple[int, str, BoundingBox]]] = defaultdict(list)
        for cell in region.cells:
            text = cell.text.strip()
            box = cell.rect.to_bounding_box().to_top_left_origin(page_height)
            if not text:
                continue
            # Cleanup also reads cells excluded by candidate area or table scope.
            if paints_value(box, text, values):
                painted_cells.add((region.id, cell.index))
            if box.area() <= 0 or (region.id, cell.index) in painted_cells:
                continue
            scope = scope_of(box, found, table_cells)
            if not scope.eligible:
                continue
            atom = atoms.setdefault(
                (cell.index, tuple(box.as_tuple()), text), len(atoms)
            )
            sources[atom].add((region.id, cell.index))
            heights.append(box.height)
            by_scope[scope].append((atom, text, box))
        for scope, cells in by_scope.items():
            for bundle in [cells, *[[cell] for cell in cells]]:
                ids = frozenset(c[0] for c in bundle)
                bbox = BoundingBox.enclosing_bbox([c[2] for c in bundle])
                if scope_of(bbox, found, table_cells) != scope:
                    continue
                key_texts[ids, scope] = KeyText(
                    " ".join(c[1] for c in bundle),
                    bbox,
                    ids,
                    scope,
                    region.label,
                    len(bundle) < len(cells),
                )
    h = statistics.median(heights) if heights else 1.0
    if len(key_texts) > MAX_KEY_TEXTS:
        # Joining keys compares every pair of blocks; a page this large
        # abstains in assign() anyway.
        return values, list(key_texts.values()), h, sources, painted_cells
    # Rebuild whole keys the layout split into pieces: first the pieces of
    # one text line, then the lines of one key. Joined keys compete
    # with their pieces; they never replace them.
    blocks = lines(
        [key_text for key_text in key_texts.values() if not key_text.fragment],
        values,
        h,
    )
    for key_text in [*blocks, *stacks(blocks, values, h)]:
        key_texts.setdefault((key_text.atoms, key_text.scope), key_text)
    return values, list(key_texts.values()), h, sources, painted_cells


def joined(parts: list[KeyText]) -> KeyText:
    return KeyText(
        " ".join(part.text for part in parts),
        BoundingBox.enclosing_bbox([part.bbox for part in parts]),
        frozenset().union(*(part.atoms for part in parts)),
        parts[0].scope,
        parts[0].layout_label,
        stack=True,
    )


def nested(a: frozenset[int], b: frozenset[int]) -> bool:
    return a <= b or b <= a


def lines(blocks: list[KeyText], values: list[Value], h: float) -> list[KeyText]:
    """Text lines the layout split into side-by-side blocks, joined back.

    Two blocks of one line join when they are each other's nearest neighbour,
    have the same height, are under a text line apart and have no value
    between them. Pieces of one line share the lines just above and below it;
    blocks with different neighbouring lines, or that each key their own
    value directly above or below (sub-keys over separate boxes), are
    separate cells and stay apart. Returns the blocks with every joined run
    replaced by its line.
    """

    def boxes_at(b: KeyText) -> frozenset[int]:
        return frozenset(
            i
            for i, v in enumerate(values)
            if v.scope == b.scope
            and span_overlap(b.bbox.l, b.bbox.r, v.bbox.l, v.bbox.r)
            >= 0.5 * min(b.bbox.width, v.bbox.width)
            and gap(b.bbox, v.bbox) <= h
        )

    def around(b: KeyText) -> frozenset[int]:
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


def stacks(blocks: list[KeyText], values: list[Value], h: float) -> list[KeyText]:
    """Keys the layout split into one block per line, joined back.

    Two single-line blocks of the same line height (one font size, so never a
    heading over a key) join when they are each other's only neighbour
    across a gap of under three quarters of a text line, share a left edge or
    a centre, and the upper block aligns with no value the lower one misses: a
    key ends at its value's row, so a line below a key that already
    sits beside its value starts another key. A heading over several
    sub-keys has several
    neighbours and never joins one of them; a line with a value inside it is an
    inline key of its own. A key merely touching its value's edge
    still joins. Chains stop at four blocks.
    """
    joinable = [
        b
        for b in blocks
        if lettered(b.text)
        and b.bbox.height <= 1.5 * h
        and b.layout_label
        not in {
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
            DocItemLabel.TITLE,
        }
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
    keys = []
    for start in range(len(joinable)):
        chain = [joinable[start]]
        index = start
        while index in below and len(chain) < 4:
            index = below[index]
            chain.append(joinable[index])
            keys.append(joined(chain))
    return keys
