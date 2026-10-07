# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Global choice of associations: pairwise penalties and the MILP."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations

import numpy as np
from docling_core.types.doc import BoundingBox, TableCell
from docling_core.types.doc.page import PdfWidget
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from docling.datamodel.base_models import Cluster
from docling.models.stages.form_field.keying.candidates import (
    candidates_for,
    siblings,
)
from docling.models.stages.form_field.keying.geometry import anchors, span_overlap
from docling.models.stages.form_field.keying.inputs import (
    MAX_LABELS,
    inputs,
    regions,
)
from docling.models.stages.form_field.keying.symmetry import (
    SIDE_COST,
    mark_sequences,
    sequences,
    structures,
)
from docling.models.stages.form_field.keying.tables import add_context, table_fields
from docling.models.stages.form_field.keying.types import (
    Assignment,
    Candidate,
    Label,
    TableSlot,
    Value,
)

# Page size limits. Candidate construction scans every label against every
# value, and the pairwise penalties every pair of surviving candidates, both in
# Python; a page beyond these limits would hold the stage's thread for tens of
# seconds. Such a page keeps its table keys and leaves free-form values
# unkeyed, as a solver timeout does.
MAX_LABEL_VALUE_PAIRS = 1_000_000
MAX_ACTIVE_CANDIDATES = 1000
# Cost of leaving a value without a key.
NULL_COST = 3.0
# Seconds the MILP solver may spend on one page.
TIME_LIMIT = 10.0


def crossing(
    a: Candidate, b: Candidate, values: list[Value], labels: list[Label]
) -> bool:
    if (
        a.kind == "choice_group"
        or b.kind == "choice_group"
        or set(a.members) & set(b.members)
    ):
        return False

    def turn(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for i in a.members:
        for j in b.members:
            p, q = anchors(labels[a.label].bbox, values[i].bbox)
            r, s = anchors(labels[b.label].bbox, values[j].bbox)
            if (
                turn(p, q, r) * turn(p, q, s) < -1e-9
                and turn(r, s, p) * turn(r, s, q) < -1e-9
            ):
                return True
    return False


def transition_penalty(
    a: Candidate, b: Candidate, values: list[Value], labels: list[Label], h: float
) -> float:
    """Compare label movement to the FIXED adjacent value movement in one lane.

    A two-column reset is not comparable within a lane and incurs no penalty.
    This never proposes or chooses a different native value order.
    """
    if a.kind == "choice_group" or b.kind == "choice_group":
        return 0.0
    if a.members[0] > b.members[0]:
        a, b = b, a
    if a.members[-1] + 1 != b.members[0]:
        return 0.0
    va, vb = values[a.members[-1]].bbox, values[b.members[0]].bbox
    la, lb = labels[a.label].bbox, labels[b.label].bbox
    if abs(va.t - vb.t) <= h:
        return 3.0 if (vb.l - va.l) * (lb.l - la.l) < -(h * h) else 0.0
    if min(va.r, vb.r) > max(va.l, vb.l):
        return 3.0 if (vb.t - va.t) * (lb.t - la.t) < -(h * h) else 0.0
    return 0.0


def share_groups(
    columns: list[int],
    proposed: list[Candidate],
    active: list[int],
    values: list[Value],
) -> list[list[int]]:
    """Partition the candidates using one text atom into groups that may co-own it.

    Single-value captions reaching the same label from the same side join a
    group when their values are siblings (transitively, along the line).
    Every other candidate is a group of its own.
    """
    parent = {column: column for column in columns}

    def root(column: int) -> int:
        while parent[column] != column:
            column = parent[column]
        return column

    for x, y in combinations(columns, 2):
        cx, cy = proposed[active[x]], proposed[active[y]]
        if (
            cx.side is not None
            and cx.side == cy.side
            and cx.label == cy.label
            and siblings(values[cx.members[0]], values[cy.members[0]], cx.side)
        ):
            parent[root(x)] = root(y)
    groups: dict[int, list[int]] = defaultdict(list)
    for column in columns:
        groups[root(column)].append(column)
    return list(groups.values())


def co_owners(a: Candidate, b: Candidate, values: list[Value]) -> bool:
    """A checkbox and a text value of one row reading the same caption.

    A row's flags and amounts form two lines under one row caption, and a
    printed cell's caption keys both its text box (below) and its checkbox
    (beside). The same caption in a column keys both kinds from the same side.
    """
    if a.side is None or b.side is None or a.label != b.label:
        return False
    x, y = values[a.members[0]].bbox, values[b.members[0]].bbox
    if values[a.members[0]].checkbox == values[b.members[0]].checkbox:
        return False
    if span_overlap(x.t, x.b, y.t, y.b) >= 0.5 * min(x.height, y.height):
        return True
    return (
        a.side == b.side
        and a.side in ("up", "down")
        and span_overlap(x.l, x.r, y.l, y.r) >= 0.5 * min(x.width, y.width)
    )


def abstained(
    values: list[Value],
    labels: list[Label],
    candidates: list[Candidate],
    status: str,
    slots: dict[int, TableSlot],
) -> Assignment:
    """No free-form key on this page; the table keys and cells still hold."""
    fixed = [i for i, c in enumerate(candidates) if c.kind == "table_cell"]
    return Assignment(values, labels, candidates, fixed, status, slots)


def _sparse(rows: list[dict[int, float]], columns: int):
    """The constraint matrix, rows and columns in their construction order."""
    rr, cc, data = [], [], []
    for r, row in enumerate(rows):
        for c, coefficient in row.items():
            rr.append(r)
            cc.append(c)
            data.append(coefficient)
    return coo_matrix((data, (rr, cc)), shape=(len(rows), columns)).tocsc()


def _one_side_per_structure(
    groups: list[list[int]],
    facing: dict[int, list[int]],
    proposed: list[Candidate],
    active: list[int],
    costs: list[float],
    constraint: Callable[[dict[int, float], float, float], None],
) -> None:
    """A repeated structure of like fields reads its captions from one side.

    The structure, along rows and columns at once, picks one side (one switch
    per side seen, exactly one on), and each field reading another side pays
    SIDE_COST. The side is not fixed in advance: the majority sets it.
    """
    for group in groups:
        by_side: dict[str, list[int]] = defaultdict(list)
        for i in group:
            for x in facing[i]:
                by_side[str(proposed[active[x]].side)].append(x)
        if len(by_side) < 2:
            continue
        switches = {}
        for side in sorted(by_side):
            switches[side] = len(costs)
            costs.append(0.0)
        constraint(dict.fromkeys(switches.values(), 1.0), 1, 1)
        for side, columns in sorted(by_side.items()):
            for x in columns:
                constraint({x: 1, switches[side]: -1, len(costs): -1}, -math.inf, 0)
                costs.append(SIDE_COST)


def assign(
    widgets: list[PdfWidget],
    clusters: list[Cluster],
    table_cells: dict[int, list[TableCell]],
    page_height: float,
    rules: list[BoundingBox] | None = None,
) -> Assignment:
    """Choose the caption of every widget value on one page.

    ``clusters`` are the page's layout clusters and ``table_cells`` the
    detected cells of each table cluster, by cluster id. ``rules`` are the
    boxes of printed rules (top-left origin); they are only read for values in
    detected tables. Nothing passed in is modified.
    """
    found = regions(clusters)
    values, labels, h, sources, painted_cells = inputs(
        widgets, found, table_cells, page_height
    )

    def keyed_in_tables() -> tuple[list[Candidate], dict[int, TableSlot]]:
        return table_fields(
            found, table_cells, rules or [], page_height, values, labels
        )

    assignment = _solve(values, labels, h, keyed_in_tables)
    assignment.sources = sources
    assignment.painted_cells = painted_cells
    return assignment


def _solve(
    values: list[Value],
    labels: list[Label],
    h: float,
    keyed_in_tables: Callable[[], tuple[list[Candidate], dict[int, TableSlot]]],
) -> Assignment:
    if len(labels) > MAX_LABELS or len(values) * len(labels) > MAX_LABEL_VALUE_PAIRS:
        tables, slots = keyed_in_tables()
        return abstained(
            values,
            labels,
            tables,
            f"skipped: {len(values)} values x {len(labels)} labels exceed the page limit",
            slots,
        )
    proposed = candidates_for(values, labels, h)
    # A decided alternating run fixes which caption each of its values reads.
    mark_sequences(proposed, sequences(values, labels, proposed, NULL_COST, h))
    # Values in detected tables are keyed from their own cells, outside the
    # free-form search: table text never competes with free-form captions.
    tables, slots = keyed_in_tables()
    fixed = list(range(len(proposed), len(proposed) + len(tables)))
    # A candidate worse than leaving all its members blank cannot help: all
    # remaining interactions are penalties. Keep the full list for diagnostics.
    active = [
        i
        for i, c in enumerate(proposed)
        if c.cost < (0 if c.kind == "choice_group" else NULL_COST * len(c.members))
    ]
    if len(active) > MAX_ACTIVE_CANDIDATES:
        return abstained(
            values,
            labels,
            proposed + tables,
            f"skipped: {len(active)} candidates exceed the page limit",
            slots,
        )
    costs = [proposed[i].cost for i in active]
    rows: list[dict[int, float]] = []
    lower, upper = [], []

    def constraint(coefficients: dict[int, float], lo: float, hi: float) -> None:
        rows.append(coefficients)
        lower.append(lo)
        upper.append(hi)

    owners: dict[int, dict[int, float]] = defaultdict(dict)
    questions: dict[int, dict[int, float]] = defaultdict(dict)
    spans: dict[int, dict[int, float]] = defaultdict(dict)
    for column, index in enumerate(active):
        candidate = proposed[index]
        for i in candidate.members:
            (questions if candidate.kind == "choice_group" else owners)[i][column] = 1.0
        for atom in labels[candidate.label].atoms:
            spans[atom][column] = 1.0
    for i, value in enumerate(values):
        if not value.scope.eligible:
            continue
        owners[i][len(costs)] = 1.0
        costs.append(NULL_COST)
        constraint(owners[i], 1, 1)
    for coefficients in questions.values():
        constraint(coefficients, 0, 1)
    # Each text atom keys one line of sibling values, so each shareable group
    # gets one switch. A row's flags and amounts form two lines under the same
    # caption; any other pair of groups excludes each other.
    for columns in spans.values():
        groups = share_groups(sorted(columns), proposed, active, values)
        switches = []
        for group in groups:
            if len(group) == 1:
                switches.append(group[0])
                continue
            switches.append(len(costs))
            costs.append(0.0)
            for column in group:
                constraint({column: 1, switches[-1]: -1}, -math.inf, 0)
        for (ga, sa), (gb, sb) in combinations(zip(groups, switches), 2):
            first, second = proposed[active[ga[0]]], proposed[active[gb[0]]]
            if not co_owners(first, second, values):
                constraint({sa: 1, sb: 1}, 0, 1)
    # Every pair of candidates is compared; MAX_ACTIVE_CANDIDATES bounds this.
    for a, b in combinations(range(len(active)), 2):
        ca, cb = proposed[active[a]], proposed[active[b]]
        if labels[ca.label].scope != labels[cb.label].scope:
            continue
        penalty = 3.0 * crossing(ca, cb, values, labels) + transition_penalty(
            ca, cb, values, labels, h
        )
        if not penalty:
            continue
        auxiliary = len(costs)
        costs.append(penalty)
        constraint({a: 1, b: 1, auxiliary: -1}, -math.inf, 1)
        constraint({auxiliary: 1, a: -1}, -math.inf, 0)
        constraint({auxiliary: 1, b: -1}, -math.inf, 0)
    # Aligned like-sized neighbours in the native order usually read their
    # captions from the same side: a soft cost, never a rule. It stays below
    # any cell or alignment difference, so it only settles close calls.
    facing: dict[int, list[int]] = defaultdict(list)
    for column, index in enumerate(active):
        if proposed[index].side is not None:
            facing[proposed[index].members[0]].append(column)
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if not (siblings(a, b, "left") or siblings(a, b, "up")):
            continue
        for x in facing[i]:
            for y in facing[i + 1]:
                if proposed[active[x]].side != proposed[active[y]].side:
                    constraint({x: 1, y: 1, len(costs): -1}, -math.inf, 1)
                    costs.append(0.5)
    _one_side_per_structure(
        structures(values, labels, proposed, NULL_COST),
        facing,
        proposed,
        active,
        costs,
        constraint,
    )
    if not costs:
        return Assignment(values, labels, proposed + tables, fixed, "optimal", slots)
    matrix = _sparse(rows, len(costs))
    result = milp(
        np.array(costs),
        integrality=np.ones(len(costs)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.0},
    )
    if result.status != 0:
        # Abstain on timeout: no unsupported confidence claims about a partial
        # solution and no missing native values.
        return abstained(values, labels, proposed + tables, str(result.message), slots)
    selected = [index for column, index in enumerate(active) if result.x[column] > 0.5]
    add_context([proposed[i] for i in selected], proposed, values, NULL_COST)
    return Assignment(
        values,
        labels,
        proposed + tables,
        selected + fixed,
        "optimal",
        slots,
    )
