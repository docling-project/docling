# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Global choice of associations: pairwise penalties and the MILP."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np

from docling.models.stages.form_field.keying.candidates import (
    candidates_for,
    siblings,
)
from docling.models.stages.form_field.keying.geometry import anchors, span_overlap
from docling.models.stages.form_field.keying.inputs import MAX_LABELS, inputs
from docling.models.stages.form_field.keying.tables import add_context, table_fields
from docling.models.stages.form_field.keying.types import (
    Assignment,
    Candidate,
    Label,
    Snapshot,
    Value,
)

# Page size limits. Candidate construction scans every label against every
# value, and the pairwise penalties every pair of surviving candidates, both in
# Python; a page far beyond the forms this was built on (at most 163 values,
# 401 labels and 361 surviving candidates) would hold the stage's thread for
# tens of seconds. Such a page keeps its table keys and leaves free-form
# values unkeyed, as a solver timeout does.
MAX_LABEL_VALUE_PAIRS = 1_000_000
MAX_ACTIVE_CANDIDATES = 1000


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
    null_cost: float,
) -> Assignment:
    """No free-form key on this page; the table keys at the end still hold."""
    fixed = [i for i, c in enumerate(candidates) if c.kind == "table_cell"]
    return Assignment(
        values,
        labels,
        candidates,
        fixed,
        status,
        null_cost * sum(v.scope.eligible for v in values),
    )


def _validated(null_cost: float, time_limit: float) -> None:
    if (
        not math.isfinite(null_cost)
        or null_cost <= 0
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("Costs and time limits must be positive and finite")


def _sparse(rows: list[dict[int, float]], columns: int):
    """The constraint matrix, rows and columns in their construction order."""
    from scipy.sparse import coo_matrix

    rr, cc, data = [], [], []
    for r, row in enumerate(rows):
        for c, coefficient in row.items():
            rr.append(r)
            cc.append(c)
            data.append(coefficient)
    return coo_matrix((data, (rr, cc)), shape=(len(rows), columns)).tocsc()


def assign(
    snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0
) -> Assignment:
    _validated(null_cost, time_limit)
    # SciPy's MILP solver is only needed when a page has widgets; keep the
    # import off the pipeline's import path.
    from scipy.optimize import Bounds, LinearConstraint, milp

    values, labels, h = inputs(snapshot)
    if len(labels) > MAX_LABELS or len(values) * len(labels) > MAX_LABEL_VALUE_PAIRS:
        tables = table_fields(snapshot, values, labels)
        return abstained(
            values,
            labels,
            tables,
            f"skipped: {len(values)} values x {len(labels)} labels exceed the page limit",
            null_cost,
        )
    proposed = candidates_for(values, labels, h)
    # Values in detected tables are keyed from their own cells, outside the
    # free-form search: table text never competes with free-form captions.
    tables = table_fields(snapshot, values, labels)
    fixed = list(range(len(proposed), len(proposed) + len(tables)))
    # A candidate worse than leaving all its members blank cannot help: all
    # remaining interactions are penalties. Keep the full list for diagnostics.
    active = [
        i
        for i, c in enumerate(proposed)
        if c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))
    ]
    if len(active) > MAX_ACTIVE_CANDIDATES:
        return abstained(
            values,
            labels,
            proposed + tables,
            f"skipped: {len(active)} candidates exceed the page limit",
            null_cost,
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
        costs.append(null_cost)
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
    # ponytail: quadratic candidate scan for small saved pages; partition by
    # connected candidates before considering larger documents.
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
    if not costs:
        return Assignment(values, labels, proposed + tables, fixed, "optimal", 0)
    matrix = _sparse(rows, len(costs))
    result = milp(
        np.array(costs),
        integrality=np.ones(len(costs)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if result.status != 0:
        # Abstain on timeout: no unsupported confidence claims about a partial
        # solution and no missing native values.
        return abstained(
            values, labels, proposed + tables, str(result.message), null_cost
        )
    selected = [index for column, index in enumerate(active) if result.x[column] > 0.5]
    add_context([proposed[i] for i in selected], proposed, values, null_cost)
    return Assignment(
        values,
        labels,
        proposed + tables,
        selected + fixed,
        "optimal",
        float(result.fun),
    )
