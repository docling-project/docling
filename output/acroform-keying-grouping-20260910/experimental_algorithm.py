# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Offline AcroForm association experiment; not imported by the PDF pipeline.

Only saved native/layout/table inputs enter candidate construction. Reviewed
annotations belong in the evaluator, never in this module. Requires SciPy >=1.9.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from docling_core.types.doc import BoundingBox, Size, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCell
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# Match the existing form stage's text-container coverage threshold. This
# tolerates imperfect text bounds only for inline-clause proposals; detected
# table/cell ownership still requires complete containment in scope_of().
INLINE_WIDGET_COVERAGE = 0.8


class NativeWidget(BaseModel):
    """Explicit snapshot contract, independent of installed parser bindings."""

    index: int
    rect: BoundingRectangle
    widget_text: str | None = None
    widget_description: str | None = None
    widget_field_name: str | None = None
    widget_field_type: str | None = None
    widget_field_flags: int
    widget_appearance_state: str | None


class Region(BaseModel):
    id: int
    label: str
    bbox: BoundingBox
    cells: list[TextCell] = Field(default_factory=list)
    children: list[Region] = Field(default_factory=list)


class DetectedTable(BaseModel):
    table_cells: list[TableCell]


class Tables(BaseModel):
    table_map: dict[int, DetectedTable] = Field(default_factory=dict)


class Snapshot(BaseModel):
    page: int
    size: Size
    widgets: list[NativeWidget]
    layout: list[Region]
    tables: Tables


@dataclass(frozen=True)
class Scope:
    table: int | None = None
    cell: int | None = None

    @property
    def eligible(self) -> bool:
        return self.table is None or self.cell is not None


@dataclass
class Value:
    native: NativeWidget
    bbox: BoundingBox
    scope: Scope

    @property
    def checkbox(self) -> bool:
        return self.native.widget_field_type == "/Btn"


@dataclass
class Label:
    text: str
    bbox: BoundingBox
    atoms: frozenset[int]
    scope: Scope
    role: str
    fragment: bool = False


@dataclass
class Candidate:
    members: tuple[int, ...]  # Positions in the supplied value list, not y order.
    label: int
    kind: Literal[
        "field_key",
        "option_caption",
        "composite_field",
        "inline_clause",
        "choice_group",
    ]
    features: dict[str, float]
    option_labels: tuple[int, ...] = ()

    @property
    def cost(self) -> float:
        return sum(self.features.values())


@dataclass
class Assignment:
    values: list[Value]
    labels: list[Label]
    candidates: list[Candidate]
    selected: list[int]
    solver_status: str
    objective: float


def overlap(a: BoundingBox, b: BoundingBox) -> float:
    return max(0.0, min(a.r, b.r) - max(a.l, b.l)) * max(
        0.0, min(a.b, b.b) - max(a.t, b.t)
    )


def contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (
        outer.l <= inner.l + 1e-6
        and outer.t <= inner.t + 1e-6
        and outer.r >= inner.r - 1e-6
        and outer.b >= inner.b - 1e-6
    )


def gap(a: BoundingBox, b: BoundingBox) -> float:
    return max(0.0, a.l - b.r, b.l - a.r) + max(0.0, a.t - b.b, b.t - a.b)


def anchors(
    a: BoundingBox, b: BoundingBox
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Facing boundary points; use the shared band when boxes are aligned."""

    def axis(
        lo: float, hi: float, other_lo: float, other_hi: float
    ) -> tuple[float, float]:
        if hi < other_lo:
            return hi, other_lo
        if other_hi < lo:
            return lo, other_hi
        mid = (max(lo, other_lo) + min(hi, other_hi)) / 2
        return mid, mid

    ax, bx = axis(a.l, a.r, b.l, b.r)
    ay, by = axis(a.t, a.b, b.t, b.b)
    return (ax, ay), (bx, by)


def obstructs(
    a: tuple[float, float], b: tuple[float, float], box: BoundingBox, tolerance: float
) -> bool:
    """Segment intersects the rectangle interior; mere border touches do not count."""
    lower, upper = 0.0, 1.0
    for start, end, lo, hi in zip(a, b, (box.l, box.t), (box.r, box.b)):
        lo, hi = lo + tolerance, hi - tolerance
        if lo >= hi:
            return False
        delta = end - start
        if abs(delta) < 1e-12:
            if not lo < start < hi:
                return False
            continue
        near, far = sorted(((lo - start) / delta, (hi - start) / delta))
        lower, upper = max(lower, near), min(upper, far)
        if lower >= upper:
            return False
    return lower < upper


def regions(snapshot: Snapshot) -> list[Region]:
    found: dict[int, Region] = {}
    pending = list(snapshot.layout)
    while pending:
        region = pending.pop()
        if region.id not in found:
            found[region.id] = region
            pending.extend(region.children)
    return sorted(found.values(), key=lambda r: r.id)


def scope_of(bbox: BoundingBox, snapshot: Snapshot) -> Scope:
    tables = [
        r
        for r in regions(snapshot)
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
        values.append(Value(native, bbox, scope_of(bbox, snapshot)))

    # Atom identity is source-backed and shared by all overlapping span choices.
    atoms: dict[tuple, int] = {}
    labels: dict[tuple[frozenset[int], Scope], Label] = {}
    heights = []
    for region in regions(snapshot):
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
            scope = scope_of(box, snapshot)
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
                if scope_of(bbox, snapshot) != scope:
                    continue
                labels[ids, scope] = Label(
                    " ".join(c[1] for c in bundle),
                    bbox,
                    ids,
                    scope,
                    region.label,
                    len(bundle) < len(cells),
                )
    return values, list(labels.values()), statistics.median(heights) if heights else 1.0


def local_features(
    label: Label, members: tuple[int, ...], values: list[Value], h: float
) -> dict[str, float]:
    distance, misalignment, obstacles = 0.0, 0.0, 0.0
    for i in members:
        value = values[i]
        a, b = label.bbox, value.bbox
        distance += math.log1p(gap(a, b) / h)
        xover = max(0.0, min(a.r, b.r) - max(a.l, b.l)) / max(
            1e-6, min(a.width, b.width)
        )
        yover = max(0.0, min(a.b, b.b) - max(a.t, b.t)) / max(
            1e-6, min(a.height, b.height)
        )
        misalignment += 0.5 * (1 - max(xover, yover))
        start, end = anchors(a, b)
        obstacles += 4 * sum(
            obstructs(start, end, other.bbox, h * 0.05)
            for j, other in enumerate(values)
            if j not in members
        )
    role = 3.0 * len(members) * (not any(c.isalpha() for c in label.text))
    role += (
        3.0 * len(members) * (label.role in {"page_header", "page_footer", "footnote"})
    )
    # A short printed component between two text boxes is weak key evidence.
    # The rule uses position and length, never a fixture/token blacklist.
    component = (
        len(label.text.strip()) <= 3
        and any(
            not v.checkbox and v.bbox.r <= label.bbox.l and gap(v.bbox, label.bbox) < h
            for v in values
        )
        and any(
            not v.checkbox and v.bbox.l >= label.bbox.r and gap(v.bbox, label.bbox) < h
            for v in values
        )
    )
    role += 3.0 * len(members) * component
    return {
        "distance": distance,
        "alignment": misalignment,
        "obstruction": obstacles,
        "role": role,
        "fragment": 0.5 * len(members) * label.fragment,
    }


def candidates_for(
    values: list[Value], labels: list[Label], h: float
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for li, label in enumerate(labels):
        eligible = [
            i
            for i, v in enumerate(values)
            if v.scope.eligible and v.scope == label.scope
        ]
        contained = tuple(
            i
            for i in eligible
            if values[i].bbox.intersection_over_self(label.bbox)
            >= INLINE_WIDGET_COVERAGE
        )
        for i in eligible:
            kind = "option_caption" if values[i].checkbox else "field_key"
            candidates.append(
                Candidate((i,), li, kind, local_features(label, (i,), values, h))
            )
        if len(contained) > 1:
            candidates.append(
                Candidate(
                    contained,
                    li,
                    "inline_clause",
                    local_features(label, contained, values, h),
                )
            )
        # A common caption over adjacent components, with no full intervening
        # caption. Do not make arbitrary runs of checkboxes into one field.
        text_values = [i for i in eligible if not values[i].checkbox]
        for start in range(len(text_values)):
            members = [text_values[start]]
            for nxt in text_values[start + 1 :]:
                prev, curr = values[members[-1]].bbox, values[nxt].bbox
                if abs(prev.t - curr.t) > h or gap(prev, curr) > 2 * h:
                    break
                members.append(nxt)
                union = BoundingBox.enclosing_bbox([values[i].bbox for i in members])
                if (
                    label.bbox.b <= union.t
                    and union.t - label.bbox.b <= 5 * h
                    and overlap(
                        BoundingBox(
                            l=union.l, r=union.r, t=label.bbox.t, b=label.bbox.b
                        ),
                        label.bbox,
                    )
                    > 0
                ):
                    sibling_caption = any(
                        other.scope == label.scope
                        and other.atoms.isdisjoint(label.atoms)
                        and other.bbox.b <= curr.t
                        and other.bbox.t >= label.bbox.t - h
                        and other.bbox.l >= curr.l
                        and other.bbox.r <= curr.r
                        for other in labels
                    )
                    if sibling_caption:
                        break
                    features = local_features(label, tuple(members), values, h)
                    # The caption describes the composite envelope. Charge its
                    # distance once PER VALUE, so a big group is not a free link.
                    features["distance"] = len(members) * math.log1p(
                        gap(label.bbox, union) / h
                    )
                    features["alignment"] = 0.0
                    features["group"] = 0.5
                    candidates.append(
                        Candidate(tuple(members), li, "composite_field", features)
                    )

    candidates.extend(choice_configurations(values, labels, h))
    return candidates

# ruff: noqa: F821
# Names are supplied by insertion into the frozen prototype; see build_experiment.py.
"""Geometric question-and-option hypotheses for an isolated experiment.

This file is inserted into a frozen prototype by build_experiment.py. It reuses
that prototype's dataclasses and geometry; it never reads annotation text.
"""


def choice_configurations(values, labels, h):
    configurations = []
    buttons = [i for i, v in enumerate(values) if v.checkbox and v.scope.eligible]
    for qi, question in enumerate(labels):
        if question.role in {"page_header", "page_footer", "footnote"}:
            continue
        eligible = [i for i in buttons if values[i].scope == question.scope]
        for side in ("right", "left"):
            captions = {}
            offsets = {}
            for i in eligible:
                widget = values[i].bbox
                alternatives = []
                for li, caption in enumerate(labels):
                    if caption.scope != question.scope or not caption.atoms.isdisjoint(
                        question.atoms
                    ):
                        continue
                    rect = caption.bbox
                    if abs(rect.t - widget.t) > h:
                        continue
                    inlined = (
                        widget.intersection_over_self(rect) >= INLINE_WIDGET_COVERAGE
                    )
                    if (
                        inlined
                        and sum(
                            values[j].bbox.intersection_over_self(rect)
                            >= INLINE_WIDGET_COVERAGE
                            for j in eligible
                        )
                        != 1
                    ):
                        continue
                    if side == "right":
                        fits = rect.l >= widget.r - 0.25 * h or (
                            inlined and rect.r > widget.r + h
                        )
                        offset = max(0, rect.l - widget.r)
                    else:
                        fits = rect.r <= widget.l + 0.25 * h or (
                            inlined and rect.l < widget.l - h
                        )
                        offset = max(0, widget.l - rect.r)
                    if not fits or offset > 4 * h:
                        continue
                    # Geometry and existing span boundaries only: no word count,
                    # character class, token identity, or native field name.
                    score = (
                        math.log1p(offset / h)
                        + abs(rect.t - widget.t) / h
                        + 0.5 * caption.fragment
                    )
                    alternatives.append((score, li, offset / h))
                if alternatives:
                    _, captions[i], offsets[i] = min(alternatives)
            for start, first in enumerate(eligible):
                if first not in captions:
                    continue
                first_box = values[first].bbox
                q = question.bbox
                above = q.b <= first_box.t and first_box.t - q.b <= 3 * h
                beside = (
                    q.r <= first_box.l and q.t <= first_box.b and q.b >= first_box.t
                )
                if not (above or beside):
                    continue
                members, option_labels, used_atoms = [], [], set(question.atoms)
                boundaries = 0
                for i in eligible[start:]:
                    if i not in captions:
                        break
                    widget = values[i].bbox
                    caption = labels[captions[i]]
                    if used_atoms.intersection(caption.atoms):
                        break
                    if members:
                        prev = values[members[-1]].bbox
                        column = abs(widget.l - prev.l) <= h and widget.t >= prev.b
                        row = abs(widget.t - prev.t) <= h and widget.l >= prev.r
                        reset = (
                            widget.l > prev.l + h
                            and abs(widget.t - first_box.t) <= h
                            and widget.t < prev.t - h
                        )
                        if not (column or row or reset):
                            break
                        # Keep both a boundary and a continuation hypothesis
                        # at intervening blocks, including side-by-side prompts.
                        bottom = max(prev.b, labels[option_labels[-1]].bbox.b)
                        barrier = column and any(
                            other.atoms.isdisjoint(used_atoms | set(caption.atoms))
                            and other.scope == question.scope
                            and other.bbox.t >= bottom
                            and (
                                (
                                    other.bbox.b <= widget.t
                                    and other.bbox.l <= widget.l + 0.25 * h
                                    and other.bbox.r >= widget.r
                                )
                                or (
                                    other.bbox.t < widget.b
                                    and other.bbox.b > widget.t
                                    and other.bbox.width >= 2 * h
                                    and (
                                        (side == "right" and other.bbox.r <= widget.l)
                                        or (side == "left" and other.bbox.l >= widget.r)
                                    )
                                )
                            )
                            for other in labels
                        )
                        boundaries += bool(barrier)
                    members.append(i)
                    option_labels.append(captions[i])
                    used_atoms.update(caption.atoms)
                    if len(members) < 2:
                        continue
                    envelope = BoundingBox.enclosing_bbox(
                        [values[j].bbox for j in members]
                        + [labels[j].bbox for j in option_labels]
                    )
                    if above and min(q.r, envelope.r) <= max(q.l, envelope.l):
                        continue
                    # Reward supported repetition, not arbitrary nearby headings.
                    # An ungrouped solution pays neither this reward nor any
                    # within-group consistency cost.
                    distance = (
                        (first_box.t - q.b) / h
                        if above
                        else (first_box.l - q.r)
                        / max(h, first_box.l - q.l + envelope.width)
                    )
                    consistency = sum(
                        abs(offsets[j] - offsets[first]) for j in members
                    ) / len(members)
                    # A block directly captioning a different checkbox is not
                    # a free-standing prompt for this group. Nested questions
                    # sharing their option caption need a separate hierarchy.
                    if any(
                        j not in members
                        and abs(values[j].bbox.t - q.t) <= h
                        and (
                            (q.l >= values[j].bbox.r and q.l - values[j].bbox.r <= h)
                            or (q.r <= values[j].bbox.l and values[j].bbox.l - q.r <= h)
                        )
                        for j in eligible
                    ):
                        continue
                    features = {
                        "question_gap": math.log1p(max(0, distance)),
                        "caption_consistency": consistency,
                        "supported_repetition": -2.0
                        * (len(members) - 1)
                        / len(members),
                        "intervening_blocks": 0.25 * boundaries,
                        "prompt_fragment": 0.5 * question.fragment,
                    }
                    configurations.append(
                        Candidate(
                            tuple(members),
                            qi,
                            "choice_group",
                            features,
                            tuple(option_labels),
                        )
                    )
    # Within one uninterrupted block, do not invent a cut between otherwise
    # compatible options. Keep cuts at intervening blocks as alternatives.
    return [
        c
        for c in configurations
        if not any(
            other.label == c.label
            and other.features["intervening_blocks"] == c.features["intervening_blocks"]
            and len(other.members) > len(c.members)
            and set(zip(c.members, c.option_labels)).issubset(
                zip(other.members, other.option_labels)
            )
            for other in configurations
        )
    ]


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


def assign(
    snapshot: Snapshot, *, null_cost: float = 3.0, time_limit: float = 10.0, forbidden_groups: frozenset[int] = frozenset()
) -> Assignment:
    if (
        not math.isfinite(null_cost)
        or null_cost <= 0
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("Costs and time limits must be positive and finite")
    values, labels, h = inputs(snapshot)
    proposed = candidates_for(values, labels, h)
    # A candidate worse than leaving all its members blank cannot help: all
    # remaining interactions are penalties. Keep the full list for diagnostics.
    active = [
        i
        for i, c in enumerate(proposed)
        if i not in forbidden_groups and c.cost < (0 if c.kind == "choice_group" else null_cost * len(c.members))
    ]
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
    for coefficients in [*questions.values(), *spans.values()]:
        constraint(coefficients, 0, 1)
    # Selecting a question configuration requires its specified local captions.
    # All ordinary primary assignments and abstention remain available without it.
    for column, index in enumerate(active):
        candidate = proposed[index]
        if not candidate.option_labels:
            continue
        for member, caption in zip(candidate.members, candidate.option_labels):
            compatible = {
                other_column: -1.0
                for other_column, other_index in enumerate(active)
                if proposed[other_index].kind != "choice_group"
                and member in proposed[other_index].members
                and proposed[other_index].label == caption
            }
            constraint({column: 1.0, **compatible}, -math.inf, 0)
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
    if not costs:
        return Assignment(values, labels, proposed, [], "optimal", 0)
    rr, cc, data = [], [], []
    for r, row in enumerate(rows):
        for c, coefficient in row.items():
            rr.append(r)
            cc.append(c)
            data.append(coefficient)
    matrix = coo_matrix((data, (rr, cc)), shape=(len(rows), len(costs))).tocsc()
    result = milp(
        np.array(costs),
        integrality=np.ones(len(costs)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if result.status != 0:
        # This first experiment abstains on timeout; no unsupported confidence
        # claims about a partial solution and no missing native values.
        return Assignment(
            values,
            labels,
            proposed,
            [],
            str(result.message),
            null_cost * sum(v.scope.eligible for v in values),
        )
    selected = [index for column, index in enumerate(active) if result.x[column] > 0.5]
    return Assignment(values, labels, proposed, selected, "optimal", float(result.fun))
