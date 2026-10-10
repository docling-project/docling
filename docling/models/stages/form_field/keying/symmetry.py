# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Symmetry of repeated fields: alternating runs and one side per structure.

Both rules read the geometry of like fields (same kind, similar size) laid out
along rows and columns, and both only add costs: strong local evidence can
still win. The weights are set from the other costs, not fitted to data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from docling_core.types.doc import BoundingBox

from docling.models.stages.form_field.keying.candidates import siblings
from docling.models.stages.form_field.keying.geometry import gap, lettered, overlap
from docling.models.stages.form_field.keying.types import (
    Candidate,
    KeyText,
    Side,
    Value,
)

# Above the usual gap between a key before and one after a value (a cell
# difference, 1) and below leaving a value blank (3): a decided run turns
# close calls, and never forces a value to stay unkeyed.
SEQUENCE_COST = 1.5
# One cell difference: a single field breaking a repeated structure's side
# must have clearly better local evidence than its siblings.
SIDE_COST = 1.0

Axis = Literal["row", "column"]
# The side that faces the next field along the axis, then the previous one.
FACING: dict[Axis, tuple[Side, Side]] = {
    "row": ("right", "left"),
    "column": ("down", "up"),
}


@dataclass(frozen=True)
class Reach:
    """The first text a value sees on one side: its atoms and nearest gap."""

    atoms: frozenset[int]
    gap: float


def reaches(
    candidates: list[Candidate],
    key_texts: list[KeyText],
    values: list[Value],
    null_cost: float,
) -> dict[tuple[int, Side], Reach]:
    """What every value sees on each side, from its plausible single keys.

    Text too costly to key a value on its own (a page header, a code) is not a
    key, so it never shapes a run or a structure.
    """
    atoms: dict[tuple[int, Side], set[int]] = defaultdict(set)
    gaps: dict[tuple[int, Side], float] = {}
    for c in candidates:
        if (
            c.side is None
            or c.side == "inside"
            or len(c.members) != 1
            or c.cost >= null_cost
        ):
            continue
        key = (c.members[0], c.side)
        key_text = key_texts[c.key]
        atoms[key] |= key_text.atoms
        distance = gap(key_text.bbox, values[c.members[0]].bbox)
        gaps[key] = min(gaps.get(key, distance), distance)
    return {key: Reach(frozenset(atoms[key]), gaps[key]) for key in atoms}


def _between(a: BoundingBox, b: BoundingBox, axis: Axis) -> BoundingBox:
    """The corridor between two aligned boxes, a before b along the axis."""
    if axis == "row":
        return BoundingBox(l=a.r, t=max(a.t, b.t), r=b.l, b=min(a.b, b.b))
    return BoundingBox(l=max(a.l, b.l), t=a.b, r=min(a.r, b.r), b=b.t)


def neighbours(values: list[Value], axis: Axis) -> dict[int, int]:
    """Each value's next like field along a row or column, when mutual.

    The next field is the nearest sibling after it on the axis with no other
    field in between; a pair counts only when each is the other's nearest.
    """
    side: Side = "left" if axis == "row" else "up"

    def start(v: Value) -> float:
        return v.bbox.l if axis == "row" else v.bbox.t

    def end(v: Value) -> float:
        return v.bbox.r if axis == "row" else v.bbox.b

    def nearest(i: int, forward: bool) -> int | None:
        found = [
            j
            for j, v in enumerate(values)
            if j != i
            and v.scope == values[i].scope
            and siblings(values[i], v, side)
            and (
                start(v) >= end(values[i]) - 1e-6
                if forward
                else end(v) <= start(values[i]) + 1e-6
            )
        ]
        if not found:
            return None
        j = (
            min(found, key=lambda k: (start(values[k]), k))
            if forward
            else max(found, key=lambda k: (end(values[k]), -k))
        )
        a, b = (i, j) if forward else (j, i)
        corridor = _between(values[a].bbox, values[b].bbox, axis)
        if any(
            k not in (a, b) and overlap(v.bbox, corridor) > 0
            for k, v in enumerate(values)
        ):
            return None
        return j

    pairs = {}
    for i, v in enumerate(values):
        if not v.scope.eligible:
            continue
        j = nearest(i, True)
        if j is not None and nearest(j, False) == i:
            pairs[i] = j
    return pairs


def _texts_between(
    a: Value, b: Value, axis: Axis, key_texts: list[KeyText]
) -> frozenset[int]:
    """Lettered text atoms whose centre lies in the corridor between a and b."""
    corridor = _between(a.bbox, b.bbox, axis)
    found = set()
    for key_text in key_texts:
        if len(key_text.atoms) != 1 or not lettered(key_text.text):
            continue
        x = (key_text.bbox.l + key_text.bbox.r) / 2
        y = (key_text.bbox.t + key_text.bbox.b) / 2
        if corridor.l <= x <= corridor.r and corridor.t <= y <= corridor.b:
            found |= key_text.atoms
    return frozenset(found)


def _leftover(
    run: list[int],
    reach: dict[tuple[int, Side], Reach],
    axis: Axis,
    values: list[Value],
    h: float,
) -> Side | None:
    """The key side of a run with text at both ends, when two signs agree.

    One text is left over: a shared prompt before the fields, or the start of the
    next field after them. Neither sign alone settles it, so both must agree:
    the inner texts all sitting nearer the field before them (keys after)
    or after them (keys before), and the layout convention of the fields.
    Along a row, options read the key after them and text boxes the one
    before; down a column, text boxes read the key above and options have
    no convention. An inner text within half a text line of both its fields
    touches both, and says nothing about which one it keys.
    """
    after, before = FACING[axis]
    pairs = list(pairwise(run))
    if any(
        reach[i, after].gap <= 0.5 * h and reach[j, before].gap <= 0.5 * h
        for i, j in pairs
    ):
        return None
    if all(reach[i, after].gap < reach[j, before].gap for i, j in pairs):
        lean = after
    elif all(reach[i, after].gap > reach[j, before].gap for i, j in pairs):
        lean = before
    else:
        return None
    checkbox = values[run[0]].checkbox
    if axis == "row":
        convention: Side | None = after if checkbox else before
    else:
        convention = None if checkbox else before
    return lean if lean == convention else None


def sequences(
    values: list[Value],
    key_texts: list[KeyText],
    candidates: list[Candidate],
    null_cost: float,
    h: float,
) -> dict[int, Side]:
    """The side each value of a decided alternating run reads its key from.

    A run is a line of like fields with exactly one text between each two: the
    text after one field is the text before the next. It is a key line
    only when every field's best key lies along it: the run settles which
    text of the line each field reads, never which axis (an option key
    beside a checkbox in a column, or a line key far to the left of an
    amount, pairs the field across the run). The ends fix the pairing: a key before
    the first field and none after the last means keys before (left,
    above); the reverse means keys after. With text at both ends one text
    is left over (see _leftover).
    """
    reach = reaches(candidates, key_texts, values, null_cost)
    best: dict[int, float] = {}
    best_on: dict[tuple[int, Side], float] = {}
    for c in candidates:
        if c.side is not None and len(c.members) == 1 and c.cost < null_cost:
            i = c.members[0]
            best[i] = min(best.get(i, c.cost), c.cost)
            best_on[i, c.side] = min(best_on.get((i, c.side), c.cost), c.cost)
    decided: dict[int, Side] = {}
    conflicts: set[int] = set()
    for axis in ("row", "column"):
        after, before = FACING[axis]
        pairs = neighbours(values, axis)
        links = {}
        for i, j in pairs.items():
            forward, backward = reach.get((i, after)), reach.get((j, before))
            if forward is None or backward is None:
                continue
            shared = forward.atoms & backward.atoms
            between = _texts_between(values[i], values[j], axis, key_texts)
            if shared and between <= shared:
                links[i] = j
        for first in sorted(set(links) - set(links.values())):
            run = [first]
            while run[-1] in links:
                run.append(links[run[-1]])
            if any(
                min(best_on.get((i, side), math.inf) for side in (after, before))
                > best[i]
                for i in run
            ):
                continue
            lead, trail = (run[0], before) in reach, (run[-1], after) in reach
            if lead and not trail:
                side = before
            elif trail and not lead:
                side = after
            elif lead and trail and _leftover(run, reach, axis, values, h):
                side = _leftover(run, reach, axis, values, h)
            else:
                continue
            for i in run:
                if decided.get(i, side) != side:
                    conflicts.add(i)
                decided[i] = side
    return {i: side for i, side in decided.items() if i not in conflicts}


def mark_sequences(candidates: list[Candidate], sides: dict[int, Side]) -> None:
    """Charge every association of a run's value but its key on the run's side."""
    for c in candidates:
        if c.kind == "choice_group":
            continue
        off = sum(
            1
            for i in c.members
            if i in sides and not (len(c.members) == 1 and c.side == sides[i])
        )
        if off:
            c.features["sequence"] = SEQUENCE_COST * off


def structures(
    values: list[Value],
    key_texts: list[KeyText],
    candidates: list[Candidate],
    null_cost: float,
) -> list[list[int]]:
    """Groups of at least three like fields repeated along rows and columns.

    Two neighbouring like fields belong together when the only text between
    them is what they themselves see (their own keys): a third text in
    between starts another part of the form.
    """
    reach = reaches(candidates, key_texts, values, null_cost)
    parent = list(range(len(values)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for axis in ("row", "column"):
        after, before = FACING[axis]
        for i, j in neighbours(values, axis).items():
            own = frozenset()
            if (i, after) in reach:
                own |= reach[i, after].atoms
            if (j, before) in reach:
                own |= reach[j, before].atoms
            if _texts_between(values[i], values[j], axis, key_texts) <= own:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, v in enumerate(values):
        if v.scope.eligible:
            groups[root(i)].append(i)
    return [group for group in groups.values() if len(group) >= 3]
