# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Candidate label-value associations and their costs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from docling_core.types.doc import BoundingBox

from docling.models.stages.form_field.keying.geometry import (
    anchors,
    corridor,
    gap,
    lettered,
    obstructs,
    overlap,
    side_of,
    span_overlap,
)
from docling.models.stages.form_field.keying.types import (
    INLINE_WIDGET_COVERAGE,
    Candidate,
    Label,
    Scope,
    Side,
    Value,
)


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
    return {
        "distance": distance,
        "alignment": misalignment,
        "obstruction": obstacles,
        "role": len(members) * role_cost(label, values, h),
        "fragment": 0.5 * len(members) * label.fragment,
    }


def role_cost(label: Label, values: list[Value], h: float) -> float:
    role = 3.0 * (not lettered(label.text))
    role += 3.0 * (label.role in {"page_header", "page_footer", "footnote"})
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
    return role + 3.0 * component


def slot_features(
    label: Label,
    value: Value,
    side: Side,
    alignment: float,
    values: list[Value],
    h: float,
) -> dict[str, float]:
    """Cost of one visible label for one value: cell, then alignment, then distance.

    A caption in the value's own printed cell (inside, or touching it above or
    below) is preferred over an aligned caption along the same line, which is
    preferred over an aligned caption farther up or down. Captions precede
    their value in reading order: a caption below costs a little more than one
    above (in stacked boxes the next box's caption touches from below), and
    text after the value on its line only captions it when adjacent, as an
    option caption follows its checkbox; farther right it starts the next cell.
    Sharing half of the band counts as aligned; below that, misalignment grows
    to the cost of a cell difference. Proximity only breaks ties.
    """
    gap_h = gap(label.bbox, value.bbox) / h
    if side == "inside" or (side in ("up", "down") and gap_h <= 0.5):
        cell = 0.0
    else:
        cell = 1.0 if side in ("left", "right") else 1.5
    after = 0.5 * (side == "down") + 1.0 * (side == "right" and gap_h > 1.0)
    return {
        "cell": cell + after,
        "misaligned": max(0.0, 1.0 - alignment / 0.5),
        "distance": 0.25 * min(math.log1p(gap_h), 2.0),
        "role": role_cost(label, values, h),
        "fragment": 0.5 * label.fragment,
    }


def blockers(labels: list[Label]) -> Callable[[Label, BoundingBox | None], bool]:
    """Whether other lettered text lies in a corridor, hiding a label behind it.

    Only the first text met on each side of a value is a candidate: the value's
    own cell, or the neighbouring cell on that side. Digits, codes and symbols
    are transparent, as are other values: line numbers, arithmetic signs and
    sibling fields sit between many captions and their values.
    """
    atoms = {
        next(iter(label.atoms)): label.bbox
        for label in labels
        if len(label.atoms) == 1 and lettered(label.text)
    }
    ids = np.array(list(atoms), dtype=int)
    boxes = np.array([[b.l, b.t, b.r, b.b] for b in atoms.values()]).reshape(-1, 4)
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    def blocked(label: Label, lane: BoundingBox | None) -> bool:
        if lane is None or not len(ids):
            return False
        width = np.minimum(boxes[:, 2], lane.r) - np.maximum(boxes[:, 0], lane.l)
        height = np.minimum(boxes[:, 3], lane.b) - np.maximum(boxes[:, 1], lane.t)
        # Text inside the corridor, or a line crossing it: the overlap covers
        # half of the smaller extent along both axes.
        across = (width >= 0.5 * np.minimum(widths, lane.width)) & (width > 0)
        across &= (height >= 0.5 * np.minimum(heights, lane.height)) & (height > 0)
        return bool(np.any(across & ~np.isin(ids, list(label.atoms))))

    return blocked


def foreign(i: int, lane: BoundingBox | None, side: Side, values: list[Value]) -> bool:
    """Whether a different kind of field sits in a column corridor.

    Walking up or down a column passes only the value's siblings (a repeated
    column of like fields under its header); another field there is another
    cell, with its own caption. Rows are different: operand boxes and flags
    legitimately sit between a line caption and its amount.
    """
    if lane is None or side not in ("up", "down"):
        return False
    return any(
        j != i
        and span_overlap(v.bbox.l, v.bbox.r, lane.l, lane.r)
        >= 0.5 * min(v.bbox.width, lane.width)
        and span_overlap(v.bbox.t, v.bbox.b, lane.t, lane.b)
        >= 0.5 * min(v.bbox.height, lane.height)
        and not siblings(values[i], v, side)
        for j, v in enumerate(values)
    )


def candidates_for(
    values: list[Value], labels: list[Label], h: float
) -> list[Candidate]:
    candidates: list[Candidate] = []
    blocked = blockers(labels)
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
            facing = side_of(label.bbox, values[i].bbox)
            if facing is None or not lettered(label.text):
                continue
            lane = corridor(label.bbox, values[i].bbox, facing[0])
            if blocked(label, lane) or foreign(i, lane, facing[0], values):
                continue
            kind = "option_caption" if values[i].checkbox else "field_key"
            features = slot_features(label, values[i], *facing, values, h)
            candidates.append(Candidate((i,), li, kind, features, facing[0]))
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
                    # Any member with its own caption above it is a separate field.
                    sibling_caption = any(
                        other.scope == label.scope
                        and other.atoms.isdisjoint(label.atoms)
                        and lettered(other.text)
                        and other.bbox.b <= member.t
                        and other.bbox.t >= label.bbox.t - h
                        and span_overlap(other.bbox.l, other.bbox.r, member.l, member.r)
                        >= 0.5 * other.bbox.width
                        for other in labels
                        for member in (values[i].bbox for i in members)
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

    # Shared PDF names are corroboration, not an assertion of field identity.
    # Also retain contiguous native checkbox runs as alternatives when captions
    # form a repeated, compact arrangement (including a column reset).
    groups: set[tuple[int, ...]] = set()
    names: dict[tuple[Scope, str], list[int]] = defaultdict(list)
    run: list[int] = []
    for i, value in enumerate(values):
        if value.checkbox and value.scope.eligible:
            column_reset = bool(run) and value.bbox.t <= values[run[0]].bbox.t + h
            if run and (
                value.scope != values[run[-1]].scope
                or (gap(value.bbox, values[run[-1]].bbox) > 4 * h and not column_reset)
            ):
                if len(run) > 1:
                    groups.add(tuple(run))
                run = []
            run.append(i)
            if value.native.widget_field_name:
                names[value.scope, value.native.widget_field_name].append(i)
        else:
            if len(run) > 1:
                groups.add(tuple(run))
            run = []
    if len(run) > 1:
        groups.add(tuple(run))
    groups.update(tuple(g) for g in names.values() if len(g) > 1)
    for members in sorted(groups):
        scope = values[members[0]].scope
        union = BoundingBox.enclosing_bbox([values[i].bbox for i in members])
        for li, label in enumerate(labels):
            if label.scope != scope or not any(c.isalpha() for c in label.text):
                continue
            # Common prompts sit above the group or alongside its first option;
            # individual short option captions are not plausible common prompts.
            first_box = values[members[0]].bbox
            above = label.bbox.b <= union.t - 0.2 * h
            beside = label.bbox.r <= first_box.l - h and label.bbox.t <= first_box.b
            if not (above or beside) or gap(label.bbox, first_box) > 10 * h:
                continue
            if len(label.text.split()) < 2:
                continue
            features = {
                "question_distance": math.log1p(gap(label.bbox, union) / h),
                "question_prior": -2.5,
                "fragment": 0.5 * label.fragment,
            }
            candidates.append(Candidate(members, li, "choice_group", features))
    # Within a caption, the whole text is the key, not one of its lines: a
    # line seen from the same side as its whole stacked caption pays extra.
    whole: dict[tuple[tuple[int, ...], Side], list[frozenset[int]]] = defaultdict(list)
    for c in candidates:
        if c.side is not None and labels[c.label].stack:
            whole[c.members, c.side].append(labels[c.label].atoms)
    for c in candidates:
        if c.side is not None and any(
            labels[c.label].atoms < atoms for atoms in whole[c.members, c.side]
        ):
            c.features["partial"] = 0.5
    return candidates


def siblings(a: Value, b: Value, side: Side) -> bool:
    """Same kind and size, aligned along the line a shared caption runs along.

    A row caption keys every like-sized value of its row, a column caption
    every like-sized value of its column. Operand boxes of a different size in
    the same row (multipliers, rates) do not inherit the row caption.
    """
    if side == "inside" or a.checkbox != b.checkbox:
        return False
    x, y = a.bbox, b.bbox
    if min(x.width, y.width) < 0.8 * max(x.width, y.width):
        return False
    if min(x.height, y.height) < 0.8 * max(x.height, y.height):
        return False
    if side in ("left", "right"):
        return span_overlap(x.t, x.b, y.t, y.b) >= 0.5 * min(x.height, y.height)
    return span_overlap(x.l, x.r, y.l, y.r) >= 0.5 * min(x.width, y.width)
