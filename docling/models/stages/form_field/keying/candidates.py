# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Candidate key-value associations and their costs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from docling_core.types.doc import BoundingBox, DocItemLabel

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
    WIDGET_COVERAGE,
    Candidate,
    KeyText,
    Scope,
    Side,
    Value,
)


def local_features(
    key_text: KeyText, members: tuple[int, ...], values: list[Value], h: float
) -> dict[str, float]:
    distance, misalignment, obstacles = 0.0, 0.0, 0.0
    for i in members:
        value = values[i]
        a, b = key_text.bbox, value.bbox
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
        "role": len(members) * role_cost(key_text, values, h),
        "fragment": 0.5 * len(members) * key_text.fragment,
    }


def role_cost(key_text: KeyText, values: list[Value], h: float) -> float:
    role = 3.0 * (not lettered(key_text.text))
    role += 3.0 * (
        key_text.layout_label
        in {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}
    )
    # A short printed component between two text boxes is weak key evidence.
    # The rule uses position and length, never a fixture/token blacklist.
    component = (
        len(key_text.text.strip()) <= 3
        and any(
            not v.checkbox
            and v.bbox.r <= key_text.bbox.l
            and gap(v.bbox, key_text.bbox) < h
            for v in values
        )
        and any(
            not v.checkbox
            and v.bbox.l >= key_text.bbox.r
            and gap(v.bbox, key_text.bbox) < h
            for v in values
        )
    )
    return role + 3.0 * component


def slot_features(
    key_text: KeyText,
    value: Value,
    side: Side,
    alignment: float,
    values: list[Value],
    h: float,
) -> dict[str, float]:
    """Cost of one candidate key for one value: cell, then alignment, then distance.

    A key in the value's own printed cell (inside, or touching it above or
    below) is preferred over an aligned key along the same line, which is
    preferred over an aligned key farther up or down. Keys precede
    their value in reading order: a key below costs a little more than one
    above (in stacked boxes the next box's key touches from below), and
    text after the value on its line only keys it when adjacent, as an
    option key follows its checkbox; farther right it starts the next cell.
    Sharing half of the band counts as aligned; below that, misalignment grows
    to the cost of a cell difference. Proximity only breaks ties.
    """
    gap_h = gap(key_text.bbox, value.bbox) / h
    if side == "inside" or (side in ("up", "down") and gap_h <= 0.5):
        cell = 0.0
    else:
        cell = 1.0 if side in ("left", "right") else 1.5
    after = 0.5 * (side == "down") + 1.0 * (side == "right" and gap_h > 1.0)
    return {
        "cell": cell + after,
        "misaligned": max(0.0, 1.0 - alignment / 0.5),
        "distance": 0.25 * min(math.log1p(gap_h), 2.0),
        "role": role_cost(key_text, values, h),
        "fragment": 0.5 * key_text.fragment,
    }


def blockers(key_texts: list[KeyText]) -> Callable[[KeyText, BoundingBox | None], bool]:
    """Whether other lettered text lies in a corridor, hiding a key behind it.

    Only the first text met on each side of a value is a candidate: the value's
    own cell, or the neighbouring cell on that side. Digits, codes and symbols
    are transparent, as are other values: line numbers, arithmetic signs and
    sibling fields sit between many keys and their values.
    """
    atoms = {
        next(iter(key_text.atoms)): key_text.bbox
        for key_text in key_texts
        if len(key_text.atoms) == 1 and lettered(key_text.text)
    }
    ids = np.array(list(atoms), dtype=int)
    boxes = np.array([[b.l, b.t, b.r, b.b] for b in atoms.values()]).reshape(-1, 4)
    widths, heights = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]

    def blocked(key_text: KeyText, lane: BoundingBox | None) -> bool:
        if lane is None or not len(ids):
            return False
        width = np.minimum(boxes[:, 2], lane.r) - np.maximum(boxes[:, 0], lane.l)
        height = np.minimum(boxes[:, 3], lane.b) - np.maximum(boxes[:, 1], lane.t)
        # Text inside the corridor, or a line crossing it: the overlap covers
        # half of the smaller extent along both axes.
        across = (width >= 0.5 * np.minimum(widths, lane.width)) & (width > 0)
        across &= (height >= 0.5 * np.minimum(heights, lane.height)) & (height > 0)
        return bool(np.any(across & ~np.isin(ids, list(key_text.atoms))))

    return blocked


def foreign(i: int, lane: BoundingBox | None, side: Side, values: list[Value]) -> bool:
    """Whether a different kind of field sits in a column corridor.

    Walking up or down a column passes only the value's siblings (a repeated
    column of like fields under its header); another field there is another
    cell, with its own key. Rows are different: operand boxes and flags
    legitimately sit between a line key and its amount.
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
    values: list[Value], key_texts: list[KeyText], h: float
) -> list[Candidate]:
    candidates: list[Candidate] = []
    blocked = blockers(key_texts)
    for li, key_text in enumerate(key_texts):
        eligible = [
            i
            for i, v in enumerate(values)
            if v.scope.eligible and v.scope == key_text.scope
        ]
        contained = tuple(
            i
            for i in eligible
            if values[i].bbox.intersection_over_self(key_text.bbox) >= WIDGET_COVERAGE
        )
        for i in eligible:
            facing = side_of(key_text.bbox, values[i].bbox)
            if facing is None or not lettered(key_text.text):
                continue
            lane = corridor(key_text.bbox, values[i].bbox, facing[0])
            if blocked(key_text, lane) or foreign(i, lane, facing[0], values):
                continue
            kind = "option_key" if values[i].checkbox else "field_key"
            features = slot_features(key_text, values[i], *facing, values, h)
            candidates.append(Candidate((i,), li, kind, features, facing[0]))
        clause = tuple(
            i
            for i in contained
            if i not in answers(contained, key_text, key_texts, values, h)
        )
        if len(clause) > 1:
            candidates.append(
                Candidate(
                    clause,
                    li,
                    "inline_clause",
                    local_features(key_text, clause, values, h),
                )
            )
        # A common key over adjacent components, with no full intervening
        # key. Do not make arbitrary runs of checkboxes into one field.
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
                    key_text.bbox.b <= union.t
                    and union.t - key_text.bbox.b <= 5 * h
                    and overlap(
                        BoundingBox(
                            l=union.l, r=union.r, t=key_text.bbox.t, b=key_text.bbox.b
                        ),
                        key_text.bbox,
                    )
                    > 0
                ):
                    # Any member with its own key above it is a separate field.
                    sibling_key = any(
                        other.scope == key_text.scope
                        and other.atoms.isdisjoint(key_text.atoms)
                        and lettered(other.text)
                        and other.bbox.b <= member.t
                        and other.bbox.t >= key_text.bbox.t - h
                        and span_overlap(other.bbox.l, other.bbox.r, member.l, member.r)
                        >= 0.5 * other.bbox.width
                        for other in key_texts
                        for member in (values[i].bbox for i in members)
                    )
                    if sibling_key:
                        break
                    features = local_features(key_text, tuple(members), values, h)
                    # The key describes the composite envelope. Charge its
                    # distance once PER VALUE, so a big group is not a free link.
                    features["distance"] = len(members) * math.log1p(
                        gap(key_text.bbox, union) / h
                    )
                    features["alignment"] = 0.0
                    features["group"] = 0.5
                    candidates.append(
                        Candidate(tuple(members), li, "composite_field", features)
                    )

    # Shared PDF names are corroboration, not an assertion of field identity.
    # Also retain contiguous native checkbox runs as alternatives when keys
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
        for li, key_text in enumerate(key_texts):
            if key_text.scope != scope or not any(c.isalpha() for c in key_text.text):
                continue
            # Common prompts sit above the group or alongside its first option;
            # individual short option keys are not plausible common prompts.
            first_box = values[members[0]].bbox
            above = key_text.bbox.b <= union.t - 0.2 * h
            beside = (
                key_text.bbox.r <= first_box.l - h and key_text.bbox.t <= first_box.b
            )
            if not (above or beside) or gap(key_text.bbox, first_box) > 10 * h:
                continue
            if len(key_text.text.split()) < 2:
                continue
            features = {
                "prompt_distance": math.log1p(gap(key_text.bbox, union) / h),
                "prompt_prior": -2.5,
                "fragment": 0.5 * key_text.fragment,
            }
            candidates.append(Candidate(members, li, "choice_group", features))
    # Within a key, the whole text is the key, not one of its lines: a
    # line seen from the same side as its whole stacked key pays extra.
    whole: dict[tuple[tuple[int, ...], Side], list[frozenset[int]]] = defaultdict(list)
    for c in candidates:
        if c.side is not None and key_texts[c.key].stack:
            whole[c.members, c.side].append(key_texts[c.key].atoms)
    for c in candidates:
        if c.side is not None and any(
            key_texts[c.key].atoms < atoms for atoms in whole[c.members, c.side]
        ):
            c.features["partial"] = 0.5
    return candidates


def answers(
    members: tuple[int, ...],
    clause: KeyText,
    key_texts: list[KeyText],
    values: list[Value],
    h: float,
) -> set[int]:
    """Checkboxes in a clause that are options of it, not blanks within it.

    An option has its own key: a separate piece of the clause's text on its
    line, beside it within a text line. Two or more such checkboxes answer the
    clause ("Business income [ ] Yes [ ] No"), so each keeps its own key. A
    single checkbox, or a clause printed as one piece of text, stays a blank of
    the clause ("check here [ ] and enter the amount").
    """
    found = set()
    for i in members:
        if not values[i].checkbox:
            continue
        box = values[i].bbox
        if any(
            len(key_text.atoms) == 1
            and key_text.atoms < clause.atoms
            and lettered(key_text.text)
            and span_overlap(key_text.bbox.t, key_text.bbox.b, box.t, box.b)
            >= 0.5 * min(key_text.bbox.height, box.height)
            and (key_text.bbox.r <= box.l or key_text.bbox.l >= box.r)
            and gap(key_text.bbox, box) <= h
            for key_text in key_texts
        ):
            found.add(i)
    return found if len(found) > 1 else set()


def siblings(a: Value, b: Value, side: Side) -> bool:
    """Same kind and size, aligned along the line a shared key runs along.

    A row key keys every like-sized value of its row, a column key
    every like-sized value of its column. Operand boxes of a different size in
    the same row (multipliers, rates) do not inherit the row key.
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
