# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Helpers shared by the form-field stage and the reading order."""

from collections.abc import Sequence

from docling_core.types.doc import BoundingBox, CoordOrigin


def without_whitespace(text: str) -> str:
    """The text with all whitespace removed.

    Printed text equals a widget's painted value when both agree without
    whitespace: the layout may split or join the painted characters.
    """
    return "".join(text.split())


def reading_order(a: BoundingBox, b: BoundingBox) -> int:
    """-1 when ``a`` reads before ``b``, 1 when after, 0 when geometry can't tell.

    Boxes on one line (sharing half the smaller height) read left to right;
    boxes in one column (sharing any width) read top to bottom. Boxes that
    share neither, as across two columns, set no order.
    """
    if a.y_overlap_with(b) >= 0.5 * min(a.height, b.height):
        first, second = a.l + a.r, b.l + b.r
    elif a.x_overlap_with(b) > 0:
        first, second = a.t + a.b, b.t + b.b
        if a.coord_origin == CoordOrigin.BOTTOMLEFT:
            first, second = -first, -second  # up is larger there
    else:
        return 0
    return -1 if first < second else 1 if first > second else 0


def text_slots(
    texts: Sequence[BoundingBox], fields: Sequence[BoundingBox]
) -> list[int]:
    """For each text, how many fields precede it once both are interleaved.

    The texts keep their order, as the reading order gives it, and so do the
    fields, in their native widget order; only how they interleave is chosen.
    The slots minimize the text-field pairs read against reading_order(). On a
    tie a text goes as late as it can: a field that geometry leaves free stays
    with the fields before it.
    """
    count = len(fields)
    previous: list[int] = []
    back: list[list[int]] = []
    for text in texts:
        order = [reading_order(text, field) for field in fields]
        # Placed after k fields: the first k must not read after the text,
        # the others must not read before it.
        cost = [0] * (count + 1)
        late = sum(o > 0 for o in order)
        early = 0
        for k in range(count + 1):
            cost[k] = early + late
            if k < count:
                early += order[k] < 0
                late -= order[k] > 0
        if not previous:
            back.append(list(range(count + 1)))
            previous = cost
            continue
        best, arg = previous[0], 0
        total, links = [], []
        for k in range(count + 1):
            if previous[k] <= best:
                best, arg = previous[k], k
            total.append(cost[k] + best)
            links.append(arg)
        back.append(links)
        previous = total
    if not previous:
        return []
    slot = max(range(count + 1), key=lambda k: (-previous[k], k))
    slots = [slot]
    for links in reversed(back[1:]):
        slots.append(links[slots[-1]])
    return slots[::-1]
