# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Printed rules of a page and the printed cell they draw around a value."""

from __future__ import annotations

from dataclasses import dataclass

from docling_core.types.doc import BoundingBox

# A painted shape at most this thick is a rule (the backend's default too).
THIN = 3.5
# Shorter marks are ticks, dots or glyph parts, not cell borders.
MIN_LENGTH = 5.0
# Collinear pieces this far off the same line are one rule: many forms draw a
# border cell by cell. Along the line they may be apart by up to THIN, where a
# crossing rule interrupts them.
MERGE = 1.0
# A rule this close to a value's edge is its own printed box, unless it runs
# past the value: then it is the border of the row or column the value sits in.
OUTLINE = 2.0


@dataclass(frozen=True)
class Rule:
    horizontal: bool
    at: float  # y of a horizontal rule, x of a vertical one (top-left points)
    start: float
    end: float


def rules_of(boxes: list[BoundingBox]) -> list[Rule]:
    """Thin boxes as horizontal or vertical rules, collinear pieces joined.

    Pieces of one line join across gaps no wider than a rule, the joints where
    crossing rules cut them.
    """
    pieces = []
    for box in boxes:
        if box.height <= THIN and box.width >= MIN_LENGTH:
            pieces.append((True, (box.t + box.b) / 2, box.l, box.r))
        elif box.width <= THIN and box.height >= MIN_LENGTH:
            pieces.append((False, (box.l + box.r) / 2, box.t, box.b))
    merged: list[list] = []
    for horizontal, at, start, end in sorted(pieces):
        last = next(
            (
                rule
                for rule in reversed(merged)
                if rule[0] == horizontal
                and abs(rule[1] - at) <= MERGE
                and start <= rule[3] + THIN
            ),
            None,
        )
        if last is None:
            merged.append([horizontal, at, start, end])
        else:
            last[2], last[3] = min(last[2], start), max(last[3], end)
    return [Rule(*rule) for rule in merged]


def printed_cell(
    box: BoundingBox, rules: list[Rule], frame: BoundingBox
) -> BoundingBox | None:
    """The cell printed around a value: the nearest rule on each side within frame.

    A rule along the value's own edge is its printed outline and bounds nothing,
    unless it runs past the value. None when a side has no rule.
    """
    middle_x, middle_y = (box.l + box.r) / 2, (box.t + box.b) / 2
    found: dict[str, tuple[float, float]] = {}

    def offer(side: str, distance: float, at: float) -> None:
        if side not in found or distance < found[side][0]:
            found[side] = (distance, at)

    for rule in rules:
        if rule.horizontal:
            if not (
                rule.start <= middle_x <= rule.end
                and frame.t - OUTLINE <= rule.at <= frame.b + OUTLINE
            ):
                continue
            longer = rule.start < box.l - OUTLINE or rule.end > box.r + OUTLINE
            for side, edge, beyond in (("up", box.t, -1), ("down", box.b, 1)):
                near = abs(rule.at - edge) <= OUTLINE
                if (near and longer) or (not near and (rule.at - edge) * beyond > 0):
                    offer(side, abs(rule.at - edge), rule.at)
        else:
            if not (
                rule.start <= middle_y <= rule.end
                and frame.l - OUTLINE <= rule.at <= frame.r + OUTLINE
            ):
                continue
            longer = rule.start < box.t - OUTLINE or rule.end > box.b + OUTLINE
            for side, edge, beyond in (("left", box.l, -1), ("right", box.r, 1)):
                near = abs(rule.at - edge) <= OUTLINE
                if (near and longer) or (not near and (rule.at - edge) * beyond > 0):
                    offer(side, abs(rule.at - edge), rule.at)
    if len(found) < 4:
        return None
    return BoundingBox(
        l=found["left"][1], t=found["up"][1], r=found["right"][1], b=found["down"][1]
    )
