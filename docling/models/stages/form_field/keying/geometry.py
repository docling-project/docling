# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Box geometry shared by the keying steps."""

from __future__ import annotations

from docling_core.types.doc import BoundingBox

from docling.models.stages.form_field.keying.types import Side


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


def span_overlap(lo: float, hi: float, other_lo: float, other_hi: float) -> float:
    return max(0.0, min(hi, other_hi) - max(lo, other_lo))


def lettered(text: str) -> bool:
    return any(c.isalpha() for c in text)


def side_of(label: BoundingBox, value: BoundingBox) -> tuple[Side, float] | None:
    """Side from which a label faces a value, and how well it is aligned.

    Left/right labels share the value's row band; up/down labels its column
    band. Alignment is the shared fraction of the smaller perpendicular extent.
    Purely diagonal labels return None: only group candidates use them.
    """
    if overlap(label, value) >= 0.5 * min(label.area(), value.area()):
        return "inside", 1.0
    rows = span_overlap(label.t, label.b, value.t, value.b)
    columns = span_overlap(label.l, label.r, value.l, value.r)
    if rows > 0 and columns > 0:
        # A partial overlap, typically a caption straddling the value's top
        # edge inside the same printed box: use the axis it sticks out along.
        vertical = rows / label.height <= columns / label.width
    else:
        vertical = columns > 0
    if vertical:
        up = label.t + label.b < value.t + value.b
        return ("up" if up else "down"), columns / min(label.width, value.width)
    if rows > 0:
        left = label.l + label.r < value.l + value.r
        return ("left" if left else "right"), rows / min(label.height, value.height)
    return None


def corridor(label: BoundingBox, value: BoundingBox, side: Side) -> BoundingBox | None:
    """Space between a facing label and its value, within their shared band."""
    if side == "left":
        box = (label.r, max(label.t, value.t), value.l, min(label.b, value.b))
    elif side == "right":
        box = (value.r, max(label.t, value.t), label.l, min(label.b, value.b))
    elif side == "up":
        box = (max(label.l, value.l), label.b, min(label.r, value.r), value.t)
    elif side == "down":
        box = (max(label.l, value.l), value.b, min(label.r, value.r), label.t)
    else:
        return None
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        return None
    return BoundingBox(l=left, t=top, r=right, b=bottom)


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


def band_index(lo: float, hi: float, bands: dict[int, tuple[float, float]]) -> int:
    """The band sharing most of [lo, hi], else the one with the nearest centre."""
    shared = {k: span_overlap(lo, hi, *band) for k, band in bands.items()}
    best = max(shared, key=lambda k: (shared[k], -k))
    if shared[best] > 0:
        return best
    middle = (lo + hi) / 2
    return min(bands, key=lambda k: (abs(sum(bands[k]) / 2 - middle), k))
