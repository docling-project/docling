from collections import Counter
from operator import itemgetter
from typing import Tuple

from docling_core.types.doc.page import TextCell


Point = Tuple[float, float]
Box = Tuple[float, float, float, float]
Size = Tuple[int, int]

CLIPPED_ORIENTATIONS = [0, 90, 180, 270]


def _clipped_orientation(angle: float) -> int:
    return min((abs(angle - o) % 360, o) for o in CLIPPED_ORIENTATIONS)[1]


def detect_orientation(cells: list[TextCell]) -> int:
    if not cells:
        return 0
    orientation_counter = Counter(_clipped_orientation(c.rect.angle_360) for c in cells)
    return max(orientation_counter.items(), key=itemgetter(1))[0]


def rotate_ltwh_bounding_box(
    box: Box, orientation: int, rotated_im_size: Size
) -> tuple[Point, Point, Point, Point]:
    # The box is left top width height in TOPLEFT coordinates
    # Bounding rectangle start with r_0 at the bottom left whatever the
    # coordinate system. Then other corners are found rotating counterclockwise
    l, t, w, h = box
    rotated_im_w, rotated_im_h = rotated_im_size
    if orientation == 0:
        r0_x = l
        r0_y = t + h
        return (r0_x, r0_y), (r0_x + w, r0_y), (r0_x + w, r0_y - h), (r0_x, r0_y - h)
    if orientation == 90:
        r0_x = t + h
        r0_y = rotated_im_w - l
        return (r0_x, r0_y), (r0_x, r0_y - w), (r0_x - h, r0_y - w), (r0_x - h, r0_y)
    if orientation == 180:
        r0_x = rotated_im_w - l
        r0_y = rotated_im_h - (t + h)
        return (r0_x, r0_y), (r0_x - w, r0_y), (r0_x - w, r0_y + h), (r0_x, r0_y + h)
    if orientation == 270:
        r0_x = rotated_im_h - (t + h)
        r0_y = l
        return (r0_x, r0_y), (r0_x, r0_y + w), (r0_x + h, r0_y + w), (r0_x, r0_y + w)
    msg = (
        f"orientation {orientation}, expected values in:"
        f" {sorted(CLIPPED_ORIENTATIONS)}"
    )
    raise ValueError(msg)
