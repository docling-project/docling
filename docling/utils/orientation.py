from collections import Counter
from operator import itemgetter
from typing import Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

CLIPPED_ORIENTATIONS = [0, 90, 180, 270]


def _clipped_orientation(angle: float) -> int:
    return min((abs(angle - o) % 360, o) for o in CLIPPED_ORIENTATIONS)[1]


def detect_orientation(cells: list[TextCell]) -> int:
    if not cells:
        return 0
    orientation_counter = Counter(_clipped_orientation(c.rect.angle_360) for c in cells)
    return max(orientation_counter.items(), key=itemgetter(1))[0]


def rotate_bounding_box(
    bbox: BoundingBox, angle: int, im_size: Tuple[int, int]
) -> BoundingRectangle:
    # The box is left top width height in TOPLEFT coordinates
    # Bounding rectangle start with r_0 at the bottom left whatever the
    # coordinate system. Then other corners are found rotating counterclockwise
    bbox = bbox.to_top_left_origin(im_size[1])
    l, t, w, h = bbox.l, bbox.t, bbox.width, bbox.height
    im_h, im_w = im_size
    angle = angle % 360
    if angle == 0:
        r_x0 = l
        r_y0 = t + h
        r_x1 = r_x0 + w
        r_y1 = r_y0
        r_x2 = r_x0 + w
        r_y2 = r_y0 - h
        r_x3 = r_x0
        r_y3 = r_y0 - h
    elif angle == 90:
        r_x0 = im_w - (t + h)
        r_y0 = l
        r_x1 = r_x0
        r_y1 = r_y0 + w
        r_x2 = r_x0 + h
        r_y2 = r_y0 + w
        r_x3 = r_x0
        r_y3 = r_y0 + w
    elif angle == 180:
        r_x0 = im_h - l
        r_y0 = im_w - (t + h)
        r_x1 = r_x0 - w
        r_y1 = r_y0
        r_x2 = r_x0 - w
        r_y2 = r_y0 + h
        r_x3 = r_x0
        r_y3 = r_y0 + h
    elif angle == 270:
        r_x0 = t + h
        r_y0 = im_h - l
        r_x1 = r_x0
        r_y1 = r_y0 - w
        r_x2 = r_x0 - h
        r_y2 = r_y0 - w
        r_x3 = r_x0 - h
        r_y3 = r_y0
    else:
        msg = (
            f"invalid orientation {angle}, expected values in:"
            f" {sorted(CLIPPED_ORIENTATIONS)}"
        )
        raise ValueError(msg)
    return BoundingRectangle(
        r_x0=r_x0,
        r_y0=r_y0,
        r_x1=r_x1,
        r_y1=r_y1,
        r_x2=r_x2,
        r_y2=r_y2,
        r_x3=r_x3,
        r_y3=r_y3,
        coord_origin=CoordOrigin.TOPLEFT,
    )
