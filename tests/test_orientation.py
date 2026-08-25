from typing import Tuple

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

from docling.utils.orientation import rotate_bounding_box

IM_SIZE = (4, 5)
BBOX = BoundingBox(l=1, t=3, r=3, b=4, coord_origin=CoordOrigin.TOPLEFT)
RECT = BoundingRectangle(
    r_x0=1,
    r_y0=4,
    r_x1=3,
    r_y1=4,
    r_x2=3,
    r_y2=3,
    r_x3=1,
    r_y3=3,
    coord_origin=CoordOrigin.TOPLEFT,
)
RECT_90 = BoundingRectangle(
    r_x0=4,
    r_y0=3,
    r_x1=4,
    r_y1=1,
    r_x2=3,
    r_y2=1,
    r_x3=3,
    r_y3=3,
    coord_origin=CoordOrigin.TOPLEFT,
)
RECT_180 = BoundingRectangle(
    r_x0=3,
    r_y0=1,
    r_x1=1,
    r_y1=1,
    r_x2=1,
    r_y2=2,
    r_x3=3,
    r_y3=2,
    coord_origin=CoordOrigin.TOPLEFT,
)
RECT_270 = BoundingRectangle(
    r_x0=1,
    r_y0=1,
    r_x1=1,
    r_y1=3,
    r_x2=2,
    r_y2=3,
    r_x3=2,
    r_y3=1,
    coord_origin=CoordOrigin.TOPLEFT,
)


@pytest.mark.parametrize(
    ["bbox", "im_size", "angle", "expected_rectangle"],
    [
        # (BBOX, IM_SIZE, 0, RECT),
        # (BBOX, IM_SIZE, 90, RECT_90),
        (BBOX, IM_SIZE, 180, RECT_180),
        # (BBOX, IM_SIZE, 270, RECT_270),
        # (BBOX, IM_SIZE, 360, RECT),
        # (BBOX, IM_SIZE, -90, RECT_270),
        (BBOX, IM_SIZE, -180, RECT_180),
        # (BBOX, IM_SIZE, -270, RECT_90),
    ],
)
def test_rotate_bounding_box(
    bbox: BoundingBox,
    im_size: Tuple[int, int],
    angle: int,
    expected_rectangle: BoundingRectangle,
):
    rotated = rotate_bounding_box(bbox, angle, im_size)

    assert rotated == expected_rectangle
    expected_angle_360 = angle % 360
    assert rotated.angle_360 == expected_angle_360
