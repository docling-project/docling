# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

CLIPPED_ORIENTATIONS = [0, 90, 180, 270]


def rotate_bounding_box(
    bbox: BoundingBox, angle: int, im_size: Tuple[int, int]
) -> BoundingRectangle:
    # The box is left top width height in TOPLEFT coordinates
    # Bounding rectangle start with r_0 at the bottom left whatever the
    # coordinate system. Then other corners are found rotating counterclockwise
    bbox = bbox.to_top_left_origin(im_size[1])
    left, top, width, height = bbox.l, bbox.t, bbox.width, bbox.height
    im_w, im_h = im_size
    angle = angle % 360
    if angle == 0:
        return BoundingRectangle.from_bounding_box(bbox)
    elif angle == 90:
        r_x0 = top + height
        r_y0 = im_w - left
        r_x1 = r_x0
        r_y1 = r_y0 - width
        r_x2 = r_x1 - height
        r_y2 = r_y1
        r_x3 = r_x2
        r_y3 = r_y0
    elif angle == 180:
        r_x0 = im_w - left
        r_y0 = im_h - (top + height)
        r_x1 = r_x0 - width
        r_y1 = r_y0
        r_x2 = r_x1
        r_y2 = r_y1 + height
        r_x3 = r_x0
        r_y3 = r_y2
    elif angle == 270:
        r_x0 = im_h - (top + height)
        r_y0 = left
        r_x1 = r_x0
        r_y1 = r_y0 + width
        r_x2 = r_x1 + height
        r_y2 = r_y1
        r_x3 = r_x2
        r_y3 = r_y0
    else:
        msg = (
            f"invalid orientation {angle}, expected values in:"
            f" {sorted(CLIPPED_ORIENTATIONS)}"
        )
        raise ValueError(msg)
    rectangle = BoundingRectangle(
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
    return rectangle


def ocr_box_to_bounding_rectangle(
    bbox: BoundingBox,
    *,
    original_offset: Optional[BoundingBox] = None,
    scale: float,
    orientation: int,
    im_size: Tuple[int, int],
) -> BoundingRectangle:
    # box is in the top, left, height, width format, top left coordinates
    rect = rotate_bounding_box(bbox, angle=orientation, im_size=im_size)
    rect = BoundingRectangle(
        r_x0=rect.r_x0 / scale,
        r_y0=rect.r_y0 / scale,
        r_x1=rect.r_x1 / scale,
        r_y1=rect.r_y1 / scale,
        r_x2=rect.r_x2 / scale,
        r_y2=rect.r_y2 / scale,
        r_x3=rect.r_x3 / scale,
        r_y3=rect.r_y3 / scale,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    if original_offset is not None:
        if original_offset.coord_origin is not CoordOrigin.TOPLEFT:
            msg = f"expected coordinate origin to be {CoordOrigin.TOPLEFT.value}"
            raise ValueError(msg)
        if original_offset is not None:
            rect.r_x0 += original_offset.l
            rect.r_x1 += original_offset.l
            rect.r_x2 += original_offset.l
            rect.r_x3 += original_offset.l
            rect.r_y0 += original_offset.t
            rect.r_y1 += original_offset.t
            rect.r_y2 += original_offset.t
            rect.r_y3 += original_offset.t
    return rect
