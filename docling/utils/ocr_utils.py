from typing import Optional

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

from docling.utils.orientation import (
    Box,
    Size,
    CLIPPED_ORIENTATIONS,
    rotate_ltwh_bounding_box,
)


def map_tesseract_script(script: str) -> str:
    r""" """
    if script == "Katakana" or script == "Hiragana":
        script = "Japanese"
    elif script == "Han":
        script = "HanS"
    elif script == "Korean":
        script = "Hangul"
    return script


def parse_tesseract_orientation(orientation: str) -> int:
    # Tesseract orientation is [0, 90, 180, 270] clockwise, bounding rectangle angles
    # are [0, 360[ counterclockwise
    parsed = int(orientation)
    if parsed not in CLIPPED_ORIENTATIONS:
        msg = (
            f"invalid tesseract document orientation {orientation}, "
            f"expected orientation: {sorted(CLIPPED_ORIENTATIONS)}"
        )
        raise ValueError(msg)
    parsed = -parsed
    parsed %= 360
    return parsed


def tesseract_box_to_bounding_rectangle(
    box: Box,
    *,
    offset: Optional[BoundingBox] = None,
    scale: float,
    orientation: int,
    rotated_image_size: Size,
) -> BoundingRectangle:
    # box is in the top, left, height, width format + top left orientation
    r_0, r_1, r_2, r_3 = rotate_ltwh_bounding_box(box, orientation, rotated_image_size)
    rect = BoundingRectangle(
        r_x0=r_0[0] / scale,
        r_y0=r_0[1] / scale,
        r_x1=r_1[0] / scale,
        r_y1=r_1[1] / scale,
        r_x2=r_2[0] / scale,
        r_y2=r_2[1] / scale,
        r_x3=r_3[0] / scale,
        r_y3=r_3[1] / scale,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    if offset is not None:
        rect.r_x0 += offset.l
        rect.r_x1 += offset.l
        rect.r_x2 += offset.l
        rect.r_x3 += offset.l
        rect.r_y0 += offset.t
        rect.r_y1 += offset.t
        rect.r_y2 += offset.t
        rect.r_y3 += offset.t
    return rect
