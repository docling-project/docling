from typing import Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

from docling.utils.orientation import CLIPPED_ORIENTATIONS, rotate_bounding_box


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
    bbox: BoundingBox,
    *,
    original_offset: Optional[BoundingBox] = None,
    scale: float,
    orientation: int,
    im_size: Tuple[int, int],
) -> BoundingRectangle:
    # box is in the top, left, height, width format, top left coordinates
    rect = rotate_bounding_box(bbox, angle=-orientation, im_size=im_size)
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
