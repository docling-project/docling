from typing import Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle

_TESSERACT_ORIENTATIONS = {0, 90, 180, 270}

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]
Size = Tuple[int, int]


def map_tesseract_script(script: str) -> str:
    r""" """
    if script == "Katakana" or script == "Hiragana":
        script = "Japanese"
    elif script == "Han":
        script = "HanS"
    elif script == "Korean":
        script = "Hangul"
    return script


def reverse_tesseract_preprocessing_rotation(
    box: Box, orientation: int, rotated_im_size: Size
) -> tuple[Point, Point, Point, Point]:
    l, t, w, h = box
    rotated_w, rotated_h = rotated_im_size
    if orientation == 0:
        return (l, t), (l + w, t), (l + w, t + h), (l, t + h)
    if orientation == 90:
        x0 = rotated_h - t
        y0 = l
        return (x0, y0), (x0, y0 + w), (x0 - h, y0 + w), (x0 - h, y0)
    if orientation == 180:
        x0 = rotated_w - l
        y0 = rotated_h - t
        return (x0, y0), (x0 - w, y0), (x0 - w, y0 - h), (x0, y0 - h)
    if orientation == 270:
        x0 = t
        y0 = rotated_w - l
        return (x0, y0), (x0, y0 - w), (x0 + h, y0 - w), (x0 + h, y0)
    msg = (
        f"invalid tesseract document orientation {orientation}, "
        f"expected orientation: {sorted(_TESSERACT_ORIENTATIONS)}"
    )
    raise ValueError(msg)


def parse_tesseract_orientation(orientation: str) -> int:
    parsed = int(orientation)
    if parsed not in _TESSERACT_ORIENTATIONS:
        msg = (
            f"invalid tesseract document orientation {orientation}, "
            f"expected orientation: {sorted(_TESSERACT_ORIENTATIONS)}"
        )
        raise ValueError(msg)
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
    r_0, r_1, r_2, r_3 = reverse_tesseract_preprocessing_rotation(
        box, orientation, rotated_image_size
    )
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
