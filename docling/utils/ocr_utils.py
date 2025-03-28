from typing import Optional, Tuple

from docling_core.types.doc import BoundingBox, CoordOrigin

_TESSERACT_ORIENTATIONS = {0, 90, 180, 270}

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
) -> Box:
    l, t, w, h = box
    rotated_w, rotated_h = rotated_im_size
    if orientation == 0:
        return box
    if orientation == 90:
        return rotated_h - (t + h), l, h, w
    if orientation == 180:
        return rotated_w - (l + w), rotated_h - (t + h), w, h
    if orientation == 270:
        return t, rotated_w - (l + w), h, w
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


def tesseract_box_to_bounding_box(
    box: Box,
    *,
    offset: Optional[BoundingBox] = None,
    scale: float,
    orientation: int,
    rotated_image_size: Size,
) -> BoundingBox:
    # box is in the top, left, height, width format + top left orientation
    original_box = reverse_tesseract_preprocessing_rotation(
        box, orientation, rotated_image_size
    )
    l, t, w, h = original_box
    r = l + w
    b = t + h
    bbox = BoundingBox.from_tuple(coord=(l, t, r, b), origin=CoordOrigin.TOPLEFT)
    bbox = bbox.scaled(1 / scale)
    if offset is not None:
        bbox.l += offset.l
        bbox.t += offset.t
        bbox.r += offset.l
        bbox.b += offset.t
    return bbox
