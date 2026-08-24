from collections.abc import Iterable

from docling_core.types.doc import BoundingBox, CoordOrigin
from rtree import index

SpatialBounds = tuple[float, float, float, float]


def ordered_bounds(bbox: BoundingBox) -> SpatialBounds:
    return (
        min(bbox.l, bbox.r),
        min(bbox.t, bbox.b),
        max(bbox.l, bbox.r),
        max(bbox.t, bbox.b),
    )


def has_positive_area(bbox: BoundingBox) -> bool:
    left, top, right, bottom = ordered_bounds(bbox)
    return left < right and top < bottom


def ordered_bounding_box(bbox: BoundingBox) -> BoundingBox:
    left, low_y, right, high_y = ordered_bounds(bbox)
    top, bottom = (
        (high_y, low_y)
        if bbox.coord_origin == CoordOrigin.BOTTOMLEFT
        else (low_y, high_y)
    )
    return BoundingBox(
        l=left,
        t=top,
        r=right,
        b=bottom,
        coord_origin=bbox.coord_origin,
    )


class BoundingBoxSpatialIndex:
    def __init__(self) -> None:
        properties = index.Property()
        properties.dimension = 2
        self._index = index.Index(properties=properties)

    def insert(self, item_id: int, bbox: BoundingBox) -> None:
        self._index.insert(item_id, ordered_bounds(bbox))

    def delete(self, item_id: int, bbox: BoundingBox) -> None:
        self._index.delete(item_id, ordered_bounds(bbox))

    def intersection(self, bbox: BoundingBox) -> Iterable[int]:
        return self._index.intersection(ordered_bounds(bbox))
