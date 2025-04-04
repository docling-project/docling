from collections import Counter
from operator import itemgetter

from docling_core.types.doc.page import TextCell

_ORIENTATIONS = [0, 90, 180, 270]


def _clipped_orientation(angle: float) -> int:
    return min((abs(angle - o) % 360, o) for o in _ORIENTATIONS)[1]


def detect_orientation(cells: list[TextCell]) -> int:
    if not cells:
        return 0
    orientation_counter = Counter(_clipped_orientation(c.rect.angle_360) for c in cells)
    return max(orientation_counter.items(), key=itemgetter(1))[0]
