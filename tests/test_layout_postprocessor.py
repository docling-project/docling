from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster
from docling.utils.layout_postprocessor import LayoutPostprocessor


def _cluster(
    cid: int, label: DocItemLabel, bbox: tuple, confidence: float = 0.8
) -> Cluster:
    left, top, right, bottom = bbox
    return Cluster(
        id=cid,
        label=label,
        bbox=BoundingBox(l=left, t=top, r=right, b=bottom),
        confidence=confidence,
    )


def _text_cell(index: int) -> TextCell:
    return TextCell(
        index=index,
        rect=BoundingRectangle(
            r_x0=0,
            r_y0=0,
            r_x1=1,
            r_y1=0,
            r_x2=1,
            r_y2=1,
            r_x3=0,
            r_y3=1,
        ),
        text=str(index),
        orig=str(index),
        from_ocr=False,
    )


def test_sort_cells_uses_native_cell_index_order() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [_text_cell(3), _text_cell(1), _text_cell(2)]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.index for cell in sorted_cells] == [1, 2, 3]
    assert [cell.index for cell in cells] == [3, 1, 2]


def _positioned_text_cell(
    index: int,
    bbox: tuple,
    *,
    text: str = "",
    confidence: float = 1.0,
    coord_origin: CoordOrigin = CoordOrigin.TOPLEFT,
) -> TextCell:
    left, top, right, bottom = bbox
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(
            BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=coord_origin)
        ),
        text=text or str(index),
        orig=text or str(index),
        confidence=confidence,
        from_ocr=True,
    )


def test_sort_cells_orders_two_ocr_passes_by_position_not_index() -> None:
    # Reproduces the real bug: a low-confidence-region retry re-OCRs part of a
    # paragraph, emitting cells indexed independently of the first pass. Pass A
    # (indices 0-1) only caught two scattered lines; pass B (higher indices,
    # 18-26) caught the whole paragraph in the correct top-to-bottom order.
    # Sorting by index alone put pass A's fragment first regardless of where it
    # physically belongs; sorting by position must interleave/order by row instead.
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(0, (35, 577, 341, 586), text="pass-a-line-2-tail"),
        _positioned_text_cell(1, (35, 629, 195, 638), text="pass-a-last-line"),
        _positioned_text_cell(18, (35, 562, 350, 576), text="pass-b-line-1"),
        _positioned_text_cell(21, (35, 588, 360, 601), text="pass-b-line-3"),
        _positioned_text_cell(23, (35, 601, 341, 614), text="pass-b-line-4"),
        _positioned_text_cell(26, (35, 614, 332, 627), text="pass-b-line-5"),
    ]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.text for cell in sorted_cells] == [
        "pass-b-line-1",
        "pass-a-line-2-tail",
        "pass-b-line-3",
        "pass-b-line-4",
        "pass-b-line-5",
        "pass-a-last-line",
    ]


def test_sort_cells_groups_same_row_by_left_x_despite_top_jitter() -> None:
    # Real per-word bboxes on one visual line never share the exact same top
    # coordinate. A naive sort by (top, left) would place "beta" before "alpha"
    # here since 100.4 < 100.6 even though alpha is to the left of beta.
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(0, (60, 100.6, 100, 110), text="alpha"),
        _positioned_text_cell(1, (10, 100.4, 55, 110), text="beta"),
    ]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.text for cell in sorted_cells] == ["beta", "alpha"]


def test_sort_cells_orders_rows_top_to_bottom() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(0, (10, 200, 100, 210), text="row-2"),
        _positioned_text_cell(1, (10, 10, 100, 20), text="row-1"),
    ]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.text for cell in sorted_cells] == ["row-1", "row-2"]


def test_sort_cells_handles_bottomleft_origin() -> None:
    # BOTTOMLEFT's t grows upward (larger t = higher on the page), the opposite
    # of TOPLEFT. A sort that assumes TOPLEFT's convention unconditionally would
    # emit these in reverse.
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(
            0,
            (10, 20, 100, 30),
            text="lower-on-page",
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
        _positioned_text_cell(
            1,
            (10, 200, 100, 210),
            text="higher-on-page",
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
    ]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.text for cell in sorted_cells] == ["higher-on-page", "lower-on-page"]


def test_deduplicate_cells_drops_overlapping_cell_keeping_higher_confidence() -> None:
    # Two OCR passes detecting the same physical line produce cells with
    # near-identical, overlapping bboxes but different (independent) indices.
    # Only cell.index-based dedup can't catch this; overlap-based dedup must.
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(0, (35, 577, 341, 586), text="pass-a", confidence=0.4),
        _positioned_text_cell(20, (35, 574, 338, 589), text="pass-b", confidence=0.9),
    ]

    result = processor._deduplicate_cells(cells)

    assert [cell.text for cell in result] == ["pass-b"]


def test_deduplicate_cells_keeps_non_overlapping_cells() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [
        _positioned_text_cell(0, (10, 10, 100, 20), text="line-1"),
        _positioned_text_cell(1, (10, 30, 100, 40), text="line-2"),
    ]

    result = processor._deduplicate_cells(cells)

    assert [cell.text for cell in result] == ["line-1", "line-2"]


def test_cross_type_overlaps_removes_picture_coinciding_with_table() -> None:
    # The layout model proposes the same region as both a PICTURE and a TABLE.
    # The PICTURE (near-identical bbox, high IoU) must be removed; the TABLE kept.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(1, DocItemLabel.TABLE, (10, 10, 200, 150), confidence=0.72)
    picture = _cluster(2, DocItemLabel.PICTURE, (10, 10, 200, 150), confidence=0.81)

    result = processor._handle_cross_type_overlaps([table, picture])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.PICTURE not in labels


def test_cross_type_overlaps_keeps_picture_not_overlapping_table() -> None:
    # A genuine figure elsewhere on the page must be preserved.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(1, DocItemLabel.TABLE, (10, 10, 200, 150))
    picture = _cluster(2, DocItemLabel.PICTURE, (10, 300, 200, 450))

    result = processor._handle_cross_type_overlaps([table, picture])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_keeps_small_picture_inside_table() -> None:
    # A small figure fully contained in a large table (high containment but low IoU)
    # must NOT be removed -- only a near-coinciding picture is a true mislabel.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(1, DocItemLabel.TABLE, (0, 0, 400, 400))
    small_picture = _cluster(2, DocItemLabel.PICTURE, (10, 10, 60, 60))

    result = processor._handle_cross_type_overlaps([table, small_picture])

    ids = {c.id for c in result}
    assert ids == {1, 2}
