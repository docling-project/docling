from docling_core.types.doc import CoordOrigin, DocItemLabel
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfCellRenderingMode,
    PdfTextCell,
    TextCell,
)

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


def _pdf_text_cell(
    index: int,
    rendering_mode: PdfCellRenderingMode,
) -> PdfTextCell:
    return PdfTextCell(
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
        rendering_mode=rendering_mode,
        widget=False,
        font_key="F1",
        font_name="Helvetica",
    )


def test_sort_cells_uses_native_cell_index_order() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [_text_cell(3), _text_cell(1), _text_cell(2)]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.index for cell in sorted_cells] == [1, 2, 3]
    assert [cell.index for cell in cells] == [3, 1, 2]


def test_unassigned_invisible_pdf_cells_do_not_become_orphan_text() -> None:
    """The layout detector remains the gate for render-mode-invisible text."""
    visible = _pdf_text_cell(1, PdfCellRenderingMode.FILL_TEXT)
    invisible = _pdf_text_cell(2, PdfCellRenderingMode.INVISIBLE)
    clipping = _pdf_text_cell(3, PdfCellRenderingMode.ONLY_CLIPPING)
    processor = object.__new__(LayoutPostprocessor)
    processor.cells = [visible, invisible, clipping]

    assert processor._find_unassigned_cells([]) == [visible]


def test_invisible_pdf_cells_are_kept_when_layout_detects_text_region() -> None:
    """Searchable scan text survives when it overlaps a detected text cluster."""
    invisible = _pdf_text_cell(1, PdfCellRenderingMode.INVISIBLE)
    processor = object.__new__(LayoutPostprocessor)
    processor.cells = [invisible]
    detected_text = _cluster(1, DocItemLabel.TEXT, (0, 0, 1, 1))
    detected_text.bbox = detected_text.bbox.model_copy(
        update={"t": 1, "b": 0, "coord_origin": CoordOrigin.BOTTOMLEFT}
    )

    assigned = processor._assign_cells_to_clusters([detected_text])

    assert assigned[0].cells == [invisible]
    assert processor._find_unassigned_cells(assigned) == []


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
