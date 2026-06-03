from docling_core.types.doc import BoundingBox, DocItemLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster
from docling.utils.layout_postprocessor import LayoutPostprocessor


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


def _cluster(id: int, label: DocItemLabel, confidence: float = 0.9) -> Cluster:
    return Cluster(
        id=id,
        label=label,
        bbox=BoundingBox(l=0, t=0, r=100, b=100),
        confidence=confidence,
    )


def test_handle_cross_type_overlaps_removes_picture_overlapping_table() -> None:
    """A PICTURE that shares the same region as a TABLE should be dropped (issue #3495)."""
    processor = object.__new__(LayoutPostprocessor)
    table = _cluster(1, DocItemLabel.TABLE, confidence=0.9)
    picture = _cluster(2, DocItemLabel.PICTURE, confidence=0.8)

    result = processor._handle_cross_type_overlaps([table, picture])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.PICTURE not in labels


def test_handle_cross_type_overlaps_removes_kvregion_overlapping_table() -> None:
    """A KEY_VALUE_REGION that heavily overlaps a TABLE should be dropped."""
    processor = object.__new__(LayoutPostprocessor)
    table = _cluster(1, DocItemLabel.TABLE, confidence=0.9)
    kvr = _cluster(2, DocItemLabel.KEY_VALUE_REGION, confidence=0.85)

    result = processor._handle_cross_type_overlaps([table, kvr])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.KEY_VALUE_REGION not in labels


def test_handle_cross_type_overlaps_keeps_picture_without_table() -> None:
    """A PICTURE with no overlapping TABLE should survive."""
    processor = object.__new__(LayoutPostprocessor)
    picture = _cluster(1, DocItemLabel.PICTURE, confidence=0.8)

    result = processor._handle_cross_type_overlaps([picture])

    assert len(result) == 1
    assert result[0].label == DocItemLabel.PICTURE


def test_sort_cells_uses_native_cell_index_order() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [_text_cell(3), _text_cell(1), _text_cell(2)]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.index for cell in sorted_cells] == [1, 2, 3]
    assert [cell.index for cell in cells] == [3, 1, 2]
