from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster, Page
from docling.datamodel.pipeline_options import LayoutOptions
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


def _make_cluster(id: int, label: DocItemLabel, confidence: float) -> Cluster:
    return Cluster(
        id=id,
        label=label,
        confidence=confidence,
        bbox=BoundingBox(l=10, t=10, r=200, b=150),
    )


def _make_postprocessor(clusters: list[Cluster]) -> LayoutPostprocessor:
    page = Page(page_no=1)
    page.size = Size(width=500, height=700)
    return LayoutPostprocessor(page, clusters, LayoutOptions(skip_cell_assignment=True))


def test_sort_cells_uses_native_cell_index_order() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [_text_cell(3), _text_cell(1), _text_cell(2)]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.index for cell in sorted_cells] == [1, 2, 3]
    assert [cell.index for cell in cells] == [3, 1, 2]


def test_picture_overlapping_table_is_removed() -> None:
    """A PICTURE cluster substantially overlapping a TABLE cluster must be dropped."""
    table = _make_cluster(0, DocItemLabel.TABLE, confidence=0.85)
    picture = _make_cluster(1, DocItemLabel.PICTURE, confidence=0.72)

    final_clusters, _ = _make_postprocessor([table, picture]).postprocess()

    labels = {c.label for c in final_clusters}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.PICTURE not in labels


def test_non_overlapping_picture_survives() -> None:
    """A PICTURE cluster on a different region must not be removed."""
    table = _make_cluster(0, DocItemLabel.TABLE, confidence=0.85)
    picture = Cluster(
        id=1,
        label=DocItemLabel.PICTURE,
        confidence=0.80,
        bbox=BoundingBox(l=250, t=200, r=450, b=350),  # non-overlapping
    )

    final_clusters, _ = _make_postprocessor([table, picture]).postprocess()

    labels = {c.label for c in final_clusters}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.PICTURE in labels
