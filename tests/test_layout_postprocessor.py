from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster
from docling.datamodel.pipeline_options import LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor, SpatialClusterIndex


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


def _special_cluster_processor(
    pictures: list[Cluster], regular_clusters: list[Cluster]
) -> LayoutPostprocessor:
    processor = object.__new__(LayoutPostprocessor)
    processor.page_size = Size(width=600, height=800)
    processor.cells = []
    processor.special_clusters = pictures
    processor.regular_clusters = regular_clusters
    processor.options = LayoutPostprocessorOptions(skip_cell_assignment=True)
    processor.picture_index = SpatialClusterIndex(pictures)
    processor.wrapper_index = SpatialClusterIndex([])
    return processor


def test_sort_cells_uses_native_cell_index_order() -> None:
    processor = object.__new__(LayoutPostprocessor)
    cells = [_text_cell(3), _text_cell(1), _text_cell(2)]

    sorted_cells = processor._sort_cells(cells)

    assert [cell.index for cell in sorted_cells] == [1, 2, 3]
    assert [cell.index for cell in cells] == [3, 1, 2]


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


def test_content_union_removes_wrapper_picture_before_deoverlap() -> None:
    outer_slide = _cluster(1, DocItemLabel.PICTURE, (10, 200, 590, 525))
    nested_chart = _cluster(2, DocItemLabel.PICTURE, (300, 250, 570, 500))
    slide_text = _cluster(3, DocItemLabel.TEXT, (30, 230, 280, 320))
    processor = _special_cluster_processor([outer_slide, nested_chart], [slide_text])

    result = processor._process_special_clusters()

    assert [cluster.id for cluster in result] == [nested_chart.id]


def test_content_union_keeps_lone_picture() -> None:
    lone_picture = _cluster(1, DocItemLabel.PICTURE, (0, 0, 600, 800))
    processor = _special_cluster_processor([lone_picture], [])

    result = processor._process_special_clusters()

    assert [cluster.id for cluster in result] == [lone_picture.id]


def test_content_union_keeps_picture_with_external_caption() -> None:
    picture = _cluster(1, DocItemLabel.PICTURE, (10, 10, 400, 400))
    caption = _cluster(2, DocItemLabel.CAPTION, (10, 420, 400, 450))
    processor = _special_cluster_processor([picture], [caption])

    result = processor._process_special_clusters()

    assert [cluster.id for cluster in result] == [picture.id]


def test_content_union_keeps_duplicate_lone_picture_proposals() -> None:
    first_picture = _cluster(1, DocItemLabel.PICTURE, (10, 10, 400, 400))
    duplicate_picture = _cluster(2, DocItemLabel.PICTURE, (10, 10, 400, 400))
    processor = _special_cluster_processor([first_picture, duplicate_picture], [])

    result = processor._process_special_clusters()

    assert len(result) == 1
    assert result[0].label == DocItemLabel.PICTURE
