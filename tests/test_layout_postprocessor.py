from docling_core.types.doc import CoordOrigin, DocItemLabel, Size
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


def _text_cell(
    index: int,
    bbox: tuple[float, float, float, float] = (0, 0, 1, 1),
    text: str | None = None,
    *,
    from_ocr: bool = False,
) -> TextCell:
    left, top, right, bottom = bbox
    cell_text = str(index) if text is None else text
    return TextCell(
        index=index,
        rect=BoundingRectangle(
            r_x0=left,
            r_y0=top,
            r_x1=right,
            r_y1=top,
            r_x2=right,
            r_y2=bottom,
            r_x3=left,
            r_y3=bottom,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        text=cell_text,
        orig=cell_text,
        from_ocr=from_ocr,
    )


def _special_cluster_processor(
    pictures: list[Cluster], cells: list[TextCell]
) -> LayoutPostprocessor:
    processor = object.__new__(LayoutPostprocessor)
    processor.page_size = Size(width=600, height=800)
    processor.cells = cells
    processor.special_clusters = pictures
    processor.regular_clusters = []
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


def test_embedded_slide_container_is_removed_before_picture_deoverlap() -> None:
    outer_slide = _cluster(1, DocItemLabel.PICTURE, (10, 200, 590, 525))
    nested_chart = _cluster(2, DocItemLabel.PICTURE, (300, 250, 570, 500))
    native_cells = [
        _text_cell(
            index,
            (30, 230 + index * 30, 280, 250 + index * 30),
            "machine-readable slide text " * 4,
        )
        for index in range(3)
    ]
    processor = _special_cluster_processor([outer_slide, nested_chart], native_cells)

    result = processor._process_special_clusters()

    assert [cluster.id for cluster in result] == [nested_chart.id]


def test_embedded_slide_container_is_kept_for_ocr_only_content() -> None:
    outer_slide = _cluster(1, DocItemLabel.PICTURE, (10, 200, 590, 525))
    nested_chart = _cluster(2, DocItemLabel.PICTURE, (300, 250, 570, 500))
    ocr_cells = [
        _text_cell(
            index,
            (30, 230 + index * 30, 280, 250 + index * 30),
            "text recognized from a raster image " * 4,
            from_ocr=True,
        )
        for index in range(3)
    ]
    processor = _special_cluster_processor([outer_slide, nested_chart], ocr_cells)

    result = processor._process_special_clusters()

    assert [cluster.id for cluster in result] == [outer_slide.id]
