import pytest
from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster, Page
from docling.datamodel.pipeline_options import LayoutOptions
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


def _postprocessor(*clusters: Cluster) -> LayoutPostprocessor:
    return LayoutPostprocessor(
        page=Page(page_no=1, size=Size(width=1000, height=1000)),
        clusters=list(clusters),
        options=LayoutOptions(skip_cell_assignment=True),
    )


def _process_special_clusters(*clusters: Cluster) -> list[Cluster]:
    processor = _postprocessor(*clusters)
    processor.regular_clusters = processor._process_regular_clusters()
    return processor._process_special_clusters()


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


@pytest.mark.parametrize(
    "container_label",
    [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION],
)
def test_container_nests_structured_children(container_label: DocItemLabel) -> None:
    container = _cluster(1, container_label, (0, 0, 400, 400), confidence=0.65)
    table = _cluster(2, DocItemLabel.TABLE, (10, 10, 150, 100), confidence=0.88)
    picture = _cluster(3, DocItemLabel.PICTURE, (200, 200, 300, 300), confidence=0.82)
    text = _cluster(4, DocItemLabel.TEXT, (20, 20, 140, 80), confidence=0.9)

    result = _process_special_clusters(container, table, picture, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert set(by_id) == {1, 2, 3}
    assert [child.id for child in by_id[1].children] == [2, 3]
    assert [child.id for child in by_id[2].children] == [4]
    assert by_id[3].children == []


def test_container_direct_text_remains_available_for_reading_order() -> None:
    container = _cluster(1, DocItemLabel.FORM, (0, 0, 400, 400), confidence=0.8)
    caption = _cluster(2, DocItemLabel.CAPTION, (10, 300, 300, 350), confidence=0.8)

    result = _postprocessor(container, caption).postprocess()

    by_id = {cluster.id: cluster for cluster in result}
    assert set(by_id) == {1, 2}
    assert [child.id for child in by_id[1].children] == [2]


@pytest.mark.parametrize(
    "child_label",
    [DocItemLabel.TABLE, DocItemLabel.PICTURE],
)
def test_container_does_not_wrap_nearly_identical_child(
    child_label: DocItemLabel,
) -> None:
    container = _cluster(1, DocItemLabel.FORM, (0, 0, 400, 400), confidence=0.65)
    child = _cluster(2, child_label, (2, 2, 398, 398), confidence=0.88)

    result = _process_special_clusters(container, child)

    assert [cluster.id for cluster in result] == [2]


def test_filtered_full_page_picture_does_not_remove_container() -> None:
    container = _cluster(1, DocItemLabel.FORM, (0, 0, 1000, 1000), confidence=0.8)
    picture = _cluster(2, DocItemLabel.PICTURE, (0, 0, 1000, 1000), confidence=0.8)

    result = _process_special_clusters(container, picture)

    assert [cluster.id for cluster in result] == [1]


def test_removed_picture_does_not_remove_container() -> None:
    container = _cluster(1, DocItemLabel.FORM, (0, 0, 100, 100), confidence=0.8)
    picture = _cluster(2, DocItemLabel.PICTURE, (10, 0, 110, 100), confidence=0.8)
    table = _cluster(3, DocItemLabel.TABLE, (20, 0, 120, 100), confidence=0.8)

    result = _process_special_clusters(container, picture, table)

    assert {cluster.id for cluster in result} == {1, 3}


def test_structured_child_uses_tightest_container() -> None:
    form = _cluster(1, DocItemLabel.FORM, (0, 0, 300, 300), confidence=0.7)
    key_value_region = _cluster(
        2, DocItemLabel.KEY_VALUE_REGION, (100, 100, 350, 350), confidence=0.7
    )
    table = _cluster(3, DocItemLabel.TABLE, (150, 150, 200, 200), confidence=0.9)
    text = _cluster(4, DocItemLabel.TEXT, (160, 160, 190, 190), confidence=0.9)

    result = _process_special_clusters(form, key_value_region, table, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert by_id[1].children == []
    assert [child.id for child in by_id[2].children] == [3]
    assert [child.id for child in by_id[3].children] == [4]


def test_direct_child_uses_tightest_container() -> None:
    form = _cluster(1, DocItemLabel.FORM, (0, 0, 300, 300), confidence=0.7)
    key_value_region = _cluster(
        2, DocItemLabel.KEY_VALUE_REGION, (100, 100, 350, 350), confidence=0.7
    )
    text = _cluster(3, DocItemLabel.TEXT, (150, 150, 200, 200), confidence=0.9)

    result = _process_special_clusters(form, key_value_region, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert by_id[1].children == []
    assert [child.id for child in by_id[2].children] == [3]
