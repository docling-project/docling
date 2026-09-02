# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
from docling_core.types.doc import DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import BoundingBox, Cluster, Page
from docling.datamodel.pipeline_options import LayoutOptions, LayoutPostprocessorOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor


def _text_cell(
    index: int, bbox: BoundingBox | None = None, text: str | None = None
) -> TextCell:
    value = f"cell-{index}" if text is None else text
    bbox = bbox or BoundingBox(l=0, t=0, r=1, b=1)
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=value,
        orig=value,
        from_ocr=False,
    )


def _cluster(
    index: int,
    bbox: BoundingBox,
    label: DocItemLabel = DocItemLabel.TEXT,
    confidence: float = 0.9,
) -> Cluster:
    return Cluster(
        id=index,
        label=label,
        bbox=bbox,
        confidence=confidence,
    )


class _PageStub:
    def __init__(self, cells: list[TextCell]) -> None:
        self.cells = cells
        self.size = Size(width=400, height=400)


def _reference_assignments(
    clusters: list[Cluster], cells: list[TextCell], min_overlap: float = 0.2
) -> dict[int, list[int]]:
    assignments = {cluster.id: [] for cluster in clusters}
    for cell in cells:
        if not cell.text.strip():
            continue

        cell_bbox = cell.rect.to_bounding_box()
        if cell_bbox.area() <= 0:
            continue

        best_overlap = min_overlap
        best_cluster = None
        for cluster in clusters:
            overlap_ratio = cell_bbox.intersection_over_self(cluster.bbox)
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_cluster = cluster

        if best_cluster is not None:
            assignments[best_cluster.id].append(cell.index)

    return assignments


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


def test_assign_cells_to_clusters_matches_exhaustive_selection() -> None:
    clusters = [
        _cluster(0, BoundingBox(l=0, t=0, r=100, b=100)),
        _cluster(1, BoundingBox(l=40, t=40, r=140, b=140)),
        _cluster(2, BoundingBox(l=300, t=300, r=360, b=360)),
        _cluster(3, BoundingBox(l=0, t=0, r=100, b=100)),
    ]
    cells = [
        _text_cell(0, BoundingBox(l=10, t=10, r=30, b=30)),
        _text_cell(1, BoundingBox(l=50, t=50, r=80, b=80)),
        _text_cell(2, BoundingBox(l=310, t=310, r=350, b=350)),
        _text_cell(3, BoundingBox(l=180, t=180, r=190, b=190)),
        _text_cell(4, BoundingBox(l=0, t=0, r=0, b=10)),
        _text_cell(5, BoundingBox(l=20, t=20, r=40, b=40), text=" "),
    ]
    page = _PageStub(cells)

    postprocessor = LayoutPostprocessor(page, clusters, LayoutPostprocessorOptions())
    assigned = postprocessor._assign_cells_to_clusters(clusters)

    assert {
        cluster.id: [cell.index for cell in cluster.cells] for cluster in assigned
    } == _reference_assignments(clusters, cells)


def test_assign_cells_to_clusters_indexes_passed_clusters() -> None:
    cells = [_text_cell(0, BoundingBox(l=10, t=10, r=30, b=30))]
    stale_clusters = [_cluster(0, BoundingBox(l=300, t=300, r=360, b=360))]
    page = _PageStub(cells)
    postprocessor = LayoutPostprocessor(
        page, stale_clusters, LayoutPostprocessorOptions()
    )
    current_clusters = [_cluster(0, BoundingBox(l=0, t=0, r=100, b=100))]

    assigned = postprocessor._assign_cells_to_clusters(current_clusters)

    assert [cell.index for cell in assigned[0].cells] == [0]


def test_cross_type_overlaps_removes_picture_coinciding_with_table() -> None:
    # The layout model proposes the same region as both a PICTURE and a TABLE.
    # The PICTURE (near-identical bbox, high IoU) must be removed; the TABLE kept.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(
        1,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.TABLE,
        confidence=0.72,
    )
    picture = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.PICTURE,
        confidence=0.81,
    )

    result = processor._handle_cross_type_overlaps([table, picture])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.PICTURE not in labels


def test_cross_type_overlaps_removes_picture_coinciding_with_document_index() -> None:
    # Same rule as PICTURE-vs-TABLE: a near-identical PICTURE against a
    # DOCUMENT_INDEX must be dropped so the richer structured label survives.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    doc_index = _cluster(
        1,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.DOCUMENT_INDEX,
        confidence=0.72,
    )
    picture = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.PICTURE,
        confidence=0.81,
    )

    result = processor._handle_cross_type_overlaps([doc_index, picture])

    labels = {c.label for c in result}
    assert DocItemLabel.DOCUMENT_INDEX in labels
    assert DocItemLabel.PICTURE not in labels


def test_cross_type_overlaps_keeps_picture_not_overlapping_table() -> None:
    # A genuine figure elsewhere on the page must be preserved.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE)
    picture = _cluster(2, BoundingBox(l=10, t=300, r=200, b=450), DocItemLabel.PICTURE)

    result = processor._handle_cross_type_overlaps([table, picture])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_keeps_both_when_picture_clearly_more_confident() -> None:
    # The near-tie label preference only fires when confidences are within 0.1.
    # A near-identical PICTURE that is clearly more confident than the TABLE is
    # outside our envelope, so we leave both in place. Downstream (or a future
    # dedicated PICTURE/TABLE sieve) can decide what to do with the pair.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.55
    )
    picture = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.PICTURE,
        confidence=0.95,
    )

    result = processor._handle_cross_type_overlaps([table, picture])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_keeps_small_picture_inside_table() -> None:
    # A small figure fully contained in a large table (high containment but low IoU)
    # must NOT be removed -- only a near-coinciding picture is a true mislabel.
    processor = object.__new__(LayoutPostprocessor)
    processor.regular_clusters = []

    table = _cluster(1, BoundingBox(l=0, t=0, r=400, b=400), DocItemLabel.TABLE)
    small_picture = _cluster(
        2, BoundingBox(l=10, t=10, r=60, b=60), DocItemLabel.PICTURE
    )

    result = processor._handle_cross_type_overlaps([table, small_picture])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def _bbox(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(l=left, t=top, r=right, b=bottom)


def test_cross_type_overlaps_removes_kvregion_coinciding_with_table() -> None:
    # A KEY_VALUE_REGION that nearly covers the same area as a TABLE should be
    # dropped in favour of the TABLE. Previously this check was unreachable because
    # TABLE is in WRAPPER_TYPES and therefore never appears in regular_clusters.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.85
    )
    kvr = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.KEY_VALUE_REGION,
        confidence=0.80,
    )

    result = processor._handle_cross_type_overlaps([table, kvr])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.KEY_VALUE_REGION not in labels


def test_cross_type_overlaps_keeps_kvregion_not_overlapping_table() -> None:
    # A KEY_VALUE_REGION on a different part of the page should not be affected.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE)
    kvr = _cluster(
        2, BoundingBox(l=10, t=300, r=200, b=450), DocItemLabel.KEY_VALUE_REGION
    )

    result = processor._handle_cross_type_overlaps([table, kvr])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_removes_form_coinciding_with_table() -> None:
    # A FORM whose bbox is near-identical to a TABLE should lose to the structured
    # TABLE.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.80
    )
    form = _cluster(
        2, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.FORM, confidence=0.75
    )

    result = processor._handle_cross_type_overlaps([table, form])

    labels = {c.label for c in result}
    assert DocItemLabel.TABLE in labels
    assert DocItemLabel.FORM not in labels


def test_cross_type_overlaps_keeps_form_containing_small_table() -> None:
    # A TABLE that only overlaps a portion of a larger FORM must NOT drop the FORM.
    # This is the guard-rail against the previous intersection_over_self behaviour,
    # which would have removed the FORM whenever a TABLE landed anywhere inside it.
    processor = object.__new__(LayoutPostprocessor)

    form = _cluster(
        1, BoundingBox(l=0, t=0, r=400, b=400), DocItemLabel.FORM, confidence=0.80
    )
    table = _cluster(
        2, BoundingBox(l=20, t=20, r=120, b=120), DocItemLabel.TABLE, confidence=0.80
    )

    result = processor._handle_cross_type_overlaps([form, table])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_keeps_form_when_clearly_more_confident() -> None:
    # Label preference only applies when the two heads are similarly confident.
    # A near-identical FORM proposal that is clearly more confident than the TABLE
    # must survive.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.55
    )
    form = _cluster(
        2, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.FORM, confidence=0.95
    )

    result = processor._handle_cross_type_overlaps([table, form])

    ids = {c.id for c in result}
    assert ids == {1, 2}


def test_cross_type_overlaps_keeps_document_index_over_coinciding_table() -> None:
    # DOCUMENT_INDEX carries the "this is an index/TOC" semantic and is treated as
    # a table downstream anyway, so a near-identical TABLE proposal must lose.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.80
    )
    doc_index = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.DOCUMENT_INDEX,
        confidence=0.75,
    )

    result = processor._handle_cross_type_overlaps([table, doc_index])

    labels = {c.label for c in result}
    assert DocItemLabel.DOCUMENT_INDEX in labels
    assert DocItemLabel.TABLE not in labels


def test_cross_type_overlaps_keeps_table_when_clearly_more_confident_than_docindex() -> (
    None
):
    # Same confidence gate in the DocIndex direction: a clearly-more-confident
    # TABLE proposal is not overridden by a low-confidence DOCUMENT_INDEX.
    processor = object.__new__(LayoutPostprocessor)

    table = _cluster(
        1, BoundingBox(l=10, t=10, r=200, b=150), DocItemLabel.TABLE, confidence=0.95
    )
    doc_index = _cluster(
        2,
        BoundingBox(l=10, t=10, r=200, b=150),
        DocItemLabel.DOCUMENT_INDEX,
        confidence=0.55,
    )

    result = processor._handle_cross_type_overlaps([table, doc_index])

    ids = {c.id for c in result}
    assert ids == {1, 2}


@pytest.mark.parametrize(
    "container_label",
    [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION],
)
def test_container_nests_structured_children(container_label: DocItemLabel) -> None:
    container = _cluster(1, _bbox(0, 0, 400, 400), container_label, confidence=0.65)
    table = _cluster(2, _bbox(10, 10, 150, 100), DocItemLabel.TABLE, confidence=0.88)
    picture = _cluster(
        3, _bbox(200, 200, 300, 300), DocItemLabel.PICTURE, confidence=0.82
    )
    text = _cluster(4, _bbox(20, 20, 140, 80), DocItemLabel.TEXT, confidence=0.9)

    result = _process_special_clusters(container, table, picture, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert set(by_id) == {1, 2, 3}
    assert [child.id for child in by_id[1].children] == [2, 3]
    assert [child.id for child in by_id[2].children] == [4]
    assert by_id[3].children == []


def test_container_direct_text_remains_available_for_reading_order() -> None:
    container = _cluster(1, _bbox(0, 0, 400, 400), DocItemLabel.FORM, confidence=0.8)
    caption = _cluster(
        2, _bbox(10, 300, 300, 350), DocItemLabel.CAPTION, confidence=0.8
    )

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
    container = _cluster(1, _bbox(0, 0, 400, 400), DocItemLabel.FORM, confidence=0.65)
    child = _cluster(2, _bbox(2, 2, 398, 398), child_label, confidence=0.88)

    result = _process_special_clusters(container, child)

    assert [cluster.id for cluster in result] == [2]


def test_filtered_full_page_picture_does_not_remove_container() -> None:
    container = _cluster(1, _bbox(0, 0, 1000, 1000), DocItemLabel.FORM, confidence=0.8)
    picture = _cluster(2, _bbox(0, 0, 1000, 1000), DocItemLabel.PICTURE, confidence=0.8)

    result = _process_special_clusters(container, picture)

    assert [cluster.id for cluster in result] == [1]


def test_removed_picture_does_not_remove_container() -> None:
    container = _cluster(1, _bbox(0, 0, 100, 100), DocItemLabel.FORM, confidence=0.8)
    picture = _cluster(2, _bbox(10, 0, 110, 100), DocItemLabel.PICTURE, confidence=0.8)
    table = _cluster(3, _bbox(20, 0, 120, 100), DocItemLabel.TABLE, confidence=0.8)

    result = _process_special_clusters(container, picture, table)

    assert {cluster.id for cluster in result} == {1, 3}


def test_structured_child_uses_tightest_container() -> None:
    form = _cluster(1, _bbox(0, 0, 300, 300), DocItemLabel.FORM, confidence=0.7)
    key_value_region = _cluster(
        2, _bbox(100, 100, 350, 350), DocItemLabel.KEY_VALUE_REGION, confidence=0.7
    )
    table = _cluster(3, _bbox(150, 150, 200, 200), DocItemLabel.TABLE, confidence=0.9)
    text = _cluster(4, _bbox(160, 160, 190, 190), DocItemLabel.TEXT, confidence=0.9)

    result = _process_special_clusters(form, key_value_region, table, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert by_id[1].children == []
    assert [child.id for child in by_id[2].children] == [3]
    assert [child.id for child in by_id[3].children] == [4]


def test_direct_child_uses_tightest_container() -> None:
    form = _cluster(1, _bbox(0, 0, 300, 300), DocItemLabel.FORM, confidence=0.7)
    key_value_region = _cluster(
        2, _bbox(100, 100, 350, 350), DocItemLabel.KEY_VALUE_REGION, confidence=0.7
    )
    text = _cluster(3, _bbox(150, 150, 200, 200), DocItemLabel.TEXT, confidence=0.9)

    result = _process_special_clusters(form, key_value_region, text)

    by_id = {cluster.id: cluster for cluster in result}
    assert by_id[1].children == []
    assert [child.id for child in by_id[2].children] == [3]
