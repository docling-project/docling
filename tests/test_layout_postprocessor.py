from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.base_models import Cluster
from docling.datamodel.pipeline_options import LayoutOptions
from docling.utils.layout_postprocessor import LayoutPostprocessor


def _text_cell(index: int, bbox: BoundingBox, text: str | None = None) -> TextCell:
    value = text or f"cell-{index}"
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=value,
        orig=value,
        from_ocr=False,
    )


def _cluster(index: int, bbox: BoundingBox, confidence: float = 0.9) -> Cluster:
    return Cluster(
        id=index,
        label=DocItemLabel.TEXT,
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

        best_overlap = min_overlap
        best_cluster = None
        cell_bbox = cell.rect.to_bounding_box()
        if cell_bbox.area() <= 0:
            continue

        for cluster in clusters:
            overlap_ratio = cell_bbox.intersection_over_self(cluster.bbox)
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_cluster = cluster

        if best_cluster is not None:
            assignments[best_cluster.id].append(cell.index)

    return assignments


def test_assign_cells_to_clusters_matches_exhaustive_selection():
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

    postprocessor = LayoutPostprocessor(page, clusters, LayoutOptions())
    assigned = postprocessor._assign_cells_to_clusters(clusters)

    assert {
        cluster.id: [cell.index for cell in cluster.cells] for cluster in assigned
    } == _reference_assignments(clusters, cells)
