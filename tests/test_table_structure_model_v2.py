import torch
from docling_core.types.doc import BoundingBox
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.models.stages.table_structure.table_structure_model_v2 import (
    TableStructureModelV2,
)


def _text_cell(index: int, bbox: BoundingBox, text: str | None = None) -> TextCell:
    value = text if text is not None else f"cell-{index}"
    return TextCell(
        index=index,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=value,
        orig=value,
        from_ocr=False,
    )


def _reference_matches(
    bboxes: list[BoundingBox],
    text_cells: list[TextCell],
    textcell_overlap: float,
) -> list[str]:
    matches = []
    for bbox in bboxes:
        overlapping = []
        for text_cell in text_cells:
            cell_bbox = text_cell.rect.to_bounding_box()
            if cell_bbox.get_intersection_bbox(bbox) is not None:
                if cell_bbox.intersection_over_self(bbox) > textcell_overlap:
                    overlapping.append(text_cell.text.strip())
        matches.append(" ".join(overlapping))
    return matches


def test_match_texts_matches_exhaustive_selection():
    bboxes = [
        BoundingBox(l=0, t=0, r=100, b=40),
        BoundingBox(l=80, t=0, r=180, b=40),
        BoundingBox(l=240, t=0, r=300, b=40),
        BoundingBox(l=10, t=80, r=70, b=140),
    ]
    text_cells = [
        _text_cell(0, BoundingBox(l=5, t=5, r=25, b=20), " A "),
        _text_cell(1, BoundingBox(l=35, t=5, r=70, b=20), "B"),
        _text_cell(2, BoundingBox(l=85, t=5, r=130, b=20), "C"),
        _text_cell(3, BoundingBox(l=150, t=5, r=175, b=20), "D"),
        _text_cell(4, BoundingBox(l=210, t=5, r=230, b=20), "outside"),
        _text_cell(5, BoundingBox(l=15, t=90, r=45, b=120), " "),
        _text_cell(6, BoundingBox(l=45, t=90, r=65, b=120), "E"),
    ]

    model = object.__new__(TableStructureModelV2)

    assert model._match_texts(bboxes, text_cells, 0.3) == _reference_matches(
        bboxes,
        text_cells,
        0.3,
    )


def test_match_texts_handles_unordered_bbox_coordinates() -> None:
    text_cells = [
        _text_cell(0, BoundingBox(l=5, t=5, r=25, b=20), "A"),
    ]
    model = object.__new__(TableStructureModelV2)

    assert model._match_texts(
        [BoundingBox(l=25, t=20, r=5, b=5)],
        text_cells,
        0.3,
    ) == ["A"]


def test_build_table_cells_orders_model_bbox_coordinates() -> None:
    model = object.__new__(TableStructureModelV2)

    cell_data, num_rows, num_cols = model._build_table_cells(
        ["fcel"],
        torch.tensor([[0.75, 0.5, 0.25, 0.0]]),
        [100, 200, 200, 300],
    )

    assert cell_data[0]["bbox"] == {
        "l": 125.0,
        "t": 200.0,
        "r": 175.0,
        "b": 250.0,
    }
    assert num_rows == 1
    assert num_cols == 1
