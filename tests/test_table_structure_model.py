from types import SimpleNamespace
from typing import Any

import numpy
import pytest
import torch
from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
from docling.datamodel.pipeline_options import (
    TableStructureOptions,
    TableStructureV2Options,
)
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
)
from docling.models.stages.table_structure.table_structure_model_v2 import (
    TableStructureModelV2,
)


class _Backend:
    def __init__(self, page_image: Image.Image) -> None:
        self.page_image = page_image

    def is_valid(self) -> bool:
        return True

    def get_page_image(self, scale: float) -> Image.Image:
        assert scale == 1.0
        return self.page_image

    def get_segmented_page(self) -> None:
        return None


class _TablePredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[numpy.ndarray, list[dict[str, Any]]]] = []

    def multi_table_predict(
        self,
        page_input: dict[str, Any],
        table_boxes: list[list[float]],
        do_matching: bool,
    ) -> list[dict[str, Any]]:
        assert len(table_boxes) == 1
        assert do_matching is True
        self.calls.append((page_input["image"].copy(), page_input["tokens"].copy()))
        return [{"tf_responses": [], "predict_details": {}}]


def _text_cell(index: int, text: str, bbox: BoundingBox) -> TextCell:
    return TextCell(
        index=index,
        text=text,
        orig=text,
        from_ocr=False,
        rect=BoundingRectangle.from_bounding_box(bbox),
    )


def _nested_table_page() -> tuple[Page, Cluster, Cluster]:
    parent_cell = _text_cell(1, "parent", BoundingBox(l=1, t=1, r=3, b=3))
    nested_cell = _text_cell(2, "nested", BoundingBox(l=5, t=5, r=7, b=7))
    parent = Cluster(
        id=1,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=0, t=0, r=9.6, b=10),
        cells=[parent_cell, nested_cell],
    )
    nested = Cluster(
        id=2,
        label=DocItemLabel.TABLE,
        bbox=BoundingBox(l=4, t=4, r=10, b=8),
        cells=[nested_cell],
    )
    page = Page(page_no=1, size=Size(width=10, height=10))
    page._backend = _Backend(  # type: ignore[assignment]
        Image.new("RGB", (10, 10), "black")
    )
    page.predictions.layout = LayoutPrediction(clusters=[parent, nested])
    return page, parent, nested


@pytest.mark.parametrize(
    ("coverage_threshold", "nested_is_masked"),
    [(0.9, True), (0.95, False)],
)
def test_nested_table_threshold_controls_parent_prediction_input(
    coverage_threshold: float, nested_is_masked: bool
) -> None:
    page, parent, nested = _nested_table_page()

    model = TableStructureModel(
        enabled=False,
        artifacts_path=None,
        options=TableStructureOptions(
            rich_cell_element_coverage_threshold=coverage_threshold
        ),
        accelerator_options=AcceleratorOptions(),
    )
    model.scale = 1.0
    predictor = _TablePredictor()
    model.tf_predictor = predictor  # type: ignore[attr-defined]

    [prediction] = model.predict_tables(SimpleNamespace(timings={}), [page])  # type: ignore[arg-type]

    assert prediction.table_map.keys() == {parent.id, nested.id}
    assert len(predictor.calls) == 2
    parent_image, parent_tokens = predictor.calls[0]
    nested_image, nested_tokens = predictor.calls[1]
    expected_nested_pixel = [255, 255, 255] if nested_is_masked else [0, 0, 0]
    assert parent_image[6, 6].tolist() == expected_nested_pixel
    assert parent_image[2, 2].tolist() == [0, 0, 0]
    expected_parent_tokens = ["parent"] if nested_is_masked else ["parent", "nested"]
    assert [token["text"] for token in parent_tokens] == expected_parent_tokens
    assert nested_image[6, 6].tolist() == [0, 0, 0]
    assert [token["text"] for token in nested_tokens] == ["nested"]


def test_table_structure_v2_masks_nested_table_only_in_parent() -> None:
    page, parent, nested = _nested_table_page()
    captured_images: list[Image.Image] = []

    def capture_image(image: Image.Image) -> torch.Tensor:
        captured_images.append(image.copy())
        return torch.zeros((3, 8, 8))

    model = TableStructureModelV2(
        enabled=False,
        artifacts_path=None,
        options=TableStructureV2Options(),
        accelerator_options=AcceleratorOptions(),
    )
    model.scale = 1.0
    model.device = "cpu"
    model.transform = capture_image  # type: ignore[attr-defined]
    model.model = SimpleNamespace(  # type: ignore[attr-defined]
        generate=lambda *_args, **_kwargs: {
            "generated_ids": torch.tensor([[1]]),
            "predicted_bboxes": torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        }
    )
    model.tokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        decode=lambda _ids: "<fcel>"
    )

    [prediction] = model.predict_tables(SimpleNamespace(timings={}), [page])  # type: ignore[arg-type]

    assert prediction.table_map.keys() == {parent.id, nested.id}
    assert captured_images[0].getpixel((6, 6)) == (255, 255, 255)
    assert captured_images[1].getpixel((2, 2)) == (0, 0, 0)
    assert prediction.table_map[parent.id].table_cells[0].text == "parent"
    assert prediction.table_map[nested.id].table_cells[0].text == "nested"
