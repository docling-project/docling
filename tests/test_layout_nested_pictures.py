# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Sequence
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.object_detection import (
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.stages.layout import layout_object_detection_model

PICTURE_BOXES = [
    (0.05, 0.05, 0.65, 0.85),
    # The inner box contains the fixture's "maximum caseload" text line.
    (0.10, 0.33, 0.29, 0.35),
    (0.11, 0.33, 0.29, 0.35),
    (0.70, 0.70, 0.90, 0.85),
]


class _PictureDetectionEngine:
    def initialize(self) -> None:
        pass

    def get_label_mapping(self) -> dict[int, str]:
        return {0: "picture"}

    def predict_batch(
        self, inputs: Sequence[ObjectDetectionEngineInput]
    ) -> list[ObjectDetectionEngineOutput]:
        return [
            ObjectDetectionEngineOutput(
                label_ids=[0, 0, 0, 0],
                scores=[0.95, 0.90, 0.85, 0.92],
                bboxes=[
                    (
                        left * entry.image.width,
                        top * entry.image.height,
                        right * entry.image.width,
                        bottom * entry.image.height,
                    )
                    for left, top, right, bottom in PICTURE_BOXES
                ],
            )
            for entry in inputs
        ]


@pytest.mark.parametrize(
    ("layout_config", "expected_ids"),
    [
        pytest.param({}, [0, 3], id="default"),
        pytest.param({"keep_nested_pictures": True}, [0, 1, 3], id="opt-in"),
    ],
)
def test_nested_picture_items_from_pdf_layout(
    monkeypatch: pytest.MonkeyPatch,
    layout_config: dict[str, bool],
    expected_ids: list[int],
) -> None:
    monkeypatch.setattr(
        layout_object_detection_model,
        "create_object_detection_engine",
        lambda **kwargs: _PictureDetectionEngine(),
    )
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        layout_options=LayoutObjectDetectionOptions.model_validate(layout_config),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )

    result = converter.convert(
        Path("tests/data/pdf/sources/table_mislabeled_as_picture.pdf")
    )

    doc = result.document
    assert len(doc.pictures) == len(expected_ids)
    pictures_by_box = {}
    for picture in doc.pictures:
        prov = picture.prov[0]
        size = doc.pages[prov.page_no].size
        box = prov.bbox.to_top_left_origin(size.height)
        normalized_box = (
            box.l / size.width,
            box.t / size.height,
            box.r / size.width,
            box.b / size.height,
        )
        pictures_by_box[normalized_box] = picture
    actual_boxes = sorted(pictures_by_box)
    for actual, expected_id in zip(actual_boxes, expected_ids):
        assert actual == pytest.approx(PICTURE_BOXES[expected_id])
    shared_text = [
        item
        for item in doc.texts
        if item.text == "the maximum caseload per lawyer at one time?"
    ]
    assert len(shared_text) == 1
    owner_index = expected_ids.index(1) if 1 in expected_ids else 0
    assert (
        shared_text[0].parent.resolve(doc) is pictures_by_box[actual_boxes[owner_index]]
    )
    assert doc.export_to_markdown().count("<!-- image -->") == len(expected_ids)
