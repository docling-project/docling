import os

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, ImageDraw

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import MistralOcrOptions
from docling.models.factories import get_ocr_factory
from docling.models.stages.ocr.mistral_ocr_model import (
    MistralOcrModel,
    _MistralOcrResponse,
)

pytestmark = pytest.mark.ml_ocr


def _make_model(*, enabled: bool = True) -> MistralOcrModel:
    return MistralOcrModel(
        enabled=enabled,
        artifacts_path=None,
        options=MistralOcrOptions(api_key="test-key", scale=2.0),
        accelerator_options=AcceleratorOptions(),
    )


def test_mistral_ocr_backend_registration() -> None:
    factory = get_ocr_factory(allow_external_plugins=False)

    model = factory.create_instance(
        options=MistralOcrOptions(),
        enabled=False,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
    )

    assert isinstance(model, MistralOcrModel)


def test_mistral_ocr_requires_api_key_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="requires an API key"):
        MistralOcrModel(
            enabled=True,
            artifacts_path=None,
            options=MistralOcrOptions(),
            accelerator_options=AcceleratorOptions(),
        )


def test_mistral_ocr_reads_api_key_from_configured_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCLING_TEST_MISTRAL_KEY", "env-key")

    model = MistralOcrModel(
        enabled=True,
        artifacts_path=None,
        options=MistralOcrOptions(api_key_env_var="DOCLING_TEST_MISTRAL_KEY"),
        accelerator_options=AcceleratorOptions(),
    )

    assert model._api_key == "env-key"


def test_mistral_ocr_payload_requests_blocks_and_separate_images() -> None:
    model = _make_model()
    image = Image.new("RGB", (8, 6), "white")

    payload = model._request_payload(image)

    assert payload["model"] == "mistral-ocr-4-0"
    assert payload["include_blocks"] is True
    assert payload["include_image_base64"] is True
    assert payload["confidence_scores_granularity"] == "word"
    assert payload["table_format"] == "html"
    assert payload["document"]["type"] == "image_url"
    assert payload["document"]["image_url"].startswith("data:image/png;base64,")


def test_mistral_ocr_blocks_map_to_cells_and_skip_image_blocks() -> None:
    model = _make_model()
    response = _MistralOcrResponse.model_validate(
        {
            "pages": [
                {
                    "markdown": "Invoice 123\n\n![img-0.jpeg](img-0.jpeg)\n\nTotal $42",
                    "confidence_scores": {
                        "average_page_confidence_score": 0.7,
                        "word_confidence_scores": [
                            {"text": "Invoice", "confidence": 0.9, "start_index": 0},
                            {"text": " 123", "confidence": 0.8, "start_index": 7},
                            {"text": "Total", "confidence": 0.6, "start_index": 37},
                            {"text": " $42", "confidence": 0.5, "start_index": 42},
                        ],
                    },
                    "blocks": [
                        {
                            "type": "text",
                            "content": "Invoice 123",
                            "top_left_x": 20,
                            "top_left_y": 30,
                            "bottom_right_x": 220,
                            "bottom_right_y": 70,
                        },
                        {
                            "type": "image",
                            "content": "![img-0.jpeg](img-0.jpeg)",
                            "top_left_x": 30,
                            "top_left_y": 90,
                            "bottom_right_x": 180,
                            "bottom_right_y": 190,
                        },
                        {
                            "type": "text",
                            "content": "Total $42",
                            "top_left_x": 40,
                            "top_left_y": 220,
                            "bottom_right_x": 190,
                            "bottom_right_y": 260,
                        },
                    ],
                }
            ]
        }
    )
    ocr_rect = BoundingBox(
        l=10,
        t=100,
        r=410,
        b=500,
        coord_origin=CoordOrigin.TOPLEFT,
    )

    cells = model._response_to_cells(response, ocr_rect=ocr_rect, start_index=3)

    assert [cell.text for cell in cells] == ["Invoice 123", "Total $42"]
    assert [cell.index for cell in cells] == [3, 4]
    assert cells[0].confidence == pytest.approx(0.85)
    assert cells[0].rect.to_bounding_box().as_tuple() == pytest.approx(
        (20.0, 115.0, 120.0, 135.0)
    )
    assert cells[1].rect.to_bounding_box().as_tuple() == pytest.approx(
        (30.0, 210.0, 105.0, 230.0)
    )


@pytest.mark.skipif(
    not os.getenv("MISTRAL_API_KEY"),
    reason="Set MISTRAL_API_KEY to run the Mistral OCR integration test.",
)
def test_mistral_ocr_live_smoke() -> None:
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 90), "Invoice 12345", fill="black")
    draw.text((80, 170), "Widget $42.00", fill="black")

    model = MistralOcrModel(
        enabled=True,
        artifacts_path=None,
        options=MistralOcrOptions(),
        accelerator_options=AcceleratorOptions(),
    )
    response = model._request_ocr(image)
    cells = model._response_to_cells(
        response,
        ocr_rect=BoundingBox(
            l=0,
            t=0,
            r=image.width,
            b=image.height,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        start_index=0,
    )

    assert any("Invoice 12345" in cell.text for cell in cells)
