from collections.abc import Iterable

import pytest
from PIL import Image

from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import (
    TransformersVlmEngineOptions,
)
from docling.models.inference_engines.vlm import VlmEngineInput, VlmEngineOutput
from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
    PictureDescriptionVlmEngineModel,
)


class _DummyEngine:
    def __init__(self):
        self.received_inputs: list[VlmEngineInput] = []

    def predict_batch(self, inputs: Iterable[VlmEngineInput]):
        self.received_inputs = list(inputs)
        return [
            VlmEngineOutput(text=f"description {i}", stop_reason="end_of_sequence")
            for i in range(len(self.received_inputs))
        ]

    def cleanup(self):
        pass


@pytest.fixture
def create_dummy_model():
    def _make(
        options: PictureDescriptionVlmEngineOptions,
    ) -> PictureDescriptionVlmEngineModel:
        model = PictureDescriptionVlmEngineModel.__new__(
            PictureDescriptionVlmEngineModel
        )
        model.options = options
        model.engine = _DummyEngine()
        return model

    return _make


def _build_options(**model_spec_overrides) -> PictureDescriptionVlmEngineOptions:
    defaults = dict(
        name="test-model",
        default_repo_id="org/test-model",
        prompt="Describe this image.",
        response_format=ResponseFormat.PLAINTEXT,
        temperature=0.1,
        max_new_tokens=300,
    )
    defaults.update(model_spec_overrides)
    return PictureDescriptionVlmEngineOptions(
        model_spec=VlmModelSpec(**defaults),
        engine_options=TransformersVlmEngineOptions(),
        prompt="Describe this image.",
        generation_config={},
    )


def test_engine_picture_description_falls_back_to_model_spec_defaults(
    create_dummy_model,
) -> None:
    options = _build_options(temperature=0.1, max_new_tokens=300)
    model = create_dummy_model(options)

    list(model._annotate_images([Image.new("RGB", (8, 8), "white")]))

    sent_input = model.engine.received_inputs[0]
    assert sent_input.max_new_tokens == 300
    assert sent_input.temperature == 0.1


def test_engine_picture_description_forwards_generation_config(
    create_dummy_model,
) -> None:
    options = PictureDescriptionVlmEngineOptions.from_preset(
        "smolvlm",
        generation_config={
            "max_new_tokens": 1234,
            "temperature": 0.42,
            "do_sample": True,
        },
    )
    model = create_dummy_model(options)

    images = [Image.new("RGB", (8, 8), "white")]
    outputs = list(model._annotate_images(images))

    assert outputs == ["description 0"]
    sent_input = model.engine.received_inputs[0]
    assert sent_input.max_new_tokens == 1234
    assert sent_input.temperature == 0.42
