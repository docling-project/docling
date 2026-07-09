from collections.abc import Iterable

import pytest
from PIL import Image

from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions
from docling.datamodel.pipeline_options_vlm_model import ResponseFormat
from docling.datamodel.stage_model_specs import EngineModelConfig, VlmModelSpec
from docling.datamodel.vlm_engine_options import (
    AutoInlineVlmEngineOptions,
    TransformersVlmEngineOptions,
    VlmEngineType,
)
from docling.models.inference_engines.vlm import VlmEngineInput, VlmEngineOutput
from docling.models.stages.picture_description.picture_description_vlm_engine_model import (
    PictureDescriptionVlmEngineModel,
)

pytestmark = pytest.mark.ml_vlm


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


def test_engine_picture_description_falls_back_to_model_spec_defaults() -> None:
    options = _build_options(temperature=0.1, max_new_tokens=300)
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()

    list(model._annotate_images([Image.new("RGB", (8, 8), "white")]))

    sent_input = model.engine.received_inputs[0]
    assert sent_input.max_new_tokens == 300
    assert sent_input.temperature == 0.1


def test_engine_picture_description_forwards_stop_strings_and_extra_config() -> None:
    options = _build_options(
        stop_strings=["<|im_end|>"],
        extra_generation_config={"skip_special_tokens": True},
    )
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()

    list(model._annotate_images([Image.new("RGB", (8, 8), "white")]))

    sent_input = model.engine.received_inputs[0]
    assert sent_input.stop_strings == ["<|im_end|>"]
    assert sent_input.extra_generation_config == {"skip_special_tokens": True}


def test_engine_picture_description_resolves_auto_inline_engine_type() -> None:
    options = _build_options(
        engine_overrides={
            VlmEngineType.MLX: EngineModelConfig(extra_config={"mlx_only_key": "x"}),
        }
    )
    options.engine_options = AutoInlineVlmEngineOptions()

    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()
    model.engine.selected_engine_type = (
        VlmEngineType.MLX
    )  # Engine hat sich für MLX entschieden

    resolved = model._resolve_runtime_engine_type()
    assert resolved == VlmEngineType.MLX


def test_engine_picture_description_skips_empty_batch() -> None:
    options = _build_options()
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()

    assert list(model._annotate_images([])) == []
    assert model.engine.received_inputs == []


def test_engine_picture_description_raises_when_engine_not_initialized() -> None:
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = _build_options()
    model.engine = None

    with pytest.raises(RuntimeError, match="Engine not initialized"):
        list(model._annotate_images([Image.new("RGB", (8, 8), "white")]))


def test_engine_picture_description_yields_empty_strings_on_engine_error() -> None:
    class _FailingEngine(_DummyEngine):
        def predict_batch(self, inputs):
            raise RuntimeError("engine exploded")

    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = _build_options()
    model.engine = _FailingEngine()

    images = [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "black")]
    outputs = list(model._annotate_images(images))

    # Batch-Alignment muss erhalten bleiben: gleiche Anzahl wie Input, leere Strings statt Crash
    assert outputs == ["", ""]


def test_engine_picture_description_shares_config_across_batch() -> None:
    options = _build_options(temperature=0.2, max_new_tokens=500)
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()

    images = [Image.new("RGB", (8, 8), "white") for _ in range(3)]
    list(model._annotate_images(images))

    assert len(model.engine.received_inputs) == 3
    assert all(i.max_new_tokens == 500 for i in model.engine.received_inputs)
    assert all(i.temperature == 0.2 for i in model.engine.received_inputs)


def test_engine_picture_description_forwards_generation_config() -> None:
    options = PictureDescriptionVlmEngineOptions.from_preset(
        "smolvlm",
        generation_config={
            "max_new_tokens": 1234,
            "temperature": 0.42,
            "do_sample": True,
        },
    )
    model = PictureDescriptionVlmEngineModel.__new__(PictureDescriptionVlmEngineModel)
    model.options = options
    model.engine = _DummyEngine()

    images = [Image.new("RGB", (8, 8), "white")]
    outputs = list(model._annotate_images(images))

    assert outputs == ["description 0"]
    sent_input = model.engine.received_inputs[0]
    assert sent_input.max_new_tokens == 1234
    assert sent_input.temperature == 0.42
