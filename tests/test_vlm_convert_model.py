from types import SimpleNamespace

from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersPromptStyle,
)
from docling.datamodel.stage_model_specs import VlmModelSpec
from docling.datamodel.vlm_engine_options import (
    AutoInlineVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.models.inference_engines.vlm.base import VlmEngineOutput, VlmEngineType
from docling.models.stages.vlm_convert.vlm_convert_model import VlmConvertModel


class _CapturingEngine:
    def __init__(self) -> None:
        self.batches = []

    def predict_batch(self, batch):
        self.batches.append(batch)
        return [
            VlmEngineOutput(text=f"output-{index}", stop_reason="unspecified")
            for index, _input in enumerate(batch)
        ]

    def cleanup(self) -> None:
        return None


def _build_model() -> tuple[VlmConvertModel, VlmModelSpec]:
    model_spec = VlmModelSpec(
        name="Test Model",
        default_repo_id="org/model",
        prompt="Convert this page to docling.",
        response_format=ResponseFormat.DOCTAGS,
        temperature=0.4,
        max_new_tokens=128,
        stop_strings=["</doctag>"],
        extra_generation_config={"top_p": 0.9},
    )

    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = _CapturingEngine()
    model.options = VlmConvertOptions(
        model_spec=model_spec,
        engine_options=AutoInlineVlmEngineOptions(),
    )
    return model, model_spec


def test_process_images_uses_configured_generation_settings() -> None:
    model, model_spec = _build_model()

    outputs = list(
        model.process_images(
            [Image.new("RGB", (8, 8), "white")],
            prompt="Use this prompt instead.",
        )
    )

    assert [prediction.text for prediction in outputs] == ["output-0"]

    batch = model.engine.batches[-1]
    assert len(batch) == 1

    engine_input = batch[0]
    assert engine_input.prompt == "Use this prompt instead."
    assert engine_input.temperature == model_spec.temperature
    assert engine_input.max_new_tokens == model_spec.max_new_tokens
    assert engine_input.stop_strings == model_spec.stop_strings
    assert engine_input.extra_generation_config == model_spec.extra_generation_config


def test_call_uses_model_spec_generation_settings() -> None:
    model, model_spec = _build_model()

    image = Image.new("RGB", (8, 8), "black")
    page = Page(page_no=1)
    page._image_cache = {1.0: image}
    page._default_image_scale = 1.0

    outputs = list(model(SimpleNamespace(timings={}), [page]))

    assert outputs == [page]
    assert page.predictions.vlm_response is not None
    assert page.predictions.vlm_response.text == "output-0"

    batch = model.engine.batches[-1]
    assert len(batch) == 1

    engine_input = batch[0]
    assert engine_input.prompt == model_spec.prompt
    assert engine_input.temperature == model_spec.temperature
    assert engine_input.max_new_tokens == model_spec.max_new_tokens
    assert engine_input.stop_strings == model_spec.stop_strings
    assert engine_input.extra_generation_config == model_spec.extra_generation_config


def test_process_images_merges_transformers_override_runtime_input_config() -> None:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = _CapturingEngine()
    model.options = VlmConvertOptions.from_preset(
        "got_ocr",
        engine_options=TransformersVlmEngineOptions(),
    )

    list(
        model.process_images(
            [Image.new("RGB", (8, 8), "white")],
            prompt="ignored",
        )
    )

    engine_input = model.engine.batches[-1][0]
    assert (
        engine_input.extra_generation_config["transformers_prompt_style"]
        == TransformersPromptStyle.NONE
    )
    assert engine_input.extra_generation_config["extra_processor_kwargs"] == {
        "format": True
    }
    assert "transformers_model_type" not in engine_input.extra_generation_config


def test_process_images_flattens_nested_override_generation_config() -> None:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = _CapturingEngine()
    model.options = VlmConvertOptions.from_preset(
        "phi4",
        engine_options=TransformersVlmEngineOptions(),
    )

    list(
        model.process_images(
            [Image.new("RGB", (8, 8), "white")],
            prompt="ignored",
        )
    )

    engine_input = model.engine.batches[-1][0]
    assert engine_input.extra_generation_config["num_logits_to_keep"] == 0
    assert "extra_generation_config" not in engine_input.extra_generation_config


def test_process_images_uses_selected_auto_inline_engine_for_runtime_input_config() -> None:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.enabled = True
    model.engine = _CapturingEngine()
    model.engine.selected_engine_type = VlmEngineType.TRANSFORMERS
    model.options = VlmConvertOptions.from_preset(
        "dolphin",
        engine_options=AutoInlineVlmEngineOptions(),
    )

    list(
        model.process_images(
            [Image.new("RGB", (8, 8), "white")],
            prompt="ignored",
        )
    )

    engine_input = model.engine.batches[-1][0]
    assert (
        engine_input.extra_generation_config["transformers_prompt_style"]
        == TransformersPromptStyle.RAW
    )
