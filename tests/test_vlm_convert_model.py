# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from docling.datamodel.base_models import Page, VlmStopReason
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
from docling.models.stages.vlm_convert import vlm_convert_model
from docling.models.stages.vlm_convert.vlm_convert_model import (
    VlmConvertModel,
    _prediction_from_engine_output,
)


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
    page._image_cache = {model.options.scale: image}
    page._default_image_scale = model.options.scale

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


def test_process_images_shares_generation_template_across_batch() -> None:
    """Batch items get consistent generation settings without per-item reallocation."""
    model, model_spec = _build_model()

    list(
        model.process_images(
            [
                Image.new("RGB", (8, 8), "white"),
                Image.new("RGB", (8, 8), "black"),
                Image.new("RGB", (8, 8), "red"),
            ],
            prompt="Prompt.",
        )
    )

    batch = model.engine.batches[-1]
    assert len(batch) == 3

    first = batch[0]
    for engine_input in batch[1:]:
        assert engine_input.temperature == first.temperature
        assert engine_input.max_new_tokens == first.max_new_tokens
        assert engine_input.stop_strings == first.stop_strings
        assert engine_input.extra_generation_config == first.extra_generation_config

    assert first.stop_strings == model_spec.stop_strings
    assert first.extra_generation_config == model_spec.extra_generation_config


def test_process_images_uses_selected_auto_inline_engine_for_runtime_input_config() -> (
    None
):
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


def test_prediction_keeps_the_incomplete_stop_reasons() -> None:
    """A truncated or filtered call must stay reported as incomplete."""
    length = _prediction_from_engine_output(
        VlmEngineOutput(text="a", stop_reason="length")
    )
    filtered = _prediction_from_engine_output(
        VlmEngineOutput(text="", stop_reason="content_filter")
    )

    assert length.stop_reason == VlmStopReason.LENGTH
    assert filtered.stop_reason == VlmStopReason.CONTENT_FILTERED


def test_prediction_reports_a_completed_call() -> None:
    """A sequence that reached its end token is a complete output."""
    completed = _prediction_from_engine_output(
        VlmEngineOutput(text="a", stop_reason="end_of_sequence")
    )

    assert completed.stop_reason == VlmStopReason.END_OF_SEQUENCE


def test_prediction_reads_a_vllm_stop_as_a_completed_call() -> None:
    """vLLM reports stop for a sequence that emitted its end token."""
    completed = _prediction_from_engine_output(
        VlmEngineOutput(text="a", stop_reason="stop")
    )

    assert completed.stop_reason == VlmStopReason.END_OF_SEQUENCE


def test_del_ignores_cleanup_failure_at_interpreter_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = VlmConvertModel.__new__(VlmConvertModel)
    model.engine = MagicMock()
    model.engine.cleanup.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(vlm_convert_model, "_log", None)

    model.__del__()
