"""Test baidu/Unlimited-OCR VLM integration."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.stage_model_specs import EngineModelConfig
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType

pytestmark = pytest.mark.ml_vlm


class _FakeUnlimitedOcrModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def infer(self, tokenizer: object, **kwargs: Any) -> str:
        image_file = Path(kwargs["image_file"])
        output_path = Path(kwargs["output_path"])
        assert image_file.exists()
        assert output_path.parent.exists()
        self.calls.append(kwargs)
        return "recognized **markdown**"


def test_unlimited_ocr_preset_exists() -> None:
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "unlimited_ocr" in preset_ids

    preset = VlmConvertOptions.get_preset("unlimited_ocr")
    assert preset.preset_id == "unlimited_ocr"
    assert preset.name == "Unlimited-OCR"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.TRANSFORMERS

    spec = preset.model_spec
    assert spec.default_repo_id == "baidu/Unlimited-OCR"
    assert spec.response_format == ResponseFormat.MARKDOWN
    assert spec.trust_remote_code is True
    assert spec.max_new_tokens == 32768
    assert spec.supported_engines == {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.API,
        VlmEngineType.API_OPENAI,
    }


def test_unlimited_ocr_preset_engine_config() -> None:
    preset = VlmConvertOptions.get_preset("unlimited_ocr")
    spec = preset.model_spec

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == "baidu/Unlimited-OCR"
    assert tf_config.torch_dtype == "bfloat16"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL
    )
    assert (
        tf_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.RAW
    )
    assert tf_config.extra_config["transformers_custom_inference"] == "unlimited_ocr"
    assert tf_config.extra_config["unlimited_ocr_base_size"] == 1024
    assert tf_config.extra_config["unlimited_ocr_image_size"] == 640
    assert tf_config.extra_config["unlimited_ocr_crop_mode"] is True
    assert tf_config.extra_config["unlimited_ocr_no_repeat_ngram_size"] == 35
    assert tf_config.extra_config["unlimited_ocr_ngram_window"] == 128

    api_params = spec.get_api_params(VlmEngineType.API)
    assert api_params["model"] == "Unlimited-OCR"
    assert api_params["max_tokens"] == 32768
    assert api_params["images_config"] == {"image_mode": "gundam"}
    assert api_params["custom_params"] == {"ngram_size": 35, "window_size": 128}


def test_unlimited_ocr_legacy_specs() -> None:
    transformers_spec = vlm_model_specs.UNLIMITED_OCR_TRANSFORMERS
    assert transformers_spec.repo_id == "baidu/Unlimited-OCR"
    assert transformers_spec.inference_framework == InferenceFramework.TRANSFORMERS
    assert transformers_spec.response_format == ResponseFormat.MARKDOWN
    assert transformers_spec.trust_remote_code is True
    assert transformers_spec.transformers_model_type == TransformersModelType.AUTOMODEL
    assert transformers_spec.transformers_prompt_style == TransformersPromptStyle.RAW
    assert transformers_spec.torch_dtype == "bfloat16"
    assert transformers_spec.max_new_tokens == 32768
    assert (
        transformers_spec.extra_generation_config["transformers_custom_inference"]
        == "unlimited_ocr"
    )

    api_spec = vlm_model_specs.UNLIMITED_OCR_SGLANG_API
    assert str(api_spec.url).startswith("http://localhost:10000")
    assert api_spec.params["model"] == "Unlimited-OCR"
    assert api_spec.params["max_tokens"] == 32768
    assert api_spec.response_format == ResponseFormat.MARKDOWN


def test_unlimited_ocr_transformers_engine_calls_custom_infer() -> None:
    try:
        from docling.models.inference_engines.vlm.transformers_engine import (
            TransformersVlmEngine,
        )
    except (ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f"Transformers engine unavailable in this environment: {exc}")

    fake_model = _FakeUnlimitedOcrModel()
    engine = TransformersVlmEngine(
        options=TransformersVlmEngineOptions(),
        accelerator_options=AcceleratorOptions(),
        artifacts_path=None,
        model_config=None,
    )
    engine._initialized = True
    engine.device = "cuda"
    engine.processor = object()
    engine.vlm_model = fake_model  # type: ignore[assignment]
    engine.custom_inference = "unlimited_ocr"
    engine.model_config = EngineModelConfig(
        extra_config={
            "unlimited_ocr_base_size": 1024,
            "unlimited_ocr_image_size": 640,
            "unlimited_ocr_crop_mode": True,
            "unlimited_ocr_no_repeat_ngram_size": 35,
            "unlimited_ocr_ngram_window": 128,
        }
    )

    outputs = engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt="document parsing.",
                max_new_tokens=32768,
            )
        ]
    )

    assert outputs[0].text == "recognized **markdown**"
    assert fake_model.calls[0]["prompt"] == "<image>document parsing."
    assert fake_model.calls[0]["base_size"] == 1024
    assert fake_model.calls[0]["image_size"] == 640
    assert fake_model.calls[0]["crop_mode"] is True
    assert fake_model.calls[0]["eval_mode"] is True
    assert fake_model.calls[0]["max_length"] == 32768
    assert fake_model.calls[0]["no_repeat_ngram_size"] == 35
    assert fake_model.calls[0]["ngram_window"] == 128


def test_unlimited_ocr_legacy_model_calls_custom_infer() -> None:
    from docling.models.vlm_pipeline_models.hf_transformers_model import (
        HuggingFaceTransformersVlmModel,
    )

    fake_model = _FakeUnlimitedOcrModel()
    model = HuggingFaceTransformersVlmModel.__new__(HuggingFaceTransformersVlmModel)
    model.device = "cuda"
    model.processor = object()
    model.vlm_model = fake_model
    model.max_new_tokens = 32768
    model.temperature = 0.0
    model.vlm_options = vlm_model_specs.UNLIMITED_OCR_TRANSFORMERS
    model.custom_inference = "unlimited_ocr"

    predictions = list(
        model._process_images_with_unlimited_ocr(
            [Image.new("RGB", (8, 8), "white")],
            ["document parsing."],
        )
    )

    assert predictions[0].text == "recognized **markdown**"
    assert fake_model.calls[0]["prompt"] == "<image>document parsing."
    assert fake_model.calls[0]["max_length"] == 32768
