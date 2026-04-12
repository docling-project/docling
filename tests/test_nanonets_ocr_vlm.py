"""Test Nanonets-OCR2-3B VLM integration."""

from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.inference_engines.vlm.base import VlmEngineType


def test_nanonets_ocr2_preset_exists():
    """Verify preset is registered with correct metadata and model spec."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "nanonets_ocr2" in preset_ids

    preset = VlmConvertOptions.get_preset("nanonets_ocr2")
    assert preset.preset_id == "nanonets_ocr2"
    assert preset.name == "Nanonets-OCR2-3B"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == "nanonets/Nanonets-OCR2-3B"
    assert spec.response_format == ResponseFormat.MARKDOWN
    assert spec.trust_remote_code is False
    assert spec.max_new_tokens == 15000
    assert spec.supported_engines == {
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.MLX,
    }


def test_nanonets_ocr2_preset_engine_config():
    """Verify engine overrides propagate correctly through get_engine_config."""
    preset = VlmConvertOptions.get_preset("nanonets_ocr2")
    spec = preset.model_spec

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == "nanonets/Nanonets-OCR2-3B"
    assert tf_config.extra_config["torch_dtype"] == "bfloat16"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert (
        tf_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )

    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "mlx-community/Nanonets-OCR2-3B-bf16"
    assert mlx_config.extra_config == {}
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is True


def test_nanonets_ocr2_preset_instantiation():
    """Verify from_preset produces a usable VlmConvertOptions with engine options."""
    options = VlmConvertOptions.from_preset("nanonets_ocr2")
    assert options.model_spec.default_repo_id == "nanonets/Nanonets-OCR2-3B"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN
    assert options.engine_options is not None


def test_nanonets_ocr2_legacy_specs():
    """Verify legacy InlineVlmOptions specs are consistent."""
    transformers_spec = vlm_model_specs.NANONETS_OCR2_TRANSFORMERS
    assert transformers_spec.repo_id == "nanonets/Nanonets-OCR2-3B"
    assert transformers_spec.inference_framework == InferenceFramework.TRANSFORMERS
    assert transformers_spec.response_format == ResponseFormat.MARKDOWN
    assert transformers_spec.torch_dtype == "bfloat16"
    assert (
        transformers_spec.transformers_model_type
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert transformers_spec.transformers_prompt_style == TransformersPromptStyle.CHAT
    assert transformers_spec.scale == 2.0
    assert transformers_spec.temperature == 0.0
    assert transformers_spec.max_new_tokens == 15000

    mlx_spec = vlm_model_specs.NANONETS_OCR2_MLX
    assert mlx_spec.repo_id == "mlx-community/Nanonets-OCR2-3B-bf16"
    assert mlx_spec.inference_framework == InferenceFramework.MLX
    assert mlx_spec.response_format == transformers_spec.response_format
    assert mlx_spec.max_new_tokens == 15000
