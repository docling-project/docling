# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test Granite for Docling 500M VLM preset and legacy specs."""

from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    TransformersVlmEngineOptions,
    VllmVlmEngineOptions,
)
from docling.datamodel.vlm_prompts import DOCLANG_PAGE_PROMPT
from docling.models.inference_engines.vlm.base import VlmEngineType

REPO_ID = "docling-project/granite-for-docling-500m"
PRESET_ID = "granite_for_docling_500m"


def test_granite_for_docling_500m_preset_exists():
    """Verify the preset is registered with doclang output and no MLX export."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert PRESET_ID in preset_ids
    assert "granite_docling" in preset_ids

    preset = VlmConvertOptions.get_preset(PRESET_ID)
    assert preset.preset_id == PRESET_ID
    assert preset.name == "Granite-for-Docling-500M"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == REPO_ID
    assert spec.prompt == DOCLANG_PAGE_PROMPT
    assert spec.response_format == ResponseFormat.DOCLANG
    assert spec.trust_remote_code is True
    assert spec.max_new_tokens == 8192
    assert spec.stop_strings == ["</doclang>", "<|end_of_text|>"]
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is False
    assert spec.has_explicit_engine_export(VlmEngineType.TRANSFORMERS) is True
    assert spec.has_explicit_engine_export(VlmEngineType.VLLM) is True


def test_granite_for_docling_500m_preset_engine_config():
    """Verify Transformers and vLLM overrides, plus API params."""
    spec = VlmConvertOptions.get_preset(PRESET_ID).model_spec

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == REPO_ID
    assert tf_config.torch_dtype == "bfloat16"
    assert tf_config.extra_config["torch_dtype"] == "bfloat16"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert (
        tf_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )

    vllm_config = spec.get_engine_config(VlmEngineType.VLLM)
    assert vllm_config.repo_id == REPO_ID
    assert vllm_config.extra_config["dtype"] == "bfloat16"
    assert vllm_config.extra_config["max_model_len"] == 32768
    assert (
        vllm_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )

    runtime = spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS)
    assert runtime["skip_special_tokens"] is False
    vllm_runtime = spec.get_runtime_input_extra_config(VlmEngineType.VLLM)
    assert vllm_runtime["skip_special_tokens"] is False
    assert vllm_runtime["transformers_prompt_style"] == TransformersPromptStyle.CHAT

    api_params = spec.get_api_params(VlmEngineType.API_OPENAI)
    assert api_params["model"] == REPO_ID
    assert api_params["max_tokens"] == 8192
    assert api_params["skip_special_tokens"] is False


def test_granite_for_docling_500m_from_preset_engines():
    """Verify from_preset works for default, Transformers, and vLLM engines."""
    options = VlmConvertOptions.from_preset(PRESET_ID)
    assert options.model_spec.default_repo_id == REPO_ID
    assert options.model_spec.response_format == ResponseFormat.DOCLANG
    assert options.engine_options.engine_type == VlmEngineType.AUTO_INLINE

    transformers = VlmConvertOptions.from_preset(
        PRESET_ID,
        engine_options=TransformersVlmEngineOptions(trust_remote_code=True),
    )
    assert transformers.engine_options.engine_type == VlmEngineType.TRANSFORMERS
    assert isinstance(transformers.engine_options, TransformersVlmEngineOptions)
    assert transformers.engine_options.trust_remote_code is True

    vllm = VlmConvertOptions.from_preset(
        PRESET_ID,
        engine_options=VllmVlmEngineOptions(trust_remote_code=True),
    )
    assert vllm.engine_options.engine_type == VlmEngineType.VLLM
    assert isinstance(vllm.engine_options, VllmVlmEngineOptions)
    assert vllm.engine_options.trust_remote_code is True


def test_granite_for_docling_500m_legacy_specs():
    """Verify legacy InlineVlmOptions / ApiVlmOptions stay aligned with the preset."""
    transformers_spec = vlm_model_specs.GRANITE_FOR_DOCLING_500M_TRANSFORMERS
    assert transformers_spec.repo_id == REPO_ID
    assert transformers_spec.prompt == DOCLANG_PAGE_PROMPT
    assert transformers_spec.inference_framework == InferenceFramework.TRANSFORMERS
    assert transformers_spec.response_format == ResponseFormat.DOCLANG
    assert transformers_spec.trust_remote_code is True
    assert transformers_spec.torch_dtype == "bfloat16"
    assert (
        transformers_spec.transformers_model_type
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert transformers_spec.transformers_prompt_style == TransformersPromptStyle.CHAT
    assert transformers_spec.max_new_tokens == 8192
    assert transformers_spec.stop_strings == ["</doclang>", "<|end_of_text|>"]
    assert transformers_spec.extra_generation_config["skip_special_tokens"] is False

    vllm_spec = vlm_model_specs.GRANITE_FOR_DOCLING_500M_VLLM
    assert vllm_spec.repo_id == REPO_ID
    assert vllm_spec.inference_framework == InferenceFramework.VLLM
    assert vllm_spec.response_format == ResponseFormat.DOCLANG

    vllm_api = vlm_model_specs.GRANITE_FOR_DOCLING_500M_VLLM_API
    assert vllm_api.params["model"] == REPO_ID
    assert vllm_api.params["max_tokens"] == 8192
    assert vllm_api.params["skip_special_tokens"] is False
    assert vllm_api.response_format == ResponseFormat.DOCLANG


def test_granite_docling_258m_default_unchanged():
    """The 258M DocTags preset remains the VLM convert default."""
    default = VlmConvertOptions.from_preset("granite_docling")
    assert default.model_spec.default_repo_id == "ibm-granite/granite-docling-258M"
    assert default.model_spec.response_format == ResponseFormat.DOCTAGS
