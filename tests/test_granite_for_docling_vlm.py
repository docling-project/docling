# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test Granite for Docling 500M VLM preset and legacy specs."""

import pytest
from pydantic import ValidationError

from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import DocLangOptions, VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
    VllmVlmEngineOptions,
)
from docling.datamodel.vlm_prompts import (
    DOCLANG_PAGE_PROMPT,
    DocLangTask,
    doclang_prompt,
)
from docling.models.inference_engines.vlm.base import VlmEngineType

REPO_ID = "ibm-granite/granite-for-docling-500m"
PRESET_ID = "granite_for_docling_500m"


def test_granite_for_docling_500m_preset_exists():
    """Verify the preset is registered with doclang output and its engines."""
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
    assert spec.trust_remote_code is False
    assert spec.max_new_tokens == 8192
    assert spec.stop_strings == ["</doclang>", "<|end_of_text|>"]
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is True
    assert spec.has_explicit_engine_export(VlmEngineType.TRANSFORMERS) is True
    assert spec.has_explicit_engine_export(VlmEngineType.VLLM) is True


def test_granite_for_docling_mlx_loads_the_same_checkpoint():
    """MLX needs the mlx-vlm release with the model, and no separate weights."""
    for preset_id, repo_id in (
        (PRESET_ID, REPO_ID),
        ("granite_for_docling_2b", "ibm-granite/granite-for-docling-2b"),
    ):
        spec = VlmConvertOptions.get_preset(preset_id).model_spec
        assert VlmEngineType.MLX in spec.supported_engines

        mlx_config = spec.get_engine_config(VlmEngineType.MLX)
        assert mlx_config.repo_id == repo_id
        assert mlx_config.min_engine_version == "0.8.1"

    mlx = VlmConvertOptions.from_preset(PRESET_ID, engine_options=MlxVlmEngineOptions())
    assert mlx.engine_options.engine_type == VlmEngineType.MLX


def test_granite_for_docling_500m_preset_engine_config():
    """Verify Transformers and vLLM overrides, plus API params."""
    spec = VlmConvertOptions.get_preset(PRESET_ID).model_spec

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == REPO_ID
    # GraniteDocling is native in Transformers from 5.23.0.
    assert tf_config.min_engine_version == "5.23.0"
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
    assert "include_stop_str_in_output" not in runtime
    vllm_runtime = spec.get_runtime_input_extra_config(VlmEngineType.VLLM)
    assert vllm_runtime["skip_special_tokens"] is False
    # The DocLang parser needs the closing </doclang> stop string.
    assert vllm_runtime["include_stop_str_in_output"] is True
    assert vllm_runtime["transformers_prompt_style"] == TransformersPromptStyle.CHAT

    api_params = spec.get_api_params(VlmEngineType.API_OPENAI)
    assert api_params["model"] == REPO_ID
    assert api_params["max_tokens"] == 8192
    assert api_params["skip_special_tokens"] is False
    assert api_params["include_stop_str_in_output"] is True


def test_granite_for_docling_500m_from_preset_engines():
    """Verify from_preset works for default, Transformers, and vLLM engines."""
    options = VlmConvertOptions.from_preset(PRESET_ID)
    assert options.model_spec.default_repo_id == REPO_ID
    assert options.model_spec.response_format == ResponseFormat.DOCLANG
    assert options.engine_options.engine_type == VlmEngineType.AUTO_INLINE

    transformers = VlmConvertOptions.from_preset(
        PRESET_ID,
        engine_options=TransformersVlmEngineOptions(),
    )
    assert transformers.engine_options.engine_type == VlmEngineType.TRANSFORMERS
    assert isinstance(transformers.engine_options, TransformersVlmEngineOptions)
    assert transformers.engine_options.trust_remote_code is False

    vllm = VlmConvertOptions.from_preset(
        PRESET_ID,
        engine_options=VllmVlmEngineOptions(),
    )
    assert vllm.engine_options.engine_type == VlmEngineType.VLLM
    assert isinstance(vllm.engine_options, VllmVlmEngineOptions)
    assert vllm.engine_options.trust_remote_code is False


def test_granite_for_docling_500m_legacy_specs():
    """Verify legacy InlineVlmOptions / ApiVlmOptions stay aligned with the preset."""
    transformers_spec = vlm_model_specs.GRANITE_FOR_DOCLING_500M_TRANSFORMERS
    assert transformers_spec.repo_id == REPO_ID
    assert transformers_spec.prompt == DOCLANG_PAGE_PROMPT
    assert transformers_spec.inference_framework == InferenceFramework.TRANSFORMERS
    assert transformers_spec.response_format == ResponseFormat.DOCLANG
    assert transformers_spec.trust_remote_code is False
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
    assert vllm_spec.extra_generation_config["include_stop_str_in_output"] is True
    assert "include_stop_str_in_output" not in (
        transformers_spec.extra_generation_config
    )

    mlx_spec = vlm_model_specs.GRANITE_FOR_DOCLING_500M_MLX
    assert mlx_spec.repo_id == REPO_ID
    assert mlx_spec.inference_framework == InferenceFramework.MLX
    assert mlx_spec.prompt == DOCLANG_PAGE_PROMPT
    assert mlx_spec.response_format == ResponseFormat.DOCLANG
    assert mlx_spec.max_new_tokens == 8192
    assert mlx_spec.stop_strings == ["</doclang>", "<|end_of_text|>"]

    vllm_api = vlm_model_specs.GRANITE_FOR_DOCLING_500M_VLLM_API
    assert vllm_api.params["model"] == REPO_ID
    assert vllm_api.params["max_tokens"] == 8192
    assert vllm_api.params["skip_special_tokens"] is False
    assert vllm_api.params["include_stop_str_in_output"] is True
    assert vllm_api.response_format == ResponseFormat.DOCLANG


def test_granite_for_docling_2b_preset():
    """The 2B preset shares the 500M setup, with its own repo and reply budget."""
    repo_id = "ibm-granite/granite-for-docling-2b"
    preset = VlmConvertOptions.get_preset("granite_for_docling_2b")
    assert preset.name == "Granite-for-Docling-2B"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == repo_id
    assert spec.prompt == DOCLANG_PAGE_PROMPT
    assert spec.response_format == ResponseFormat.DOCLANG
    assert spec.trust_remote_code is False
    assert spec.max_new_tokens == 16384
    assert spec.stop_strings == ["</doclang>", "<|end_of_text|>"]

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == repo_id
    assert tf_config.min_engine_version == "5.23.0"
    vllm_config = spec.get_engine_config(VlmEngineType.VLLM)
    assert vllm_config.repo_id == repo_id
    assert vllm_config.extra_config["max_model_len"] == 32768
    vllm_runtime = spec.get_runtime_input_extra_config(VlmEngineType.VLLM)
    assert vllm_runtime["include_stop_str_in_output"] is True

    for engine_type in (VlmEngineType.API, VlmEngineType.API_OPENAI):
        api_params = spec.get_api_params(engine_type)
        assert api_params["model"] == repo_id
        assert api_params["max_tokens"] == 16384
        assert api_params["include_stop_str_in_output"] is True

    # The 500M preset keeps its own repo and budget.
    spec_500m = VlmConvertOptions.get_preset(PRESET_ID).model_spec
    assert spec_500m.default_repo_id == REPO_ID
    assert spec_500m.max_new_tokens == 8192
    assert spec_500m.get_api_params(VlmEngineType.API)["model"] == REPO_ID


def test_granite_for_docling_2b_legacy_specs():
    """Legacy 2B specs follow the 2B preset and leave the 500M specs unchanged."""
    repo_id = "ibm-granite/granite-for-docling-2b"
    transformers_spec = vlm_model_specs.GRANITE_FOR_DOCLING_2B_TRANSFORMERS
    assert transformers_spec.repo_id == repo_id
    assert transformers_spec.max_new_tokens == 16384
    assert transformers_spec.trust_remote_code is False
    assert "include_stop_str_in_output" not in (
        transformers_spec.extra_generation_config
    )

    mlx_spec = vlm_model_specs.GRANITE_FOR_DOCLING_2B_MLX
    assert mlx_spec.repo_id == repo_id
    assert mlx_spec.inference_framework == InferenceFramework.MLX
    assert mlx_spec.max_new_tokens == 16384
    assert vlm_model_specs.GRANITE_FOR_DOCLING_500M_MLX.repo_id == REPO_ID
    assert vlm_model_specs.GRANITE_FOR_DOCLING_500M_MLX.max_new_tokens == 8192

    vllm_spec = vlm_model_specs.GRANITE_FOR_DOCLING_2B_VLLM
    assert vllm_spec.repo_id == repo_id
    assert vllm_spec.inference_framework == InferenceFramework.VLLM
    assert vllm_spec.extra_generation_config["include_stop_str_in_output"] is True

    vllm_api = vlm_model_specs.GRANITE_FOR_DOCLING_2B_VLLM_API
    assert vllm_api.params["model"] == repo_id
    assert vllm_api.params["max_tokens"] == 16384
    assert vllm_api.params["include_stop_str_in_output"] is True

    assert vlm_model_specs.GRANITE_FOR_DOCLING_500M_TRANSFORMERS.repo_id == REPO_ID
    assert vlm_model_specs.GRANITE_FOR_DOCLING_500M_VLLM_API.params["model"] == REPO_ID
    assert (
        vlm_model_specs.GRANITE_FOR_DOCLING_500M_VLLM_API.params["max_tokens"] == 8192
    )


def test_granite_docling_258m_default_unchanged():
    """The 258M DocTags preset remains the VLM convert default."""
    default = VlmConvertOptions.from_preset("granite_docling")
    assert default.model_spec.default_repo_id == "ibm-granite/granite-docling-258M"
    assert default.model_spec.response_format == ResponseFormat.DOCTAGS


def test_doclang_prompt_lists_the_requested_tasks():
    assert doclang_prompt() == "<doclang>"
    assert doclang_prompt([DocLangTask.DETAILED]) == "<doclang> [<detailed>]"
    # Order has no effect on the model, so it is kept; repeats are dropped.
    assert (
        doclang_prompt([DocLangTask.TABLE, DocLangTask.LAYOUT, DocLangTask.TABLE])
        == "<doclang> [<table>, <layout>]"
    )


@pytest.mark.parametrize("preset_id", [PRESET_ID, "granite_for_docling_2b"])
def test_doclang_tasks_select_the_page_prompt(preset_id):
    assert VlmConvertOptions.from_preset(preset_id).page_prompt() == "<doclang>"

    options = VlmConvertOptions.from_preset(
        preset_id,
        doclang=DocLangOptions(
            tasks=[DocLangTask.OCR, DocLangTask.LAYOUT, DocLangTask.OCR]
        ),
    )
    assert options.doclang.tasks == [DocLangTask.OCR, DocLangTask.LAYOUT]
    assert options.page_prompt() == "<doclang> [<ocr>, <layout>]"


def test_doclang_tasks_are_rejected_for_other_response_formats():
    with pytest.raises(ValidationError, match="doclang response format"):
        VlmConvertOptions.from_preset(
            "granite_docling", doclang=DocLangOptions(tasks=[DocLangTask.TABLE])
        )


def test_doclang_tasks_cannot_be_empty():
    with pytest.raises(ValidationError, match="at least one task"):
        DocLangOptions(tasks=[])
    with pytest.raises(ValidationError):
        DocLangOptions(tasks=["chart"])
