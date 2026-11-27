# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test the Granite Docling v2 VLM presets."""

import pytest
from PIL import Image
from pydantic import ValidationError

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ApiImageRequestResult, VlmStopReason
from docling.datamodel.pipeline_options import DocLangOptions, VlmConvertOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
    VllmVlmEngineOptions,
)
from docling.datamodel.vlm_prompts import (
    DOCLANG_PAGE_PROMPT,
    DocLangTask,
    doclang_prompt,
)
from docling.models.inference_engines.vlm.api_openai_compatible_engine import (
    ApiVlmEngine,
)
from docling.models.inference_engines.vlm.base import VlmEngineInput, VlmEngineType
from docling.models.inference_engines.vlm.factory import create_vlm_engine

REPO_ID = "ibm-granite/granite-docling-v2-500m"
PRESET_ID = "granite_docling_v2_500m"


def test_granite_docling_v2_500m_preset_exists():
    """Verify the preset is registered with doclang output and its engines."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert PRESET_ID in preset_ids
    assert "granite_docling" in preset_ids

    preset = VlmConvertOptions.get_preset(PRESET_ID)
    assert preset.preset_id == PRESET_ID
    assert preset.name == "Granite-Docling-v2-500M"
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


def test_granite_docling_v2_mlx_loads_the_same_checkpoint():
    """MLX needs the mlx-vlm release with the model, and no separate weights."""
    for preset_id, repo_id in (
        (PRESET_ID, REPO_ID),
        ("granite_docling_v2_2b", "ibm-granite/granite-docling-v2-2b"),
    ):
        spec = VlmConvertOptions.get_preset(preset_id).model_spec
        assert VlmEngineType.MLX in spec.supported_engines

        mlx_config = spec.get_engine_config(VlmEngineType.MLX)
        assert mlx_config.repo_id == repo_id
        assert mlx_config.min_engine_version == "0.8.1"

    mlx = VlmConvertOptions.from_preset(PRESET_ID, engine_options=MlxVlmEngineOptions())
    assert mlx.engine_options.engine_type == VlmEngineType.MLX


def test_granite_docling_v2_500m_preset_engine_config():
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

    # The API flavour is a vLLM server; OpenAI does not serve the model.
    assert VlmEngineType.API in spec.supported_engines
    assert VlmEngineType.API_OPENAI not in spec.supported_engines
    api_config = spec.get_api_config(VlmEngineType.API)
    assert api_config.params == {"model": REPO_ID}
    assert (api_config.timeout, api_config.concurrency) == (90.0, 4)
    # The decoding settings go with each API request, so user params keep them.
    api_runtime = spec.get_runtime_input_extra_config(VlmEngineType.API)
    assert api_runtime["api_params"] == {
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
    }


@pytest.mark.parametrize("preset_id", [PRESET_ID, "granite_docling_v2_2b"])
@pytest.mark.parametrize("user_params", [{}, {"model": "served-name"}])
def test_granite_docling_v2_api_request_keeps_the_closing_tag(
    monkeypatch: pytest.MonkeyPatch, preset_id: str, user_params: dict
) -> None:
    """User params (e.g. a served model name) must not drop the stop-string settings."""
    captured: dict = {}

    def _fake_api_image_request(**kwargs):
        captured.update(kwargs)
        return ApiImageRequestResult("ok", 1, VlmStopReason.END_OF_SEQUENCE)

    monkeypatch.setattr(
        "docling.models.inference_engines.vlm.api_openai_compatible_engine.api_image_request",
        _fake_api_image_request,
    )
    spec = VlmConvertOptions.get_preset(preset_id).model_spec
    options = ApiVlmEngineOptions(
        engine_type=VlmEngineType.API,
        url="http://localhost:8000/v1/chat/completions",
        params=user_params,
    )
    engine = create_vlm_engine(
        options=options,
        model_spec=spec,
        enable_remote_services=True,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
    )
    engine.predict_batch(
        [
            VlmEngineInput(
                image=Image.new("RGB", (8, 8), "white"),
                prompt=spec.prompt,
                stop_strings=list(spec.stop_strings),
                extra_generation_config=spec.get_runtime_input_extra_config(
                    VlmEngineType.API
                ),
                max_new_tokens=spec.max_new_tokens,
            )
        ]
    )
    assert captured["include_stop_str_in_output"] is True
    assert captured["skip_special_tokens"] is False
    assert captured["model"] == user_params.get("model", spec.default_repo_id)
    assert captured["max_tokens"] == spec.max_new_tokens
    assert captured["timeout"] == (90.0 if preset_id == PRESET_ID else 180.0)
    assert isinstance(engine, ApiVlmEngine)
    assert engine.concurrency == 4


@pytest.mark.parametrize("preset_id", [PRESET_ID, "granite_docling_v2_2b"])
def test_granite_docling_v2_api_user_timeout_and_concurrency_win(
    preset_id: str,
) -> None:
    """Values the user sets on the options override the preset's API values."""
    spec = VlmConvertOptions.get_preset(preset_id).model_spec
    engine = create_vlm_engine(
        options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url="http://localhost:8000/v1/chat/completions",
            timeout=30.0,
            concurrency=16,
        ),
        model_spec=spec,
        enable_remote_services=True,
        artifacts_path=None,
        accelerator_options=AcceleratorOptions(),
    )
    assert isinstance(engine, ApiVlmEngine)
    assert (engine.timeout, engine.concurrency) == (30.0, 16)


def test_granite_docling_v2_500m_from_preset_engines():
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


def test_granite_docling_v2_2b_preset():
    """The 2B preset shares the 500M setup, with its own repo and reply budget."""
    repo_id = "ibm-granite/granite-docling-v2-2b"
    preset = VlmConvertOptions.get_preset("granite_docling_v2_2b")
    assert preset.name == "Granite-Docling-v2-2B"
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

    assert VlmEngineType.API_OPENAI not in spec.supported_engines
    api_config = spec.get_api_config(VlmEngineType.API)
    assert api_config.params == {"model": repo_id}
    assert (api_config.timeout, api_config.concurrency) == (180.0, 4)
    api_runtime = spec.get_runtime_input_extra_config(VlmEngineType.API)
    assert api_runtime["api_params"]["include_stop_str_in_output"] is True

    # The 500M preset keeps its own repo and budget.
    spec_500m = VlmConvertOptions.get_preset(PRESET_ID).model_spec
    assert spec_500m.default_repo_id == REPO_ID
    assert spec_500m.max_new_tokens == 8192
    assert spec_500m.get_api_params(VlmEngineType.API)["model"] == REPO_ID


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


@pytest.mark.parametrize("preset_id", [PRESET_ID, "granite_docling_v2_2b"])
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
