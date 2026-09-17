# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import AnyUrl, BaseModel, Field

from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmOptions,
)
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
)
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_API,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.exceptions import OperationNotAllowed
from docling.models.extraction.api_extraction_model import ApiExtractionVlmModel
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Invoice(BaseModel):
    invoice_date: str = Field(description="The date the invoice was issued")
    total: float = Field(description="The invoice total")


def test_api_options_dispatch_to_extraction_model() -> None:
    pipeline = ExtractionVlmPipeline(
        VlmExtractionPipelineOptions(
            vlm_options=GRANITE_VISION_4_1_API,
            enable_remote_services=True,
        )
    )
    assert isinstance(pipeline.vlm_model, ApiExtractionVlmModel)
    assert pipeline.vlm_model.params["model"] == ("ibm-granite/granite-vision-4.1-4b")
    assert pipeline.vlm_model.params["max_tokens"] == (
        GRANITE_VISION_4_1_API.model_spec.max_new_tokens
    )


def test_api_engine_requires_enable_remote_services() -> None:
    with pytest.raises(OperationNotAllowed):
        ExtractionVlmPipeline(
            VlmExtractionPipelineOptions(
                vlm_options=GRANITE_VISION_4_1_API,
                enable_remote_services=False,
            )
        )


def test_api_engine_uses_model_spec_defaults() -> None:
    options = ExtractionVlmOptions.from_preset(
        "nuextract_2b",
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=AnyUrl("https://example.test/v1/chat/completions"),
        ),
    )

    assert options.get_api_params() == {"model": "numind/NuExtract-2.0-2B"}


def test_unsupported_local_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not support the mlx VLM engine"):
        ExtractionVlmOptions.from_preset(
            "nuextract_2b", engine_options=MlxVlmEngineOptions()
        )


def _prompt_only_pipeline(style: ExtractionPromptStyle) -> ExtractionVlmPipeline:
    spec = {
        ExtractionPromptStyle.NUEXTRACT: NU_EXTRACT_2B_TRANSFORMERS,
        ExtractionPromptStyle.GRANITE_VISION: GRANITE_VISION_4_1_API,
    }[style]
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = cast(
        VlmExtractionPipelineOptions, SimpleNamespace(vlm_options=spec)
    )
    return pipeline


def test_granite_prompt_is_schema_plus_instruction() -> None:
    pipeline = _prompt_only_pipeline(ExtractionPromptStyle.GRANITE_VISION)
    prompt = pipeline._prepare_target(_Invoice).prompt

    assert "Extract structured data" in prompt
    assert "Return ONLY valid JSON" in prompt
    body_start = prompt.index("{")
    body = json.loads(prompt[body_start : prompt.rindex("}") + 1])
    assert set(body["properties"]) == {"invoice_date", "total"}
    assert body["properties"]["invoice_date"]["description"] == (
        "The date the invoice was issued"
    )


def test_nuextract_prompt_is_passthrough_instance() -> None:
    pipeline = _prompt_only_pipeline(ExtractionPromptStyle.NUEXTRACT)
    prompt = pipeline._prepare_target(
        '{"invoice_date": "string"}'
    ).chat_template_kwargs["template"]

    assert prompt == '{"invoice_date": "string"}'
    assert "Extract structured data" not in prompt


@pytest.mark.parametrize("serialized", [False, True])
def test_legacy_inline_options_are_adapted(serialized: bool) -> None:
    legacy = InlineVlmOptions(
        repo_id="numind/NuExtract-2.0-2B",
        prompt="",
        inference_framework=InferenceFramework.TRANSFORMERS,
        response_format=ResponseFormat.PLAINTEXT,
        quantized=True,
        torch_dtype="float16",
        trust_remote_code=True,
        use_kv_cache=False,
    )
    value: Any = legacy.model_dump(mode="json") if serialized else legacy

    with pytest.warns(DeprecationWarning):
        options = VlmExtractionPipelineOptions(vlm_options=value)

    engine = options.vlm_options.engine_options
    assert isinstance(engine, TransformersVlmEngineOptions)
    assert engine.quantized is True
    assert engine.torch_dtype == "float16"
    assert engine.trust_remote_code is True
    assert engine.use_kv_cache is False


@pytest.fixture
def extraction_http(monkeypatch):
    """Capture the shared transport's HTTP boundary without calling a service."""
    from unittest.mock import MagicMock

    from docling.utils import api_image_request

    session = MagicMock()
    response = session.__enter__.return_value.post.return_value
    response.ok = True
    response.text = json.dumps(
        {
            "id": "test",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
    )
    monkeypatch.setattr(api_image_request, "_make_retry_session", lambda: session)
    return session.__enter__.return_value.post, response


def _api_model(
    *,
    preparation="generic_chat",
    output_mode="prompt_only",
    params=None,
    engine_type=VlmEngineType.API,
    api_defaults=None,
):
    from docling.datamodel.extraction_options import (
        GRANITE_VISION_4_1_SPEC,
        NUEXTRACT_2B_SPEC,
    )
    from docling.datamodel.stage_model_specs import ApiModelConfig

    spec = (
        NUEXTRACT_2B_SPEC if preparation == "nuextract" else GRANITE_VISION_4_1_SPEC
    ).model_copy(
        update={
            "extra_chat_template_kwargs": {
                "enable_thinking": True,
                "nested": {"model": True},
            },
            "api_overrides": {engine_type: ApiModelConfig(params=api_defaults or {})},
        }
    )
    options = ExtractionVlmOptions(
        model_spec=spec,
        engine_options=ApiVlmEngineOptions(
            engine_type=engine_type,
            params=params or {},
            headers={"Authorization": "Bearer test"},
            timeout=13,
        ),
        output_mode=output_mode,
    )
    return ApiExtractionVlmModel(True, True, options)


@pytest.mark.parametrize("preparation", ["generic_chat", "nuextract"])
def test_ordered_api_payload_and_cached_call_isolation(extraction_http, preparation):
    from copy import deepcopy

    from PIL import Image

    from docling.datamodel.extraction import (
        ExtractionTarget,
        ExtractionTemplate,
        ImageContentItem,
        TextContentItem,
    )
    from docling.models.extraction.prompt_utils import prepare_target

    post, _ = extraction_http
    model = _api_model(
        preparation=preparation,
        params={
            "model": "override",
            "temperature": 0.2,
            "max_tokens": 99,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "nested": {"engine": True},
            },
        },
        api_defaults={
            "chat_template_kwargs": {"mode": "structured", "model_default": True}
        },
    )
    saved = deepcopy(model.params)
    with Image.new("RGB", (2, 3), "red") as image:
        contents = [
            TextContentItem(text="before"),
            ImageContentItem(image=image),
            TextContentItem(text="after"),
        ]
        for field in ("total", "date"):
            target = ExtractionTarget(
                template=ExtractionTemplate(
                    format="nuextract"
                    if preparation == "nuextract"
                    else "example_json",
                    value={field: "string"},
                ),
                instructions=f"Extract {field}",
            )
            prepared = prepare_target(target, model.model_spec)
            prediction = next(iter(model.process([contents], prepared)))
            payload = post.call_args.kwargs["json"]
            content = payload["messages"][0]["content"]
            assert [item["type"] for item in content[:3]] == [
                "text",
                "image_url",
                "text",
            ]
            assert content[0]["text"] == "before" and content[2]["text"] == "after"
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
            chat = payload["chat_template_kwargs"]
            assert chat["enable_thinking"] is False and chat["nested"] == {
                "engine": True
            }
            assert chat["mode"] == "structured" and chat["model_default"] is True
            if preparation == "nuextract":
                assert json.loads(chat["template"]) == {field: "string"}
                assert f"Extract {field}" in chat["instructions"] and len(content) == 3
            else:
                assert (
                    content[3]["text"] == prepared.prompt
                    and f"Extract {field}" in prepared.prompt
                )
            assert "response_format" not in payload
            assert (
                payload["model"] == "override"
                and payload["max_tokens"] == 99
                and payload["temperature"] == 0.2
            )
            assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer test"}
            assert post.call_args.kwargs["timeout"] == 13
            assert prediction.num_tokens == 7 and prediction.usage["total_tokens"] == 7
            assert prediction.stop_reason.value == "length"
            chat["nested"]["engine"] = "mutated"
    assert model.params == saved
    assert model.engine_options.params["chat_template_kwargs"]["nested"] == {
        "engine": True
    }
    assert "template" not in model.model_spec.extra_chat_template_kwargs


def test_dynamic_vllm_constraints_and_provider_rejection_without_fallback(
    extraction_http,
):
    from copy import deepcopy

    import requests

    from docling.datamodel.extraction import ExtractionTarget, TextContentItem
    from docling.models.extraction.prompt_utils import prepare_target

    post, response = extraction_http
    model = _api_model(output_mode="schema_constrained")
    schema = {
        "type": "object",
        "$defs": {
            "entry": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["paid", "due"]},
                    "total": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": ["status"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "entries": {"type": "array", "items": {"$ref": "#/$defs/entry"}}
        },
        "required": ["entries"],
    }
    original = deepcopy(schema)
    prepared = prepare_target(ExtractionTarget(output_schema=schema), model.model_spec)
    next(iter(model.process([[TextContentItem(text="invoice")]], prepared)))
    constraint = post.call_args.kwargs["json"]["response_format"]
    assert constraint["type"] == "json_schema"
    sent = constraint["json_schema"]["schema"]
    assert sent["properties"]["entries"]["items"] == schema["$defs"]["entry"]
    assert "$defs" not in sent and schema == original
    assert prepared.constraint_schema is None and prepared.validator.schema == schema
    response.ok = False
    response.status_code = 400
    response.text = "schema compilation rejected"
    response.raise_for_status.side_effect = requests.HTTPError(
        "400 schema compilation rejected"
    )
    other = prepare_target(
        ExtractionTarget(
            output_schema={"type": "object", "properties": {"date": {"type": "string"}}}
        ),
        model.model_spec,
    )
    with pytest.raises(RuntimeError, match="compilation rejected"):
        list(model.process([[TextContentItem(text="invoice")]], other))
    assert post.call_count == 2
    assert (
        post.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"]
        == other.target.output_schema
    )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("pattern", "[a-z]+"),
        ("minimum", 1),
        ("oneOf", [{"type": "string"}, {"type": "number"}]),
        ("allOf", [{"type": "string"}]),
        ("uniqueItems", True),
        ("format", "date"),
    ],
)
def test_constrained_subset_fails_before_http(extraction_http, keyword, value):
    from docling.datamodel.extraction import ExtractionTarget, TextContentItem
    from docling.models.extraction.prompt_utils import prepare_target

    post, _ = extraction_http
    model = _api_model(output_mode="schema_constrained")
    prepared = prepare_target(
        ExtractionTarget(
            output_schema={"type": "object", "properties": {"value": {keyword: value}}}
        ),
        model.model_spec,
    )
    with pytest.raises(ValueError, match=f"#/properties/value/{keyword}"):
        list(model.process([[TextContentItem(text="text")]], prepared))
    post.assert_not_called()


def test_constrained_requires_schema_and_prompt_only_is_explicit(extraction_http):
    from docling.datamodel.extraction import (
        ExtractionTarget,
        ExtractionTemplate,
        TextContentItem,
    )
    from docling.models.extraction.prompt_utils import (
        prepare_legacy_target,
        prepare_target,
    )

    post, _ = extraction_http
    model = _api_model(output_mode="schema_constrained")
    targets = [
        prepare_legacy_target("{}", model.model_spec),
        prepare_target(
            ExtractionTarget(
                template=ExtractionTemplate(format="example_json", value={})
            ),
            model.model_spec,
        ),
    ]
    for target in targets:
        with pytest.raises(ValueError, match="requires an output schema"):
            list(model.process([[TextContentItem(text="text")]], target))
    post.assert_not_called()
    model = _api_model()
    schema = {"type": "object", "properties": {"x": {"type": "number", "minimum": 1}}}
    target = prepare_target(ExtractionTarget(output_schema=schema), model.model_spec)
    next(iter(model.process([[TextContentItem(text="text")]], target)))
    assert "response_format" not in post.call_args.kwargs["json"]
    assert target.validator.schema == schema


@pytest.mark.parametrize(
    "engine_type",
    [
        VlmEngineType.TRANSFORMERS,
        VlmEngineType.API_OLLAMA,
        VlmEngineType.API_LMSTUDIO,
        VlmEngineType.API_OPENAI,
    ],
)
def test_unverified_constrained_engines_rejected(engine_type):
    engine = (
        TransformersVlmEngineOptions()
        if engine_type == VlmEngineType.TRANSFORMERS
        else ApiVlmEngineOptions(engine_type=engine_type)
    )
    with pytest.raises(ValueError, match="explicitly configured vLLM"):
        ExtractionVlmOptions.from_preset(
            "nuextract_2b", engine_options=engine, output_mode="schema_constrained"
        )


@pytest.mark.parametrize(
    "owner", ["model_chat", "api_defaults", "api_chat_defaults", "engine", "engine_top"]
)
@pytest.mark.parametrize(
    "key",
    [
        "messages",
        "template",
        "instructions",
        "response_format",
        "structured_outputs",
        "guided_json",
    ],
)
def test_static_request_field_collisions_rejected(extraction_http, owner, key):
    from docling.datamodel.extraction import TextContentItem
    from docling.models.extraction.prompt_utils import prepare_legacy_target

    post, _ = extraction_http
    model = _api_model(
        params={"chat_template_kwargs": {key: "static"}}
        if owner == "engine"
        else {key: "static"}
        if owner == "engine_top"
        else {},
        api_defaults={key: "static"}
        if owner == "api_defaults"
        else {"chat_template_kwargs": {key: "static"}}
        if owner == "api_chat_defaults"
        else {},
    )
    if owner == "model_chat":
        model.model_spec.extra_chat_template_kwargs[key] = "static"
    with pytest.raises(ValueError, match="request-owned"):
        target = prepare_legacy_target("{}", model.model_spec)
        list(model.process([[TextContentItem(text="text")]], target))
    post.assert_not_called()


def test_api_image_wrapper_uses_final_prompt_once(extraction_http):
    import numpy as np

    post, _ = extraction_http
    model = _api_model()
    predictions = list(
        model.process_images(
            [np.zeros((2, 2, 3), dtype=np.uint8)] * 2, ["final one", "final two"]
        )
    )
    assert len(predictions) == 2
    assert [
        call.kwargs["json"]["messages"][0]["content"][1]["text"]
        for call in post.call_args_list
    ] == ["final one", "final two"]
    with pytest.raises(ValueError, match="must match"):
        list(model.process_images([np.zeros((2, 2))], ["one", "two"]))


@pytest.mark.parametrize(
    "engine_type",
    [VlmEngineType.API_OLLAMA, VlmEngineType.API_LMSTUDIO, VlmEngineType.API_OPENAI],
)
def test_named_api_prompt_only_omits_vllm_options(extraction_http, engine_type):
    from docling.datamodel.extraction import TextContentItem
    from docling.models.extraction.prompt_utils import prepare_legacy_target

    post, _ = extraction_http
    model = _api_model(engine_type=engine_type)
    model.model_spec.extra_chat_template_kwargs.clear()
    next(
        iter(
            model.process(
                [[TextContentItem(text="text")]],
                prepare_legacy_target("{}", model.model_spec),
            )
        )
    )
    assert "chat_template_kwargs" not in post.call_args.kwargs["json"]
    assert "response_format" not in post.call_args.kwargs["json"]
    model.model_spec.extra_chat_template_kwargs["enable_thinking"] = False
    with pytest.raises(ValueError, match="explicitly configured vLLM"):
        list(
            model.process(
                [[TextContentItem(text="text")]],
                prepare_legacy_target("{}", model.model_spec),
            )
        )
    assert post.call_count == 1


@pytest.mark.parametrize(
    "text,reason", [("", "stop"), ("", "content_filter"), ("{}", "content_filter")]
)
def test_extraction_retains_empty_and_filtered_remote_behavior(
    extraction_http, text, reason
):
    from docling.datamodel.extraction import TextContentItem
    from docling.models.extraction.prompt_utils import prepare_legacy_target

    post, response = extraction_http
    payload = json.loads(response.text)
    payload["choices"][0]["message"]["content"] = text
    payload["choices"][0]["finish_reason"] = reason
    response.text = json.dumps(payload)
    model = _api_model()
    predictions = model.process(
        [[TextContentItem(text="text")]], prepare_legacy_target("{}", model.model_spec)
    )
    if not text:
        with pytest.raises(RuntimeError, match="returned no content"):
            list(predictions)
    else:
        assert next(iter(predictions)).stop_reason.value == "content_filter"
    assert post.call_count == 1
