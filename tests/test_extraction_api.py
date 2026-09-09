# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import AnyUrl, BaseModel, Field

from docling.datamodel.extraction_options import (
    ExtractionPromptStyle,
    ExtractionVlmOptions,
)
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    MlxVlmEngineOptions,
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
    prompt = pipeline._build_prompt(_Invoice)

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
    prompt = pipeline._build_prompt('{"invoice_date": "string"}')

    assert prompt == '{"invoice_date": "string"}'
    assert "Extract structured data" not in prompt
