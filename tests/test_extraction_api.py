# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Extraction pipeline with an OpenAI-conformant API engine."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from docling.datamodel.extraction_options import ExtractionPromptStyle
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_model_specs import GRANITE_VISION_4_1_API
from docling.exceptions import OperationNotAllowed
from docling.models.vlm_pipeline_models.api_vlm_model import ApiVlmModel
from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline


class _Invoice(BaseModel):
    invoice_date: str = Field(description="The date the invoice was issued")
    total: float = Field(description="The invoice total")


def test_api_options_dispatch_to_api_model() -> None:
    """ApiVlmOptions selects the remote engine, no local model is built."""
    pipeline = ExtractionVlmPipeline(
        VlmExtractionPipelineOptions(
            vlm_options=GRANITE_VISION_4_1_API,
            extraction_prompt_style=ExtractionPromptStyle.GRANITE_VISION,
            enable_remote_services=True,
        )
    )
    assert isinstance(pipeline.vlm_model, ApiVlmModel)


def test_api_engine_requires_enable_remote_services() -> None:
    """The remote engine must be opted into explicitly."""
    with pytest.raises(OperationNotAllowed):
        ExtractionVlmPipeline(
            VlmExtractionPipelineOptions(
                vlm_options=GRANITE_VISION_4_1_API,
                extraction_prompt_style=ExtractionPromptStyle.GRANITE_VISION,
                enable_remote_services=False,
            )
        )


def _prompt_only_pipeline(style: ExtractionPromptStyle) -> ExtractionVlmPipeline:
    """Pipeline whose prompt builder can be exercised without loading a model."""
    pipeline = ExtractionVlmPipeline.__new__(ExtractionVlmPipeline)
    pipeline.pipeline_options = SimpleNamespace(extraction_prompt_style=style)  # type: ignore[assignment]
    return pipeline


def test_varex_prompt_is_template_plus_instruction() -> None:
    """Granite/VAREX style wraps the serialized template in the instruction."""
    pipeline = _prompt_only_pipeline(ExtractionPromptStyle.GRANITE_VISION)
    prompt = pipeline._build_prompt(_Invoice)

    assert "Extract structured data" in prompt
    assert "Return ONLY valid JSON" in prompt
    # The wrapped body is the shared serialization: a sample instance keyed by
    # field name (what NuExtract also gets), not a JSON Schema — a schema would
    # nest the fields under a top-level "properties" key instead.
    body_start = prompt.index("{")
    body = json.loads(prompt[body_start : prompt.rindex("}") + 1])
    assert set(body) == {"invoice_date", "total"}


def test_nuextract_prompt_is_passthrough_instance() -> None:
    """NuExtract style passes a sample instance through, no VAREX wrapper."""
    pipeline = _prompt_only_pipeline(ExtractionPromptStyle.NUEXTRACT)
    prompt = pipeline._build_prompt('{"invoice_date": "string"}')

    assert prompt == '{"invoice_date": "string"}'
    assert "Extract structured data" not in prompt
