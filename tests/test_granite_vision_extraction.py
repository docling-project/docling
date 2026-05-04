"""Unit tests for Granite Vision extraction model integration."""

from unittest.mock import patch

import pytest

from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.pipeline.extraction_vlm_pipeline import (
    _GRANITE_VISION_REPO_PREFIX,
    ExtractionVlmPipeline,
)


def test_granite_vision_spec_has_correct_repo_id() -> None:
    """Verify the Granite Vision 4.1 spec points to the correct model."""
    assert (
        GRANITE_VISION_4_1_TRANSFORMERS.repo_id == "ibm-granite/granite-vision-4.1-4b"
    )
    assert GRANITE_VISION_4_1_TRANSFORMERS.trust_remote_code is True


def test_granite_vision_spec_repo_matches_prefix() -> None:
    """Verify the spec repo_id matches the prefix used for model selection."""
    assert GRANITE_VISION_4_1_TRANSFORMERS.repo_id.startswith(
        _GRANITE_VISION_REPO_PREFIX
    )


def test_nuextract_spec_does_not_match_granite_prefix() -> None:
    """Verify NuExtract spec does not trigger Granite Vision model selection."""
    assert not NU_EXTRACT_2B_TRANSFORMERS.repo_id.startswith(
        _GRANITE_VISION_REPO_PREFIX
    )


@patch(
    "docling.pipeline.extraction_vlm_pipeline.ExtractionVlmPipeline._create_vlm_model"
)
def test_pipeline_selects_granite_vision_model(mock_create: object) -> None:
    """Verify pipeline dispatches to GraniteVisionExtractionModel for granite repos."""
    options = VlmExtractionPipelineOptions(
        vlm_options=GRANITE_VISION_4_1_TRANSFORMERS,
    )
    _ = ExtractionVlmPipeline(pipeline_options=options)
    mock_create.assert_called_once_with(options)  # type: ignore[union-attr]


@patch(
    "docling.pipeline.extraction_vlm_pipeline.ExtractionVlmPipeline._create_vlm_model"
)
def test_pipeline_selects_nuextract_model_by_default(mock_create: object) -> None:
    """Verify pipeline dispatches to NuExtract for the default spec."""
    options = VlmExtractionPipelineOptions()
    _ = ExtractionVlmPipeline(pipeline_options=options)
    mock_create.assert_called_once_with(options)  # type: ignore[union-attr]


def test_build_extraction_prompt() -> None:
    """Verify the extraction prompt is formatted correctly."""
    from docling.models.extraction.granite_vision_extraction_model import (
        GraniteVisionExtractionModel,
    )

    template = '{"name": "string", "age": "integer"}'
    prompt = GraniteVisionExtractionModel._build_extraction_prompt(template)

    assert template in prompt
    assert "Extract structured data" in prompt
    assert "Return ONLY valid JSON" in prompt
    assert "Return null for fields" in prompt
