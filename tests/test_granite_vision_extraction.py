# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for extraction model prompt style dispatch."""

from unittest.mock import patch

from docling.datamodel.extraction_options import ExtractionPromptStyle
from docling.datamodel.pipeline_options import VlmExtractionPipelineOptions
from docling.datamodel.vlm_model_specs import (
    GRANITE_VISION_4_1_TRANSFORMERS,
    NU_EXTRACT_2B_TRANSFORMERS,
)
from docling.models.extraction.prompt_utils import _build_extraction_prompt


def test_granite_vision_spec_has_correct_repo_id() -> None:
    """Verify the Granite Vision 4.1 spec points to the correct model."""
    assert (
        GRANITE_VISION_4_1_TRANSFORMERS.repo_id == "ibm-granite/granite-vision-4.1-4b"
    )
    assert GRANITE_VISION_4_1_TRANSFORMERS.trust_remote_code is True


def test_default_spec_uses_nuextract_style() -> None:
    """The default preset carries NUEXTRACT style."""
    options = VlmExtractionPipelineOptions()
    assert (
        options.vlm_options.extraction_prompt_style == ExtractionPromptStyle.NUEXTRACT
    )


def test_granite_vision_preset_carries_varex_style() -> None:
    """The Granite preset welds VAREX style to the model — no separate setting."""
    options = VlmExtractionPipelineOptions(vlm_options=GRANITE_VISION_4_1_TRANSFORMERS)
    assert (
        options.vlm_options.extraction_prompt_style
        == ExtractionPromptStyle.GRANITE_VISION
    )
    assert options.vlm_options.repo_id == "ibm-granite/granite-vision-4.1-4b"


@patch(
    "docling.pipeline.extraction_vlm_pipeline.TransformersExtractionModel",
)
def test_pipeline_hands_style_carrying_spec_to_model(mock_model_cls: object) -> None:
    """The pipeline hands the model the spec, which carries the style itself."""
    from docling.pipeline.extraction_vlm_pipeline import ExtractionVlmPipeline

    options = VlmExtractionPipelineOptions(vlm_options=GRANITE_VISION_4_1_TRANSFORMERS)
    _ = ExtractionVlmPipeline(pipeline_options=options)
    mock_model_cls.assert_called_once()  # type: ignore[union-attr]
    call_kwargs = mock_model_cls.call_args[1]  # type: ignore[union-attr]
    assert "prompt_style" not in call_kwargs
    assert (
        call_kwargs["vlm_options"].extraction_prompt_style
        == ExtractionPromptStyle.GRANITE_VISION
    )


def test_build_extraction_prompt() -> None:
    """Verify the extraction prompt is formatted correctly."""
    template = '{"name": "string", "age": "integer"}'
    prompt = _build_extraction_prompt(template)

    assert template in prompt
    assert "Extract structured data" in prompt
    assert "Return ONLY valid JSON" in prompt
    assert "Return null for fields" in prompt
