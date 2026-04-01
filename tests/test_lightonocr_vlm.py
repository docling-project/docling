"""Test LightOnOCR-2-1B VLM integration."""

import os
from pathlib import Path

import pytest

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline


def test_lightonocr_preset_exists():
    """Verify preset is registered with correct metadata and model spec."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "lightonocr" in preset_ids

    preset = VlmConvertOptions.get_preset("lightonocr")
    assert preset.preset_id == "lightonocr"
    assert preset.name == "LightOnOCR-2-1B"
    assert preset.scale == 2.0
    assert preset.max_size == 1540
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == "lightonai/LightOnOCR-2-1B"
    assert spec.response_format == ResponseFormat.MARKDOWN
    assert spec.prompt == ""
    assert spec.trust_remote_code is False
    assert spec.max_new_tokens == 4096


def test_lightonocr_preset_engine_config():
    """Verify engine overrides propagate correctly through get_engine_config."""
    preset = VlmConvertOptions.get_preset("lightonocr")
    spec = preset.model_spec

    # Transformers engine config should carry torch_dtype and model type
    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == "lightonai/LightOnOCR-2-1B"
    assert tf_config.extra_config["torch_dtype"] == "bfloat16"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert (
        tf_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )

    # API overrides should have correct model params
    api_overrides = spec.api_overrides
    assert VlmEngineType.API in api_overrides
    assert (
        api_overrides[VlmEngineType.API].params["model"] == "lightonai/LightOnOCR-2-1B"
    )
    assert api_overrides[VlmEngineType.API].params["max_tokens"] == 4096
    assert VlmEngineType.API_OPENAI in api_overrides
    assert api_overrides[VlmEngineType.API_OPENAI].params["model"] == "lightonocr-2-1b"

    # No MLX override -- engine config should fall back to default repo_id
    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "lightonai/LightOnOCR-2-1B"
    assert mlx_config.extra_config == {}


def test_lightonocr_preset_instantiation():
    """Verify from_preset produces a usable VlmConvertOptions with engine options."""
    options = VlmConvertOptions.from_preset("lightonocr")
    assert options.model_spec.default_repo_id == "lightonai/LightOnOCR-2-1B"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN
    assert options.engine_options is not None
    assert options.max_size == 1540


def test_lightonocr_legacy_specs():
    """Verify legacy InlineVlmOptions/ApiVlmOptions specs are consistent."""
    # Transformers spec
    t = vlm_model_specs.LIGHTONOCR_TRANSFORMERS
    assert t.repo_id == "lightonai/LightOnOCR-2-1B"
    assert t.inference_framework == InferenceFramework.TRANSFORMERS
    assert t.response_format == ResponseFormat.MARKDOWN
    assert t.torch_dtype == "bfloat16"
    assert t.transformers_prompt_style == TransformersPromptStyle.CHAT
    assert t.transformers_model_type == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    assert t.scale == 2.0
    assert t.temperature == 0.0
    assert t.prompt == ""
    assert t.max_new_tokens == 4096

    # VLLM spec should share repo_id but differ in framework
    v = vlm_model_specs.LIGHTONOCR_VLLM
    assert v.repo_id == t.repo_id
    assert v.inference_framework == InferenceFramework.VLLM
    assert v.response_format == t.response_format

    # API spec
    a = vlm_model_specs.LIGHTONOCR_VLLM_API
    assert a.params["model"] == "lightonai/LightOnOCR-2-1B"
    assert a.params["max_tokens"] == 4096
    assert a.response_format == ResponseFormat.MARKDOWN
    assert a.concurrency == 4
    assert a.timeout == 90


def test_e2e_lightonocr_conversion():
    """E2E test with vLLM server (skipped in CI and when server is unavailable)."""
    if os.getenv("CI"):
        pytest.skip("Skipping in CI environment")

    try:
        import requests

        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        if response.status_code != 200:
            pytest.skip("vLLM server is not available")
    except Exception:
        pytest.skip("vLLM server is not available")

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.LIGHTONOCR_VLLM_API,
        enable_remote_services=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    pdf_path = Path("./tests/data/pdf/2206.01062.pdf")
    conv_result = converter.convert(pdf_path)
    doc = conv_result.document

    assert len(doc.pages) > 0, "Document should have pages"
    assert len(doc.texts) > 0, "Document should have text elements"


if __name__ == "__main__":
    test_lightonocr_preset_exists()
    test_lightonocr_preset_engine_config()
    test_lightonocr_preset_instantiation()
    test_lightonocr_legacy_specs()
    test_e2e_lightonocr_conversion()
