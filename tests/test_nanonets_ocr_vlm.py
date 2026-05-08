"""Test Nanonets-OCR2-3B VLM integration."""

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

pytestmark = pytest.mark.ml_vlm


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


def test_nanonets_ocr2_preset_engine_config():
    """Verify engine overrides propagate correctly through get_engine_config."""
    preset = VlmConvertOptions.get_preset("nanonets_ocr2")
    spec = preset.model_spec

    # Transformers engine config should carry torch_dtype and model type
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

    # MLX uses a converted qwen2_5_vl-compatible checkpoint on mlx-community
    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "mlx-community/Nanonets-OCR2-3B-bf16"
    assert mlx_config.extra_config == {}
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is True

    # API overrides should have correct model params
    api_overrides = spec.api_overrides
    assert VlmEngineType.API in api_overrides
    assert (
        api_overrides[VlmEngineType.API].params["model"] == "nanonets/Nanonets-OCR2-3B"
    )
    assert api_overrides[VlmEngineType.API].params["max_tokens"] == 15000
    assert VlmEngineType.API_LMSTUDIO in api_overrides
    assert (
        api_overrides[VlmEngineType.API_LMSTUDIO].params["model"] == "nanonets-ocr2-3b"
    )
    assert VlmEngineType.API_OPENAI in api_overrides
    assert api_overrides[VlmEngineType.API_OPENAI].params["model"] == "nanonets-ocr2-3b"


def test_nanonets_ocr2_preset_instantiation():
    """Verify from_preset produces a usable VlmConvertOptions with engine options."""
    options = VlmConvertOptions.from_preset("nanonets_ocr2")
    assert options.model_spec.default_repo_id == "nanonets/Nanonets-OCR2-3B"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN
    assert options.engine_options is not None


def test_nanonets_ocr2_legacy_specs():
    """Verify legacy InlineVlmOptions/ApiVlmOptions specs are consistent."""
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

    # VLLM spec should share repo_id but differ in framework
    vllm_spec = vlm_model_specs.NANONETS_OCR2_VLLM
    assert vllm_spec.repo_id == transformers_spec.repo_id
    assert vllm_spec.inference_framework == InferenceFramework.VLLM
    assert vllm_spec.response_format == transformers_spec.response_format

    # vLLM-compatible API spec
    vllm_api = vlm_model_specs.NANONETS_OCR2_VLLM_API
    assert vllm_api.params["model"] == "nanonets/Nanonets-OCR2-3B"
    assert vllm_api.params["max_tokens"] == 15000
    assert vllm_api.response_format == ResponseFormat.MARKDOWN
    assert vllm_api.concurrency == 4
    assert vllm_api.timeout == 90

    # LM Studio API spec
    lmstudio = vlm_model_specs.NANONETS_OCR2_LMSTUDIO_API
    assert lmstudio.params["model"] == "nanonets-ocr2-3b"
    assert lmstudio.params["max_tokens"] == 15000
    assert lmstudio.response_format == ResponseFormat.MARKDOWN
    assert str(lmstudio.url).startswith("http://localhost:1234")


def test_e2e_nanonets_ocr2_conversion():
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
        vlm_options=vlm_model_specs.NANONETS_OCR2_VLLM_API,
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
    test_nanonets_ocr2_preset_exists()
    test_nanonets_ocr2_preset_engine_config()
    test_nanonets_ocr2_preset_instantiation()
    test_nanonets_ocr2_legacy_specs()
    test_e2e_nanonets_ocr2_conversion()
