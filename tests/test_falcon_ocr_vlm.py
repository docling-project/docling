"""Test Falcon-OCR VLM integration."""

import os
from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.models.inference_engines.vlm.transformers_runtime_adapters import (
    FalconOCRTransformersAdapter,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

pytestmark = pytest.mark.skipif(
    os.getenv("CI"),
    reason="Skipping VLM unit tests in CI",
)


def test_falcon_ocr_preset_exists():
    """Verify preset is registered with correct metadata and model spec."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "falcon_ocr" in preset_ids

    preset = VlmConvertOptions.get_preset("falcon_ocr")
    assert preset.preset_id == "falcon_ocr"
    assert preset.name == "Falcon-OCR"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == "tiiuae/Falcon-OCR"
    assert spec.response_format == ResponseFormat.MARKDOWN
    assert spec.prompt == ""
    assert spec.trust_remote_code is True


def test_falcon_ocr_preset_engine_config():
    """Verify engine overrides propagate correctly through get_engine_config."""
    preset = VlmConvertOptions.get_preset("falcon_ocr")
    spec = preset.model_spec

    # Transformers engine config should carry torch_dtype and model type
    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == "tiiuae/Falcon-OCR"
    assert tf_config.extra_config["torch_dtype"] == "bfloat16"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_CAUSALLM
    )
    assert tf_config.extra_config["attn_implementation"] == "eager"
    assert (
        tf_config.extra_config["transformers_runtime_adapter"]
        == FalconOCRTransformersAdapter
    )

    # API overrides should have correct model params
    api_overrides = spec.api_overrides
    assert VlmEngineType.API_LMSTUDIO in api_overrides
    assert api_overrides[VlmEngineType.API_LMSTUDIO].params["model"] == "falcon-ocr"
    assert api_overrides[VlmEngineType.API_LMSTUDIO].params["max_tokens"] == 4096
    assert VlmEngineType.API_OPENAI in api_overrides
    assert api_overrides[VlmEngineType.API_OPENAI].params["model"] == "falcon-ocr"

    # Falcon-OCR now uses the dedicated mlx-community export.
    mlx_config = spec.get_engine_config(VlmEngineType.MLX)
    assert mlx_config.repo_id == "mlx-community/Falcon-OCR-bf16"
    assert mlx_config.extra_config == {}
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is True


def test_falcon_ocr_preset_instantiation():
    """Verify from_preset produces a usable VlmConvertOptions with engine options."""
    options = VlmConvertOptions.from_preset("falcon_ocr")
    assert options.model_spec.default_repo_id == "tiiuae/Falcon-OCR"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN
    assert options.engine_options is not None
    assert (
        options.extra_generation_config["transformers_prompt_style"]
        == TransformersPromptStyle.RAW
    )
    assert (
        options.build_prompt("Keep formulas")
        == "<|image|>Keep formulas\n<|OCR_PLAIN|>"
    )
    assert (
        options.build_prompt("")
        == "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
    )


def test_e2e_falcon_ocr_conversion():
    """E2E test with an OpenAI-compatible Falcon-OCR server."""
    try:
        import requests

        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        if response.status_code != 200:
            pytest.skip("vLLM server is not available")
    except Exception:
        pytest.skip("vLLM server is not available")

    pipeline_options = VlmPipelineOptions(
        vlm_options=VlmConvertOptions.from_preset(
            "falcon_ocr",
            engine_options=ApiVlmEngineOptions(
                engine_type=VlmEngineType.API_OPENAI,
                url="http://localhost:8000/v1/chat/completions",
                timeout=90.0,
                concurrency=4,
            ),
        ),
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
    test_falcon_ocr_preset_exists()
    test_falcon_ocr_preset_engine_config()
    test_falcon_ocr_preset_instantiation()
    test_e2e_falcon_ocr_conversion()
