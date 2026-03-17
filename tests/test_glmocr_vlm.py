"""Test GLM-OCR VLM integration."""

import os
from pathlib import Path

import pytest

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def test_glmocr_preset_exists():
    """Verify preset is registered and retrievable."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "glmocr" in preset_ids

    preset = VlmConvertOptions.get_preset("glmocr")
    assert preset.preset_id == "glmocr"
    assert preset.name == "GLM-OCR"
    assert preset.model_spec.default_repo_id == "zai-org/GLM-OCR"
    assert preset.model_spec.response_format == ResponseFormat.MARKDOWN
    assert preset.model_spec.prompt == "Text Recognition:"


def test_glmocr_preset_instantiation():
    """Verify VlmConvertOptions.from_preset('glmocr') works."""
    options = VlmConvertOptions.from_preset("glmocr")
    assert options is not None
    assert options.model_spec.default_repo_id == "zai-org/GLM-OCR"
    assert options.model_spec.prompt == "Text Recognition:"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN


def test_glmocr_legacy_specs():
    """Verify legacy InlineVlmOptions/ApiVlmOptions specs are accessible."""
    # Transformers spec
    t = vlm_model_specs.GLMOCR_TRANSFORMERS
    assert t.repo_id == "zai-org/GLM-OCR"
    assert t.inference_framework == InferenceFramework.TRANSFORMERS
    assert t.response_format == ResponseFormat.MARKDOWN
    assert t.torch_dtype == "bfloat16"

    # VLLM spec
    v = vlm_model_specs.GLMOCR_VLLM
    assert v.repo_id == "zai-org/GLM-OCR"
    assert v.inference_framework == InferenceFramework.VLLM

    # API spec
    a = vlm_model_specs.GLMOCR_VLLM_API
    assert a.params["model"] == "zai-org/GLM-OCR"
    assert a.response_format == ResponseFormat.MARKDOWN


def test_e2e_glmocr_conversion():
    """E2E test with vLLM server (skipped in CI and when server is unavailable)."""
    if os.getenv("CI"):
        pytest.skip("Skipping in CI environment")

    try:
        import requests

        response = requests.get(
            "http://localhost:8000/v1/models", timeout=2
        )
        if response.status_code != 200:
            pytest.skip("vLLM server is not available")
    except Exception:
        pytest.skip("vLLM server is not available")

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GLMOCR_VLLM_API,
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
    test_glmocr_preset_exists()
    test_glmocr_preset_instantiation()
    test_glmocr_legacy_specs()
    test_e2e_glmocr_conversion()
