# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test MiniCPM-OCR VLM integration."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    AutoInlineVlmEngineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from tests.fakes.http_service import FakeService
from tests.fakes.openai_compatible import FakeOpenAiApi

pytestmark = pytest.mark.ml_vlm

_MINICPM_OCR_REVISION = "c5833cddb363aa62f1cb60ddde48c832e8f1a92c"
_API_ENGINES = {
    VlmEngineType.API,
    VlmEngineType.API_OPENAI,
    VlmEngineType.API_OLLAMA,
    VlmEngineType.API_LMSTUDIO,
}


def test_minicpm_ocr_preset_exists():
    """Verify preset is registered with the expected model and generation contract."""
    preset_ids = VlmConvertOptions.list_preset_ids()
    assert "minicpm_ocr" in preset_ids

    preset = VlmConvertOptions.get_preset("minicpm_ocr")
    assert preset.preset_id == "minicpm_ocr"
    assert preset.name == "MiniCPM-OCR"
    assert preset.scale == 2.0
    assert preset.default_engine_type == VlmEngineType.AUTO_INLINE

    spec = preset.model_spec
    assert spec.default_repo_id == "openbmb/MiniCPM-OCR-1.0"
    assert spec.revision == _MINICPM_OCR_REVISION
    assert spec.response_format == ResponseFormat.MARKDOWN
    assert spec.prompt == "Convert this page to markdown."
    assert spec.max_new_tokens == 4096
    assert spec.stop_strings == ["<|im_end|>", "<|endoftext|>"]
    assert spec.extra_generation_config == {"skip_special_tokens": True}
    assert spec.trust_remote_code is False


def test_minicpm_ocr_preset_engine_config():
    """Verify engine overrides propagate through get_engine_config."""
    preset = VlmConvertOptions.get_preset("minicpm_ocr")
    spec = preset.model_spec

    tf_config = spec.get_engine_config(VlmEngineType.TRANSFORMERS)
    assert tf_config.repo_id == "openbmb/MiniCPM-OCR-1.0"
    assert tf_config.revision == _MINICPM_OCR_REVISION
    assert tf_config.torch_dtype == "bfloat16"
    assert tf_config.min_engine_version == "5.21.0"
    assert (
        tf_config.extra_config["transformers_model_type"]
        == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
    )
    assert (
        tf_config.extra_config["transformers_prompt_style"]
        == TransformersPromptStyle.CHAT
    )
    assert tf_config.extra_config["transformers_strip_stop_strings"] is True
    assert tf_config.extra_config["extra_processor_kwargs"] == {"downsample_mode": "4x"}

    # Runtime-level generation settings must carry the decoder options.
    runtime_config = spec.get_runtime_input_extra_config(VlmEngineType.TRANSFORMERS)
    assert runtime_config["skip_special_tokens"] is True
    assert runtime_config["no_repeat_ngram_size"] == 100
    assert runtime_config["transformers_prompt_style"] == TransformersPromptStyle.CHAT
    assert runtime_config["extra_processor_kwargs"] == {"downsample_mode": "4x"}

    # The model only declares the engines that were verified.
    assert spec.has_explicit_engine_export(VlmEngineType.TRANSFORMERS) is True
    assert spec.has_explicit_engine_export(VlmEngineType.MLX) is False
    assert spec.has_explicit_engine_export(VlmEngineType.VLLM) is False

    for engine_type in _API_ENGINES:
        params = spec.get_api_params(engine_type)
        assert params["model"] == "minicpm-ocr"
        assert params["max_tokens"] == 4096


def test_minicpm_ocr_preset_instantiation():
    """Verify from_preset produces usable options with engine configuration."""
    options = VlmConvertOptions.from_preset("minicpm_ocr")
    assert options.model_spec.default_repo_id == "openbmb/MiniCPM-OCR-1.0"
    assert options.model_spec.response_format == ResponseFormat.MARKDOWN
    assert options.engine_options is not None


@pytest.fixture
def api() -> Iterator[FakeOpenAiApi]:
    service = FakeService()
    fake = FakeOpenAiApi(completion="MiniCPM-OCR fake page.")
    service.include(fake.router)
    service.start()
    fake.service = service
    try:
        yield fake
    finally:
        service.stop()


def test_minicpm_ocr_api_preset_request(api: FakeOpenAiApi, tmp_path: Path):
    """Verify the preset's API overrides govern the outgoing request."""
    pytest.importorskip("marko")
    image_path = tmp_path / "page.png"
    Image.new("RGB", (64, 64), "white").save(image_path)

    endpoint = AnyUrl(f"{api.service.base_url}/v1/chat/completions")
    options = VlmConvertOptions.from_preset(
        "minicpm_ocr",
        engine_options=ApiVlmEngineOptions(
            engine_type=VlmEngineType.API,
            url=endpoint,
        ),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(
                    vlm_options=options,
                    enable_remote_services=True,
                ),
            ),
        }
    )

    conv_result = converter.convert(image_path)
    assert conv_result.status == ConversionStatus.SUCCESS

    requests = api.service.requests_for("POST", r"/v1/chat/completions")
    assert requests, "no call reached the API"
    body = requests[-1].json()
    assert body["model"] == "minicpm-ocr"
    assert body["max_tokens"] == 4096
    assert body["stop"] == ["<|im_end|>", "<|endoftext|>"]

    content = body["messages"][0]["content"]
    text_part = next(part for part in content if part["type"] == "text")
    assert text_part["text"] == "Convert this page to markdown."


@pytest.mark.skipif(
    os.getenv("DOCLING_TEST_MINICPM_OCR") != "1",
    reason="set DOCLING_TEST_MINICPM_OCR=1 to run the MiniCPM-OCR inference test",
)
def test_e2e_minicpm_ocr_conversion():
    """Run MiniCPM-OCR on a bundled PDF with the local Transformers engine."""
    options = VlmConvertOptions.from_preset(
        "minicpm_ocr",
        engine_options=AutoInlineVlmEngineOptions(),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(vlm_options=options),
            ),
        }
    )

    pdf_path = Path("tests/data/pdf/sources/table_mislabeled_as_picture.pdf")
    conv_result = converter.convert(pdf_path)
    doc = conv_result.document

    # SUCCESS excludes PARTIAL_SUCCESS, which the pipeline sets for a page
    # with an inference error or a length-limited response.
    assert conv_result.status == ConversionStatus.SUCCESS
    assert len(doc.pages) > 0, "Document should have pages"
    assert len(doc.texts) > 0, "Document should have text elements"
