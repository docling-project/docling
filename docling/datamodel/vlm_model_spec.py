import logging
from enum import Enum

from pydantic import (
    AnyUrl,
)

from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)

_log = logging.getLogger(__name__)


# SmolDocling
SMOLDOCLING_MLX = InlineVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview-mlx-bf16",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

SMOLDOCLING_TRANSFORMERS = InlineVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.TRANSFORMERS_VISION2SEQ,
    scale=2.0,
    temperature=0.0,
)

# GraniteVision
GRANITE_VISION_TRANSFORMERS = InlineVlmOptions(
    repo_id="ibm-granite/granite-vision-3.2-2b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_VISION2SEQ,
    scale=2.0,
    temperature=0.0,
)

GRANITE_VISION_OLLAMA = ApiVlmOptions(
    url=AnyUrl("http://localhost:11434/v1/chat/completions"),
    params={"model": "granite3.2-vision:2b"},
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    scale=1.0,
    timeout=120,
    response_format=ResponseFormat.MARKDOWN,
    temperature=0.0,
)

# Pixtral
PIXTRAL_12B_TRANSFORMERS = InlineVlmOptions(
    repo_id="mistral-community/pixtral-12b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_VISION2SEQ,
    scale=2.0,
    temperature=0.0,
)

PIXTRAL_12B_MLX = InlineVlmOptions(
    repo_id="mlx-community/pixtral-12b-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

# Phi4
PHI4_TRANSFORMERS = InlineVlmOptions(
    repo_id="microsoft/Phi-4-multimodal-instruct",
    prompt="Convert this page to MarkDown. Do not miss any text and only output the bare markdown",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_CAUSALLM,
    scale=2.0,
    temperature=0.0,
)

# Qwen
QWEN25_VL_3B_MLX = InlineVlmOptions(
    repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

# Gemma-3
GEMMA3_12B_MLX = InlineVlmOptions(
    repo_id="mlx-community/gemma-3-12b-it-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

GEMMA3_27B_MLX = InlineVlmOptions(
    repo_id="mlx-community/gemma-3-27b-it-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)


class VlmModelType(str, Enum):
    SMOLDOCLING = "smoldocling"
    GRANITE_VISION = "granite_vision"
    GRANITE_VISION_OLLAMA = "granite_vision_ollama"
