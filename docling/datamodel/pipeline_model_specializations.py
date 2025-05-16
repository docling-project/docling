import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import (
    AnyUrl,
    BaseModel,
)

_log = logging.getLogger(__name__)


class BaseVlmOptions(BaseModel):
    kind: str
    prompt: str


class ResponseFormat(str, Enum):
    DOCTAGS = "doctags"
    MARKDOWN = "markdown"
    HTML = "html"


class InferenceFramework(str, Enum):
    MLX = "mlx"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    TRANSFORMERS_AutoModelForVision2Seq = "transformers-AutoModelForVision2Seq"
    TRANSFORMERS_AutoModelForCausalLM = "transformers-AutoModelForCausalLM"
    TRANSFORMERS_LlavaForConditionalGeneration = (
        "transformers-LlavaForConditionalGeneration"
    )


class HuggingFaceVlmOptions(BaseVlmOptions):
    kind: Literal["hf_model_options"] = "hf_model_options"

    repo_id: str
    load_in_8bit: bool = True
    llm_int8_threshold: float = 6.0
    quantized: bool = False

    inference_framework: InferenceFramework
    response_format: ResponseFormat

    scale: float = 2.0 

    temperature: float = 0.0
    stop_strings: list[str] = []
    
    use_kv_cache: bool = True
    max_new_tokens: int = 4096

    @property
    def repo_cache_folder(self) -> str:
        return self.repo_id.replace("/", "--")


class ApiVlmOptions(BaseVlmOptions):
    kind: Literal["api_model_options"] = "api_model_options"

    url: AnyUrl = AnyUrl(
        "http://localhost:11434/v1/chat/completions"
    )  # Default to ollama
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    scale: float = 2.0
    timeout: float = 60
    concurrency: int = 1
    response_format: ResponseFormat


class VlmModelType(str, Enum):
    SMOLDOCLING = "smoldocling"
    GRANITE_VISION = "granite_vision"
    GRANITE_VISION_OLLAMA = "granite_vision_ollama"


# SmolDocling
smoldocling_vlm_mlx_conversion_options = HuggingFaceVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview-mlx-bf16",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

smoldocling_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="ds4sd/SmolDocling-256M-preview",
    prompt="Convert this page to docling.",
    response_format=ResponseFormat.DOCTAGS,
    inference_framework=InferenceFramework.TRANSFORMERS_AutoModelForVision2Seq,
    scale=2.0,
    temperature=0.0,
)

# GraniteVision
granite_vision_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="ibm-granite/granite-vision-3.2-2b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_AutoModelForVision2Seq,
    scale=2.0,
    temperature=0.0,
)

granite_vision_vlm_ollama_conversion_options = ApiVlmOptions(
    url=AnyUrl("http://localhost:11434/v1/chat/completions"),
    params={"model": "granite3.2-vision:2b"},
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    scale=1.0,
    timeout=120,
    response_format=ResponseFormat.MARKDOWN,
    temperature=0.0,
)

# Pixtral
pixtral_12b_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="mistral-community/pixtral-12b",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_LlavaForConditionalGeneration,
    scale=2.0,
    temperature=0.0,
)

pixtral_12b_vlm_mlx_conversion_options = HuggingFaceVlmOptions(
    repo_id="mlx-community/pixtral-12b-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)

# Phi4
phi_vlm_conversion_options = HuggingFaceVlmOptions(
    repo_id="microsoft/Phi-4-multimodal-instruct",
    prompt="Convert this page to MarkDown. Do not miss any text and only output the bare MarkDown",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.TRANSFORMERS_AutoModelForCausalLM,
    scale=2.0,
    temperature=0.0,
)

# Qwen
qwen25_vl_3b_vlm_mlx_conversion_options = HuggingFaceVlmOptions(
    repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    prompt="Convert this page to markdown. Do not miss any text and only output the bare MarkDown!",
    response_format=ResponseFormat.MARKDOWN,
    inference_framework=InferenceFramework.MLX,
    scale=2.0,
    temperature=0.0,
)
