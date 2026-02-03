"""Engine options for VLM inference.

This module defines engine-specific configuration options that are independent
of model specifications and prompts.
"""

import logging
from typing import Any, Dict, Literal, Optional

from pydantic import AnyUrl, Field

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.models.runtimes.base import BaseVlmEngineOptions, VlmEngineType

_log = logging.getLogger(__name__)


# =============================================================================
# AUTO_INLINE ENGINE OPTIONS
# =============================================================================


class AutoInlineVlmEngineOptions(BaseVlmEngineOptions):
    """Options for auto-selecting the best local inference engine.

    Automatically selects the best available local engine based on:
    - Platform (macOS -> MLX, Linux/Windows -> Transformers/VLLM)
    - Available hardware (CUDA, MPS, CPU)
    - Model support
    """

    engine_type: Literal[VlmEngineType.AUTO_INLINE] = VlmEngineType.AUTO_INLINE

    prefer_vllm: bool = Field(
        default=False,
        description="Prefer VLLM over Transformers when both are available on CUDA",
    )


# =============================================================================
# TRANSFORMERS ENGINE OPTIONS
# =============================================================================


class TransformersVlmEngineOptions(BaseVlmEngineOptions):
    """Options for HuggingFace Transformers inference engine."""

    engine_type: Literal[VlmEngineType.TRANSFORMERS] = VlmEngineType.TRANSFORMERS

    device: Optional[AcceleratorDevice] = Field(
        default=None, description="Device to use (auto-detected if None)"
    )

    load_in_8bit: bool = Field(
        default=True, description="Load model in 8-bit precision using bitsandbytes"
    )

    llm_int8_threshold: float = Field(
        default=6.0, description="Threshold for LLM.int8() quantization"
    )

    quantized: bool = Field(
        default=False, description="Whether the model is pre-quantized"
    )

    torch_dtype: Optional[str] = Field(
        default=None, description="PyTorch dtype (e.g., 'float16', 'bfloat16')"
    )

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )

    use_kv_cache: bool = Field(
        default=True, description="Enable key-value caching for attention"
    )


# =============================================================================
# MLX ENGINE OPTIONS
# =============================================================================


class MlxVlmEngineOptions(BaseVlmEngineOptions):
    """Options for Apple MLX inference engine (Apple Silicon only)."""

    engine_type: Literal[VlmEngineType.MLX] = VlmEngineType.MLX

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )


# =============================================================================
# VLLM ENGINE OPTIONS
# =============================================================================


class VllmVlmEngineOptions(BaseVlmEngineOptions):
    """Options for vLLM inference engine (high-throughput serving)."""

    engine_type: Literal[VlmEngineType.VLLM] = VlmEngineType.VLLM

    device: Optional[AcceleratorDevice] = Field(
        default=None, description="Device to use (auto-detected if None)"
    )

    tensor_parallel_size: int = Field(
        default=1, description="Number of GPUs for tensor parallelism"
    )

    gpu_memory_utilization: float = Field(
        default=0.9, description="Fraction of GPU memory to use"
    )

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )


# =============================================================================
# API ENGINE OPTIONS
# =============================================================================


class ApiVlmEngineOptions(BaseVlmEngineOptions):
    """Options for API-based VLM services.

    Supports multiple API variants:
    - Generic OpenAI-compatible API
    - Ollama
    - LM Studio
    - OpenAI
    """

    engine_type: VlmEngineType = Field(
        default=VlmEngineType.API, description="API variant to use"
    )

    url: AnyUrl = Field(
        default=AnyUrl("http://localhost:11434/v1/chat/completions"),
        description="API endpoint URL",
    )

    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers for authentication"
    )

    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional API parameters (model, max_tokens, etc.)",
    )

    timeout: float = Field(default=60.0, description="Request timeout in seconds")

    concurrency: int = Field(default=1, description="Number of concurrent requests")

    def __init__(self, **data):
        """Initialize with default URLs based on engine type."""
        if "engine_type" in data and "url" not in data:
            engine_type = data["engine_type"]
            if engine_type == VlmEngineType.API_OLLAMA:
                data["url"] = "http://localhost:11434/v1/chat/completions"
            elif engine_type == VlmEngineType.API_LMSTUDIO:
                data["url"] = "http://localhost:1234/v1/chat/completions"
            elif engine_type == VlmEngineType.API_OPENAI:
                data["url"] = "https://api.openai.com/v1/chat/completions"

        super().__init__(**data)
