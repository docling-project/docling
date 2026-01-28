"""Runtime options for VLM inference.

This module defines runtime-specific configuration options that are independent
of model specifications and prompts.
"""

import logging
from typing import Any, Dict, Literal, Optional

from pydantic import AnyUrl, Field

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.models.runtimes.base import BaseVlmRuntimeOptions, VlmRuntimeType

_log = logging.getLogger(__name__)


# =============================================================================
# AUTO_INLINE RUNTIME OPTIONS
# =============================================================================


class AutoInlineVlmRuntimeOptions(BaseVlmRuntimeOptions):
    """Options for auto-selecting the best local runtime.

    Automatically selects the best available local runtime based on:
    - Platform (macOS -> MLX, Linux/Windows -> Transformers/VLLM)
    - Available hardware (CUDA, MPS, CPU)
    - Model support
    """

    runtime_type: Literal[VlmRuntimeType.AUTO_INLINE] = VlmRuntimeType.AUTO_INLINE

    prefer_vllm: bool = Field(
        default=False,
        description="Prefer VLLM over Transformers when both are available on CUDA",
    )


# =============================================================================
# TRANSFORMERS RUNTIME OPTIONS
# =============================================================================


class TransformersVlmRuntimeOptions(BaseVlmRuntimeOptions):
    """Options for HuggingFace Transformers runtime."""

    runtime_type: Literal[VlmRuntimeType.TRANSFORMERS] = VlmRuntimeType.TRANSFORMERS

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
# MLX RUNTIME OPTIONS
# =============================================================================


class MlxVlmRuntimeOptions(BaseVlmRuntimeOptions):
    """Options for Apple MLX runtime (Apple Silicon only)."""

    runtime_type: Literal[VlmRuntimeType.MLX] = VlmRuntimeType.MLX

    trust_remote_code: bool = Field(
        default=False, description="Allow execution of custom code from model repo"
    )


# =============================================================================
# VLLM RUNTIME OPTIONS
# =============================================================================


class VllmVlmRuntimeOptions(BaseVlmRuntimeOptions):
    """Options for vLLM runtime (high-throughput serving)."""

    runtime_type: Literal[VlmRuntimeType.VLLM] = VlmRuntimeType.VLLM

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
# API RUNTIME OPTIONS
# =============================================================================


class ApiVlmRuntimeOptions(BaseVlmRuntimeOptions):
    """Options for API-based VLM services.

    Supports multiple API variants:
    - Generic OpenAI-compatible API
    - Ollama
    - LM Studio
    - OpenAI
    """

    runtime_type: VlmRuntimeType = Field(
        default=VlmRuntimeType.API, description="API variant to use"
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
        """Initialize with default URLs based on runtime type."""
        if "runtime_type" in data and "url" not in data:
            runtime_type = data["runtime_type"]
            if runtime_type == VlmRuntimeType.API_OLLAMA:
                data["url"] = "http://localhost:11434/v1/chat/completions"
            elif runtime_type == VlmRuntimeType.API_LMSTUDIO:
                data["url"] = "http://localhost:1234/v1/chat/completions"
            elif runtime_type == VlmRuntimeType.API_OPENAI:
                data["url"] = "https://api.openai.com/v1/chat/completions"

        super().__init__(**data)
