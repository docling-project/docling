"""VLM Runtime system for Docling.

This package provides a pluggable runtime system for vision-language models,
decoupling the inference backend from pipeline stages.
"""

from docling.models.runtimes.base import (
    BaseVlmRuntime,
    BaseVlmRuntimeOptions,
    VlmRuntimeType,
)
from docling.models.runtimes.factory import create_vlm_runtime

__all__ = [
    "BaseVlmRuntime",
    "BaseVlmRuntimeOptions",
    "VlmRuntimeType",
    "create_vlm_runtime",
]
