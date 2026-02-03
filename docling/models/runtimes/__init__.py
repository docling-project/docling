"""VLM inference engine system for Docling.

This package provides a pluggable inference engine system for vision-language models,
decoupling the inference backend from pipeline stages.
"""

from docling.models.runtimes.base import (
    BaseVlmEngine,
    BaseVlmEngineOptions,
    VlmEngineType,
)
from docling.models.runtimes.factory import create_vlm_engine

__all__ = [
    "BaseVlmEngine",
    "BaseVlmEngineOptions",
    "VlmEngineType",
    "create_vlm_engine",
]
