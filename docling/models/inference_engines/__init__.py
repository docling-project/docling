"""Inference engine system for Docling.

This package provides a pluggable inference engine system, decoupling
the inference backend from pipeline stages.

Each model family (VLM, object detection, etc.) has its own subfolder
with complete implementation.
"""

# No exports at root level - import from specific model families
# Example: from docling.models.inference_engines.vlm import VlmEngineType

__all__ = []
