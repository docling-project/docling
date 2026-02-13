"""Shared inference-engine utilities."""

from docling.models.inference_engines.common.hf_vision_base import HfVisionModelMixin
from docling.models.inference_engines.common.kserve_v2_http import KserveV2HttpClient

__all__ = ["HfVisionModelMixin", "KserveV2HttpClient"]
