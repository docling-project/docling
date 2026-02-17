"""VLM (Vision-Language Model) inference engines."""

# Import base classes and types (no circular dependency)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    BaseVlmEngineOptions,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
)

# Import factory (no circular dependency)
from docling.models.inference_engines.vlm.factory import create_vlm_engine

# Engine implementations are NOT imported here to avoid circular imports
# They can be imported directly when needed:
#   from docling.models.inference_engines.vlm.transformers_engine import TransformersVlmEngine
# Or accessed via the factory:
#   engine = create_vlm_engine(options)

__all__ = [
    # Base classes and types
    "BaseVlmEngine",
    "BaseVlmEngineOptions",
    "VlmEngineInput",
    "VlmEngineOutput",
    "VlmEngineType",
    # Factory
    "create_vlm_engine",
    # Note: Engine implementations are not exported to avoid circular imports
    # Import them directly from their modules if needed
]
