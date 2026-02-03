"""Factory for creating VLM inference engines."""

import logging
from typing import TYPE_CHECKING, Optional

from docling.models.runtimes.base import (
    BaseVlmEngine,
    BaseVlmEngineOptions,
    VlmEngineType,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig, VlmModelSpec
    from docling.datamodel.vlm_engine_options import (
        ApiVlmEngineOptions,
        AutoInlineVlmEngineOptions,
        MlxVlmEngineOptions,
        TransformersVlmEngineOptions,
        VllmVlmEngineOptions,
    )

_log = logging.getLogger(__name__)


def create_vlm_engine(
    options: BaseVlmEngineOptions,
    model_spec: Optional["VlmModelSpec"] = None,
) -> BaseVlmEngine:
    """Create a VLM inference engine from options.

    Args:
        options: Engine configuration options
        model_spec: Model specification (for generating engine-specific configs)

    Returns:
        Initialized engine instance

    Raises:
        ValueError: If engine type is not supported
        ImportError: If required dependencies are not installed
    """
    engine_type = options.engine_type

    # Generate model_config from model_spec if provided
    model_config: Optional[EngineModelConfig] = None
    if model_spec is not None and engine_type != VlmEngineType.AUTO_INLINE:
        # AUTO_INLINE handles model_spec internally
        model_config = model_spec.get_engine_config(engine_type)

        # For API engines, add API params to extra_config
        if VlmEngineType.is_api_variant(engine_type):
            api_params = model_spec.get_api_params(engine_type)
            model_config.extra_config["api_params"] = api_params

    if engine_type == VlmEngineType.AUTO_INLINE:
        from docling.datamodel.vlm_engine_options import AutoInlineVlmEngineOptions
        from docling.models.runtimes.vlm.auto_inline_engine import (
            AutoInlineVlmEngine,
        )

        if not isinstance(options, AutoInlineVlmEngineOptions):
            raise ValueError(
                f"Expected AutoInlineVlmEngineOptions, got {type(options)}"
            )
        return AutoInlineVlmEngine(options, model_spec=model_spec)

    elif engine_type == VlmEngineType.TRANSFORMERS:
        from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
        from docling.models.runtimes.vlm.transformers_engine import (
            TransformersVlmEngine,
        )

        if not isinstance(options, TransformersVlmEngineOptions):
            raise ValueError(
                f"Expected TransformersVlmEngineOptions, got {type(options)}"
            )
        return TransformersVlmEngine(options, model_config=model_config)

    elif engine_type == VlmEngineType.MLX:
        from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
        from docling.models.runtimes.vlm.mlx_engine import MlxVlmEngine

        if not isinstance(options, MlxVlmEngineOptions):
            raise ValueError(f"Expected MlxVlmEngineOptions, got {type(options)}")
        return MlxVlmEngine(options, model_config=model_config)

    elif engine_type == VlmEngineType.VLLM:
        from docling.datamodel.vlm_engine_options import VllmVlmEngineOptions
        from docling.models.runtimes.vlm.vllm_engine import VllmVlmEngine

        if not isinstance(options, VllmVlmEngineOptions):
            raise ValueError(f"Expected VllmVlmEngineOptions, got {type(options)}")
        return VllmVlmEngine(options, model_config=model_config)

    elif VlmEngineType.is_api_variant(engine_type):
        from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
        from docling.models.runtimes.vlm.api_openai_compatible_engine import (
            ApiVlmEngine,
        )

        if not isinstance(options, ApiVlmEngineOptions):
            raise ValueError(f"Expected ApiVlmEngineOptions, got {type(options)}")
        return ApiVlmEngine(options, model_config=model_config)

    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")
