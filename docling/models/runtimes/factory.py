"""Factory for creating VLM runtimes."""

import logging
from typing import TYPE_CHECKING

from docling.models.runtimes.base import (
    BaseVlmRuntime,
    BaseVlmRuntimeOptions,
    VlmRuntimeType,
)

if TYPE_CHECKING:
    from docling.models.runtimes.api_runtime import ApiVlmRuntimeOptions
    from docling.models.runtimes.auto_inline_runtime import AutoInlineVlmRuntimeOptions
    from docling.models.runtimes.mlx_runtime import MlxVlmRuntimeOptions
    from docling.models.runtimes.transformers_runtime import (
        TransformersVlmRuntimeOptions,
    )
    from docling.models.runtimes.vllm_runtime import VllmVlmRuntimeOptions

_log = logging.getLogger(__name__)


def create_vlm_runtime(options: BaseVlmRuntimeOptions) -> BaseVlmRuntime:
    """Create a VLM runtime from options.

    Args:
        options: Runtime configuration options

    Returns:
        Initialized runtime instance

    Raises:
        ValueError: If runtime type is not supported
        ImportError: If required dependencies are not installed
    """
    runtime_type = options.runtime_type

    if runtime_type == VlmRuntimeType.AUTO_INLINE:
        from docling.models.runtimes.auto_inline_runtime import (
            AutoInlineVlmRuntime,
            AutoInlineVlmRuntimeOptions,
        )

        if not isinstance(options, AutoInlineVlmRuntimeOptions):
            raise ValueError(
                f"Expected AutoInlineVlmRuntimeOptions, got {type(options)}"
            )
        return AutoInlineVlmRuntime(options)

    elif runtime_type == VlmRuntimeType.TRANSFORMERS:
        from docling.models.runtimes.transformers_runtime import (
            TransformersVlmRuntime,
            TransformersVlmRuntimeOptions,
        )

        if not isinstance(options, TransformersVlmRuntimeOptions):
            raise ValueError(
                f"Expected TransformersVlmRuntimeOptions, got {type(options)}"
            )
        return TransformersVlmRuntime(options)

    elif runtime_type == VlmRuntimeType.MLX:
        from docling.models.runtimes.mlx_runtime import (
            MlxVlmRuntime,
            MlxVlmRuntimeOptions,
        )

        if not isinstance(options, MlxVlmRuntimeOptions):
            raise ValueError(f"Expected MlxVlmRuntimeOptions, got {type(options)}")
        return MlxVlmRuntime(options)

    elif runtime_type == VlmRuntimeType.VLLM:
        from docling.models.runtimes.vllm_runtime import (
            VllmVlmRuntime,
            VllmVlmRuntimeOptions,
        )

        if not isinstance(options, VllmVlmRuntimeOptions):
            raise ValueError(f"Expected VllmVlmRuntimeOptions, got {type(options)}")
        return VllmVlmRuntime(options)

    elif VlmRuntimeType.is_api_variant(runtime_type):
        from docling.models.runtimes.api_runtime import (
            ApiVlmRuntime,
            ApiVlmRuntimeOptions,
        )

        if not isinstance(options, ApiVlmRuntimeOptions):
            raise ValueError(f"Expected ApiVlmRuntimeOptions, got {type(options)}")
        return ApiVlmRuntime(options)

    else:
        raise ValueError(f"Unsupported runtime type: {runtime_type}")
