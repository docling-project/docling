"""Auto-inline VLM runtime that selects the best local runtime."""

import logging
import platform
from typing import Optional

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.vlm_runtime_options import (
    AutoInlineVlmRuntimeOptions,
    MlxVlmRuntimeOptions,
    TransformersVlmRuntimeOptions,
    VllmVlmRuntimeOptions,
)
from docling.models.runtimes.base import (
    BaseVlmRuntime,
    VlmRuntimeInput,
    VlmRuntimeOutput,
    VlmRuntimeType,
)
from docling.utils.accelerator_utils import decide_device

_log = logging.getLogger(__name__)


class AutoInlineVlmRuntime(BaseVlmRuntime):
    """Auto-selecting runtime that picks the best local runtime.

    Selection logic:
    1. On macOS with Apple Silicon (MPS available) -> MLX
    2. On Linux/Windows with CUDA and prefer_vllm=True -> vLLM
    3. Otherwise -> Transformers

    This runtime delegates to the selected runtime after initialization.
    """

    def __init__(
        self,
        options: AutoInlineVlmRuntimeOptions,
        accelerator_options: Optional[AcceleratorOptions] = None,
        artifacts_path=None,
    ):
        """Initialize the auto-inline runtime.

        Args:
            options: Auto-inline runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
        """
        super().__init__(options)
        self.options: AutoInlineVlmRuntimeOptions = options
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.artifacts_path = artifacts_path

        # The actual runtime will be set during initialization
        self.actual_runtime: Optional[BaseVlmRuntime] = None
        self.selected_runtime_type: Optional[VlmRuntimeType] = None

    def _select_runtime(self) -> VlmRuntimeType:
        """Select the best runtime based on platform and hardware.

        Returns:
            The selected runtime type
        """
        system = platform.system()

        # Detect available device
        device = decide_device(
            self.accelerator_options.device,
            supported_devices=[
                AcceleratorDevice.CPU,
                AcceleratorDevice.CUDA,
                AcceleratorDevice.MPS,
                AcceleratorDevice.XPU,
            ],
        )

        _log.info(f"Auto-selecting runtime for system={system}, device={device}")

        # macOS with Apple Silicon -> MLX
        if system == "Darwin" and device == "mps":
            try:
                import mlx_vlm

                _log.info("Selected MLX runtime (Apple Silicon detected)")
                return VlmRuntimeType.MLX
            except ImportError:
                _log.warning(
                    "MLX not available on Apple Silicon, falling back to Transformers"
                )

        # CUDA with prefer_vllm -> vLLM
        if device.startswith("cuda") and self.options.prefer_vllm:
            try:
                import vllm

                _log.info("Selected vLLM runtime (CUDA + prefer_vllm=True)")
                return VlmRuntimeType.VLLM
            except ImportError:
                _log.warning("vLLM not available, falling back to Transformers")

        # Default to Transformers
        _log.info("Selected Transformers runtime (default)")
        return VlmRuntimeType.TRANSFORMERS

    def initialize(self) -> None:
        """Initialize by selecting and creating the actual runtime."""
        if self._initialized:
            return

        _log.info("Initializing auto-inline VLM runtime...")

        # Select the best runtime
        self.selected_runtime_type = self._select_runtime()

        # Create the actual runtime
        if self.selected_runtime_type == VlmRuntimeType.MLX:
            from docling.models.runtimes.mlx_runtime import MlxVlmRuntime

            mlx_options = MlxVlmRuntimeOptions(
                trust_remote_code=self.options.trust_remote_code
                if hasattr(self.options, "trust_remote_code")
                else False,
            )
            self.actual_runtime = MlxVlmRuntime(
                options=mlx_options,
                artifacts_path=self.artifacts_path,
            )

        elif self.selected_runtime_type == VlmRuntimeType.VLLM:
            from docling.models.runtimes.vllm_runtime import VllmVlmRuntime

            vllm_options = VllmVlmRuntimeOptions()
            self.actual_runtime = VllmVlmRuntime(
                options=vllm_options,
                accelerator_options=self.accelerator_options,
                artifacts_path=self.artifacts_path,
            )

        else:  # TRANSFORMERS
            from docling.models.runtimes.transformers_runtime import (
                TransformersVlmRuntime,
            )

            transformers_options = TransformersVlmRuntimeOptions()
            self.actual_runtime = TransformersVlmRuntime(
                options=transformers_options,
                accelerator_options=self.accelerator_options,
                artifacts_path=self.artifacts_path,
            )

        # Initialize the actual runtime
        self.actual_runtime.initialize()

        self._initialized = True
        _log.info(
            f"Auto-inline runtime initialized with {self.selected_runtime_type.value}"
        )

    def predict(self, input_data: VlmRuntimeInput) -> VlmRuntimeOutput:
        """Run inference using the selected runtime.

        Args:
            input_data: Input containing image, prompt, and configuration

        Returns:
            Generated text and metadata
        """
        if not self._initialized:
            self.initialize()

        assert self.actual_runtime is not None, "Runtime not initialized"

        # Delegate to the actual runtime
        return self.actual_runtime.predict(input_data)

    def cleanup(self) -> None:
        """Clean up the actual runtime resources."""
        if self.actual_runtime is not None:
            self.actual_runtime.cleanup()
            self.actual_runtime = None

        _log.info("Auto-inline runtime cleaned up")
