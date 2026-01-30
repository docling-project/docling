"""Auto-inline VLM runtime that selects the best local runtime."""

import logging
import platform
from typing import TYPE_CHECKING, List, Optional

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

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import RuntimeModelConfig, VlmModelSpec

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
        model_spec: Optional["VlmModelSpec"] = None,
    ):
        """Initialize the auto-inline runtime.

        Args:
            options: Auto-inline runtime options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_spec: Model specification (for generating runtime-specific configs)
        """
        super().__init__(options, model_config=None)
        self.options: AutoInlineVlmRuntimeOptions = options
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.artifacts_path = artifacts_path
        self.model_spec = model_spec

        # The actual runtime will be set during initialization
        self.actual_runtime: Optional[BaseVlmRuntime] = None
        self.selected_runtime_type: Optional[VlmRuntimeType] = None

        # Initialize immediately if model_spec is provided
        if self.model_spec is not None:
            self.initialize()

    def _select_runtime(self) -> VlmRuntimeType:
        """Select the best runtime based on platform and hardware.

        Respects model's supported_runtimes if model_spec is provided.

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

        # Get supported runtimes from model_spec if available
        supported_runtimes = None
        if self.model_spec is not None:
            supported_runtimes = self.model_spec.supported_runtimes

        # macOS with Apple Silicon -> MLX (if supported)
        if system == "Darwin" and device == "mps":
            if supported_runtimes is None or VlmRuntimeType.MLX in supported_runtimes:
                try:
                    import mlx_vlm

                    _log.info("Selected MLX runtime (Apple Silicon detected)")
                    return VlmRuntimeType.MLX
                except ImportError:
                    _log.warning(
                        "MLX not available on Apple Silicon, falling back to Transformers"
                    )
            else:
                _log.info("MLX not in supported_runtimes, skipping")

        # CUDA with prefer_vllm -> vLLM (if supported)
        if device.startswith("cuda") and self.options.prefer_vllm:
            if supported_runtimes is None or VlmRuntimeType.VLLM in supported_runtimes:
                try:
                    import vllm

                    _log.info("Selected vLLM runtime (CUDA + prefer_vllm=True)")
                    return VlmRuntimeType.VLLM
                except ImportError:
                    _log.warning("vLLM not available, falling back to Transformers")
            else:
                _log.info("vLLM not in supported_runtimes, skipping")

        # Default to Transformers (should always be supported)
        _log.info("Selected Transformers runtime (default)")
        return VlmRuntimeType.TRANSFORMERS

    def initialize(self) -> None:
        """Initialize by selecting and creating the actual runtime."""
        if self._initialized:
            return

        _log.info("Initializing auto-inline VLM runtime...")

        # Select the best runtime
        self.selected_runtime_type = self._select_runtime()

        # Generate model_config for the selected runtime
        model_config = None
        if self.model_spec is not None:
            model_config = self.model_spec.get_runtime_config(
                self.selected_runtime_type
            )
            _log.info(
                f"Generated config for {self.selected_runtime_type.value}: "
                f"repo_id={model_config.repo_id}, extra_config={model_config.extra_config}"
            )

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
                model_config=model_config,
            )

        elif self.selected_runtime_type == VlmRuntimeType.VLLM:
            from docling.models.runtimes.vllm_runtime import VllmVlmRuntime

            vllm_options = VllmVlmRuntimeOptions()
            self.actual_runtime = VllmVlmRuntime(
                options=vllm_options,
                accelerator_options=self.accelerator_options,
                artifacts_path=self.artifacts_path,
                model_config=model_config,
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
                model_config=model_config,
            )

        # Note: actual_runtime.initialize() is called automatically in their __init__
        # if model_config is provided

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

    def predict_batch(
        self, input_batch: List[VlmRuntimeInput]
    ) -> List[VlmRuntimeOutput]:
        """Run inference on a batch of inputs using the selected runtime.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        assert self.actual_runtime is not None, "Runtime not initialized"

        # Delegate to the actual runtime's batch implementation
        return self.actual_runtime.predict_batch(input_batch)

    def cleanup(self) -> None:
        """Clean up the actual runtime resources."""
        if self.actual_runtime is not None:
            self.actual_runtime.cleanup()
            self.actual_runtime = None

        _log.info("Auto-inline runtime cleaned up")
