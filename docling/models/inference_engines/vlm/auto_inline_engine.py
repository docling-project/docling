"""Auto-inline VLM inference engine that selects the best local engine."""

import logging
import platform
from typing import TYPE_CHECKING, List, Optional

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.vlm_engine_options import (
    AutoInlineVlmEngineOptions,
    MlxVlmEngineOptions,
    TransformersVlmEngineOptions,
    VllmVlmEngineOptions,
)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
    VlmEngineType,
)
from docling.utils.accelerator_utils import decide_device

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig, VlmModelSpec

_log = logging.getLogger(__name__)


class AutoInlineVlmEngine(BaseVlmEngine):
    """Auto-selecting engine that picks the best local implementation.

    Selection logic:
    1. On macOS with Apple Silicon (MPS available) -> MLX
    2. On Linux/Windows with CUDA and prefer_vllm=True -> vLLM
    3. Otherwise -> Transformers

    This engine delegates to the selected engine after initialization.
    """

    def __init__(
        self,
        options: AutoInlineVlmEngineOptions,
        accelerator_options: Optional[AcceleratorOptions] = None,
        artifacts_path=None,
        model_spec: Optional["VlmModelSpec"] = None,
    ):
        """Initialize the auto-inline engine.

        Args:
            options: Auto-inline engine options
            accelerator_options: Hardware accelerator configuration
            artifacts_path: Path to cached model artifacts
            model_spec: Model specification (for generating engine-specific configs)
        """
        super().__init__(options, model_config=None)
        self.options: AutoInlineVlmEngineOptions = options
        self.accelerator_options = accelerator_options or AcceleratorOptions()
        self.artifacts_path = artifacts_path
        self.model_spec = model_spec

        # The actual engine will be set during initialization
        self.actual_engine: Optional[BaseVlmEngine] = None
        self.selected_engine_type: Optional[VlmEngineType] = None

        # Initialize immediately if model_spec is provided
        if self.model_spec is not None:
            self.initialize()

    def _select_engine(self) -> VlmEngineType:
        """Select the best engine based on platform and hardware.

        Respects model's supported_engines if model_spec is provided.

        Returns:
            The selected engine type
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

        _log.info(f"Auto-selecting engine for system={system}, device={device}")

        # macOS with Apple Silicon -> MLX (if explicitly supported)
        if system == "Darwin" and device == "mps":
            # Check if model has explicit MLX export
            has_mlx_export = False
            if self.model_spec is not None:
                has_mlx_export = self.model_spec.has_explicit_engine_export(
                    VlmEngineType.MLX
                )

            if has_mlx_export:
                try:
                    import mlx_vlm

                    _log.info(
                        "Selected MLX engine (Apple Silicon with explicit MLX export)"
                    )
                    return VlmEngineType.MLX
                except ImportError:
                    _log.warning(
                        "MLX not available on Apple Silicon, falling back to Transformers"
                    )
            else:
                _log.info(
                    "MLX not selected: no explicit MLX export found for this model "
                    "(no different repo_id in engine_overrides or not in supported_engines). "
                    "Falling back to Transformers."
                )

        # CUDA with prefer_vllm -> vLLM (if supported)
        if device.startswith("cuda") and self.options.prefer_vllm:
            # For vLLM, check supported_engines if explicitly set
            # (vLLM typically uses the same repo_id, so we only check explicit restrictions)
            has_vllm_support = True
            if (
                self.model_spec is not None
                and self.model_spec.supported_engines is not None
            ):
                has_vllm_support = (
                    VlmEngineType.VLLM in self.model_spec.supported_engines
                )

            if has_vllm_support:
                try:
                    import vllm

                    _log.info("Selected vLLM engine (CUDA + prefer_vllm=True)")
                    return VlmEngineType.VLLM
                except ImportError:
                    _log.warning("vLLM not available, falling back to Transformers")
            else:
                _log.info(
                    "vLLM not selected: not in model's supported_engines. "
                    "Falling back to Transformers."
                )

        # Default to Transformers (should always be supported)
        _log.info("Selected Transformers engine (default)")
        return VlmEngineType.TRANSFORMERS

    def initialize(self) -> None:
        """Initialize by selecting and creating the actual engine."""
        if self._initialized:
            return

        _log.info("Initializing auto-inline VLM inference engine...")

        # Select the best engine
        self.selected_engine_type = self._select_engine()

        # Generate model_config for the selected engine
        model_config = None
        if self.model_spec is not None:
            model_config = self.model_spec.get_engine_config(self.selected_engine_type)
            _log.info(
                f"Generated config for {self.selected_engine_type.value}: "
                f"repo_id={model_config.repo_id}, extra_config={model_config.extra_config}"
            )

        # Create the actual engine
        if self.selected_engine_type == VlmEngineType.MLX:
            from docling.models.inference_engines.vlm.mlx_engine import MlxVlmEngine

            mlx_options = MlxVlmEngineOptions(
                trust_remote_code=self.options.trust_remote_code
                if hasattr(self.options, "trust_remote_code")
                else False,
            )
            self.actual_engine = MlxVlmEngine(
                options=mlx_options,
                artifacts_path=self.artifacts_path,
                model_config=model_config,
            )

        elif self.selected_engine_type == VlmEngineType.VLLM:
            from docling.models.inference_engines.vlm.vllm_engine import VllmVlmEngine

            vllm_options = VllmVlmEngineOptions()
            self.actual_engine = VllmVlmEngine(
                options=vllm_options,
                accelerator_options=self.accelerator_options,
                artifacts_path=self.artifacts_path,
                model_config=model_config,
            )

        else:  # TRANSFORMERS
            from docling.models.inference_engines.vlm.transformers_engine import (
                TransformersVlmEngine,
            )

            transformers_options = TransformersVlmEngineOptions()
            self.actual_engine = TransformersVlmEngine(
                options=transformers_options,
                accelerator_options=self.accelerator_options,
                artifacts_path=self.artifacts_path,
                model_config=model_config,
            )

        # Note: actual_engine.initialize() is called automatically in their __init__
        # if model_config is provided

        self._initialized = True
        _log.info(
            f"Auto-inline engine initialized with {self.selected_engine_type.value}"
        )

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs using the selected engine.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        assert self.actual_engine is not None, "Engine not initialized"

        # Delegate to the actual engine's batch implementation
        return self.actual_engine.predict_batch(input_batch)

    def cleanup(self) -> None:
        """Clean up the actual engine resources."""
        if self.actual_engine is not None:
            self.actual_engine.cleanup()
            self.actual_engine = None

        _log.info("Auto-inline engine cleaned up")
