"""MLX-based VLM inference engine for Apple Silicon."""

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from PIL.Image import Image

from docling.datamodel.vlm_engine_options import MlxVlmEngineOptions
from docling.models.runtimes._utils import (
    extract_generation_stoppers,
    preprocess_image_batch,
)
from docling.models.runtimes.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)

# Global lock for MLX model calls - MLX models are not thread-safe
# All MLX models share this lock to prevent concurrent MLX operations
_MLX_GLOBAL_LOCK = threading.Lock()


class MlxVlmEngine(BaseVlmEngine, HuggingFaceModelDownloadMixin):
    """MLX engine for VLM inference on Apple Silicon.

    This engine uses the mlx-vlm library to run vision-language models
    efficiently on Apple Silicon (M1/M2/M3) using the Metal Performance Shaders.

    Note: MLX models are not thread-safe and use a global lock.
    """

    def __init__(
        self,
        options: MlxVlmEngineOptions,
        artifacts_path: Optional[Path] = None,
        model_config: Optional["EngineModelConfig"] = None,
    ):
        """Initialize the MLX engine.

        Args:
            options: MLX-specific runtime options
            artifacts_path: Path to cached model artifacts
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: MlxVlmEngineOptions = options
        self.artifacts_path = artifacts_path

        # These will be set during initialization
        # MLX types are complex and external, using Any with type: ignore
        self.vlm_model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self.apply_chat_template: Any = None
        self.stream_generate: Any = None

        # Initialize immediately if model_config is provided
        if self.model_config is not None:
            self.initialize()

    def initialize(self) -> None:
        """Initialize the MLX model and processor."""
        if self._initialized:
            return

        _log.info("Initializing MLX VLM inference engine...")

        try:
            from mlx_vlm import load, stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError:
            raise ImportError(
                "mlx-vlm is not installed. Please install it via `pip install mlx-vlm` "
                "to use MLX VLM models on Apple Silicon."
            )

        self.apply_chat_template = apply_chat_template  # type: ignore[assignment]
        self.stream_generate = stream_generate  # type: ignore[assignment]

        # Load model if model_config is provided
        if self.model_config is not None and self.model_config.repo_id is not None:
            repo_id = self.model_config.repo_id
            revision = self.model_config.revision or "main"

            _log.info(f"Loading MLX model {repo_id} (revision: {revision})")
            self._load_model_for_repo(repo_id, revision=revision)

        self._initialized = True
        _log.info("MLX runtime initialized")

    def _load_model_for_repo(self, repo_id: str, revision: str = "main") -> None:
        """Load model and processor for a specific repository.

        Args:
            repo_id: HuggingFace repository ID
            revision: Model revision
        """
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        # Download or locate model artifacts
        repo_cache_folder = repo_id.replace("/", "--")
        if self.artifacts_path is None:
            artifacts_path = self.download_models(repo_id, revision=revision)
        elif (self.artifacts_path / repo_cache_folder).exists():
            artifacts_path = self.artifacts_path / repo_cache_folder
        else:
            artifacts_path = self.artifacts_path

        # Load the model
        self.vlm_model, self.processor = load(artifacts_path)
        self.config = load_config(artifacts_path)

        _log.info(f"Loaded MLX model {repo_id} (revision: {revision})")

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
        """Run inference on a batch of inputs.

        Note: MLX models are not thread-safe and use a global lock, so batch
        processing is done sequentially. This method is provided for API
        consistency but does not provide performance benefits over sequential
        processing.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        if not input_batch:
            return []

        # Model should already be loaded via initialize()
        if self.vlm_model is None or self.processor is None or self.config is None:
            raise RuntimeError(
                "Model not loaded. Ensure EngineModelConfig was provided during initialization."
            )

        _log.debug(
            f"MLX runtime processing batch of {len(input_batch)} images sequentially "
            "(MLX does not support batched inference)"
        )

        outputs: List[VlmEngineOutput] = []

        # MLX models are not thread-safe - use global lock to serialize access
        with _MLX_GLOBAL_LOCK:
            _log.debug("MLX model: Acquired global lock for thread safety")

            for input_data in input_batch:
                # Preprocess image
                images = preprocess_image_batch([input_data.image])
                image = images[0]

                # Format prompt using MLX's chat template
                formatted_prompt = self.apply_chat_template(
                    self.processor, self.config, input_data.prompt, num_images=1
                )

                # Extract custom stopping criteria
                custom_stoppers = extract_generation_stoppers(
                    input_data.extra_generation_config
                )

                # Stream generate with stop strings and custom stopping criteria support
                start_time = time.time()
                _log.debug("Starting MLX generation...")

                output_text = ""
                stop_reason = "unspecified"

                # Use stream_generate for proper stop string handling
                for token in self.stream_generate(
                    self.vlm_model,
                    self.processor,
                    formatted_prompt,
                    [image],  # MLX stream_generate expects list of images
                    max_tokens=input_data.max_new_tokens,
                    verbose=False,
                    temp=input_data.temperature,
                ):
                    output_text += token.text

                    # Check for configured stop strings
                    if input_data.stop_strings:
                        if any(
                            stop_str in output_text
                            for stop_str in input_data.stop_strings
                        ):
                            _log.debug("Stopping generation due to stop string match")
                            stop_reason = "stop_string"
                            break

                    # Check for custom stopping criteria
                    if custom_stoppers:
                        for stopper in custom_stoppers:
                            # Determine the text window to check based on lookback_tokens
                            lookback_tokens = stopper.lookback_tokens()
                            text_to_check = (
                                output_text[-lookback_tokens:]
                                if len(output_text) > lookback_tokens
                                else output_text
                            )

                            try:
                                if stopper.should_stop(text_to_check):
                                    _log.info(
                                        f"Stopping generation due to GenerationStopper: {type(stopper).__name__}"
                                    )
                                    stop_reason = "custom_criteria"
                                    break
                            except Exception as e:
                                _log.warning(
                                    f"Error in GenerationStopper.should_stop: {e}"
                                )
                                continue
                        else:
                            # for-else: only executed if inner loop didn't break
                            continue
                        # Break outer loop if any stopper triggered
                        break

                generation_time = time.time() - start_time

                _log.debug(
                    f"MLX generation completed in {generation_time:.2f}s, "
                    f"stop_reason: {stop_reason}"
                )

                # Create output
                outputs.append(
                    VlmEngineOutput(
                        text=output_text,
                        stop_reason=stop_reason,
                        metadata={
                            "generation_time": generation_time,
                            "model": self.model_config.repo_id
                            if self.model_config
                            else "unknown",
                        },
                    )
                )

            _log.debug("MLX model: Released global lock")

        return outputs

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        _log.info("MLX runtime cleaned up")
