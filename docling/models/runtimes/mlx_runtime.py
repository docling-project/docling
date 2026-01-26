"""MLX-based VLM runtime for Apple Silicon."""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL.Image import Image

from docling.datamodel.vlm_runtime_options import MlxVlmRuntimeOptions
from docling.models.runtimes.base import (
    BaseVlmRuntime,
    VlmRuntimeInput,
    VlmRuntimeOutput,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin

_log = logging.getLogger(__name__)

# Global lock for MLX model calls - MLX models are not thread-safe
# All MLX models share this lock to prevent concurrent MLX operations
_MLX_GLOBAL_LOCK = threading.Lock()


class MlxVlmRuntime(BaseVlmRuntime, HuggingFaceModelDownloadMixin):
    """MLX runtime for VLM inference on Apple Silicon.

    This runtime uses the mlx-vlm library to run vision-language models
    efficiently on Apple Silicon (M1/M2/M3) using the Metal Performance Shaders.

    Note: MLX models are not thread-safe and use a global lock.
    """

    def __init__(
        self,
        options: MlxVlmRuntimeOptions,
        artifacts_path: Optional[Path] = None,
    ):
        """Initialize the MLX runtime.

        Args:
            options: MLX-specific runtime options
            artifacts_path: Path to cached model artifacts
        """
        super().__init__(options)
        self.options: MlxVlmRuntimeOptions = options
        self.artifacts_path = artifacts_path

        # These will be set during initialization
        # MLX types are complex and external, using Any with type: ignore
        self.vlm_model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self.apply_chat_template: Any = None
        self.stream_generate: Any = None

    def initialize(self) -> None:
        """Initialize the MLX model and processor."""
        if self._initialized:
            return

        _log.info("Initializing MLX VLM runtime...")

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

    def predict(self, input_data: VlmRuntimeInput) -> VlmRuntimeOutput:
        """Run inference on a single image.

        Args:
            input_data: Input containing image, prompt, and configuration

        Returns:
            Generated text and metadata
        """
        if not self._initialized:
            self.initialize()

        # Load model if not already loaded
        if self.vlm_model is None or self.processor is None:
            revision = input_data.extra_generation_config.get("revision", "main")
            self._load_model_for_repo(input_data.repo_id, revision=revision)

        # Prepare image
        image = input_data.image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Format prompt using MLX's chat template
        formatted_prompt = self.apply_chat_template(  # type: ignore[misc]
            self.processor,
            self.config,
            input_data.prompt,
            num_images=1,
        )

        # Check for custom stopping criteria
        custom_stoppers = []
        custom_criteria = input_data.extra_generation_config.get(
            "custom_stopping_criteria", []
        )
        for criteria in custom_criteria:
            if isinstance(criteria, GenerationStopper):
                custom_stoppers.append(criteria)
            elif isinstance(criteria, type) and issubclass(criteria, GenerationStopper):
                custom_stoppers.append(criteria())

        # Use global lock for thread safety
        with _MLX_GLOBAL_LOCK:
            start_time = time.time()

            if custom_stoppers:
                # Streaming generation with early abort support
                generated_text = ""
                num_tokens = 0
                stop_reason = "unspecified"

                for chunk in self.stream_generate(  # type: ignore[misc]
                    self.vlm_model,
                    self.processor,
                    image,
                    formatted_prompt,
                    max_tokens=input_data.max_new_tokens,
                    temp=input_data.temperature,
                    verbose=False,
                ):
                    generated_text = chunk
                    num_tokens += 1

                    # Check stopping criteria
                    for stopper in custom_stoppers:
                        if stopper.should_stop(generated_text):
                            stop_reason = "custom_criteria"
                            break

                    if stop_reason != "unspecified":
                        break
            else:
                # Non-streaming generation
                from mlx_vlm import generate

                generated_text = generate(
                    self.vlm_model,
                    self.processor,
                    image,
                    formatted_prompt,
                    max_tokens=input_data.max_new_tokens,
                    temp=input_data.temperature,
                    verbose=False,
                )
                num_tokens = len(generated_text.split())  # Rough estimate
                stop_reason = "unspecified"

            generation_time = time.time() - start_time

        # Clean up the generated text
        if input_data.stop_strings:
            for stop_string in input_data.stop_strings:
                if stop_string in generated_text:
                    generated_text = generated_text.split(stop_string)[0]
                    stop_reason = "stop_string"
                    break

        return VlmRuntimeOutput(
            text=generated_text,
            stop_reason=stop_reason,
            metadata={
                "generation_time": generation_time,
                "num_tokens": num_tokens,
            },
        )

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        _log.info("MLX runtime cleaned up")
