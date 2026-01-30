"""API-based VLM runtime for remote services."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional

from PIL.Image import Image

from docling.datamodel.vlm_runtime_options import ApiVlmRuntimeOptions
from docling.models.runtimes.base import (
    BaseVlmRuntime,
    VlmRuntimeInput,
    VlmRuntimeOutput,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.api_image_request import (
    api_image_request,
    api_image_request_streaming,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import RuntimeModelConfig

_log = logging.getLogger(__name__)


class ApiVlmRuntime(BaseVlmRuntime):
    """API runtime for VLM inference via remote services.

    This runtime supports OpenAI-compatible API endpoints including:
    - Generic OpenAI-compatible APIs
    - Ollama
    - LM Studio
    - OpenAI
    """

    def __init__(
        self,
        options: ApiVlmRuntimeOptions,
        model_config: Optional["RuntimeModelConfig"] = None,
    ):
        """Initialize the API runtime.

        Args:
            options: API-specific runtime options
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.options: ApiVlmRuntimeOptions = options

    def initialize(self) -> None:
        """Initialize the API runtime.

        For API runtimes, initialization is minimal - just validate options.
        """
        if self._initialized:
            return

        _log.info(f"Initializing API VLM runtime (endpoint: {self.options.url})")

        # Validate that we have a URL
        if not self.options.url:
            raise ValueError("API runtime requires a URL")

        self._initialized = True
        _log.info("API runtime initialized")

    def predict(self, input_data: VlmRuntimeInput) -> VlmRuntimeOutput:
        """Run inference via API.

        Args:
            input_data: Input containing image, prompt, and configuration

        Returns:
            Generated text and metadata
        """
        if not self._initialized:
            self.initialize()

        # Prepare image
        image = input_data.image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Prepare API parameters
        api_params = {
            **self.options.params,
            "temperature": input_data.temperature,
        }

        # Add max_tokens if specified
        if input_data.max_new_tokens:
            api_params["max_tokens"] = input_data.max_new_tokens

        # Add stop strings if specified
        if input_data.stop_strings:
            api_params["stop"] = input_data.stop_strings

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

        start_time = time.time()
        stop_reason = "unspecified"

        if custom_stoppers:
            # Streaming path with early abort support
            generated_text, num_tokens = api_image_request_streaming(
                url=self.options.url,  # type: ignore[arg-type]
                image=image,
                prompt=input_data.prompt,
                headers=self.options.headers,
                generation_stoppers=custom_stoppers,
                timeout=self.options.timeout,
                **api_params,
            )

            # Check if stopped by custom criteria
            for stopper in custom_stoppers:
                if stopper.should_stop(generated_text):
                    stop_reason = "custom_criteria"
                    break
        else:
            # Non-streaming path
            generated_text, num_tokens, api_stop_reason = api_image_request(
                url=self.options.url,  # type: ignore[arg-type]
                image=image,
                prompt=input_data.prompt,
                headers=self.options.headers,
                timeout=self.options.timeout,
                **api_params,
            )
            stop_reason = api_stop_reason

        generation_time = time.time() - start_time

        return VlmRuntimeOutput(
            text=generated_text,
            stop_reason=stop_reason,
            metadata={
                "generation_time": generation_time,
                "num_tokens": num_tokens,
            },
        )

    def predict_batch(
        self, input_batch: List[VlmRuntimeInput]
    ) -> List[VlmRuntimeOutput]:
        """Run inference on a batch of inputs using concurrent API requests.

        This method processes multiple images concurrently using a thread pool,
        which can significantly improve throughput for API-based runtimes.

        Args:
            input_batch: List of inputs to process

        Returns:
            List of outputs, one per input
        """
        if not self._initialized:
            self.initialize()

        if not input_batch:
            return []

        # Use ThreadPoolExecutor for concurrent API requests
        max_workers = min(self.options.concurrency, len(input_batch))

        _log.info(
            f"Processing batch of {len(input_batch)} images with "
            f"{max_workers} concurrent requests"
        )

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests
            futures = [
                executor.submit(self.predict, input_data) for input_data in input_batch
            ]

            # Collect results in order
            outputs = [future.result() for future in futures]

        total_time = time.time() - start_time

        _log.info(
            f"Batch processed {len(input_batch)} images in {total_time:.2f}s "
            f"({total_time / len(input_batch):.2f}s per image, "
            f"{max_workers} concurrent requests)"
        )

        return outputs

    def cleanup(self) -> None:
        """Clean up API runtime resources.

        For API runtimes, there's nothing to clean up.
        """
        _log.info("API runtime cleaned up")
