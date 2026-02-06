"""API-based VLM inference engine for remote services."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional

from PIL.Image import Image

from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.exceptions import OperationNotAllowed
from docling.models.inference_engines.vlm._utils import (
    extract_generation_stoppers,
    preprocess_image_batch,
)
from docling.models.inference_engines.vlm.base import (
    BaseVlmEngine,
    VlmEngineInput,
    VlmEngineOutput,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.api_image_request import (
    api_image_request,
    api_image_request_streaming,
)

if TYPE_CHECKING:
    from docling.datamodel.stage_model_specs import EngineModelConfig

_log = logging.getLogger(__name__)


class ApiVlmEngine(BaseVlmEngine):
    """API engine for VLM inference via remote services.

    This runtime supports OpenAI-compatible API endpoints including:
    - Generic OpenAI-compatible APIs
    - Ollama
    - LM Studio
    - OpenAI
    """

    def __init__(
        self,
        enable_remote_services: bool,
        options: ApiVlmEngineOptions,
        model_config: Optional["EngineModelConfig"] = None,
    ):
        """Initialize the API engine.

        Args:
            options: API-specific runtime options
            model_config: Model configuration (repo_id, revision, extra_config)
        """
        super().__init__(options, model_config=model_config)
        self.enable_remote_services = enable_remote_services
        self.options: ApiVlmEngineOptions = options

        if not self.enable_remote_services:
            raise OperationNotAllowed(
                "Connections to remote services is only allowed when set explicitly. "
                "pipeline_options.enable_remote_services=True."
            )

        # Merge model_config extra_config (which contains API params from model spec)
        # with runtime options params. Runtime options take precedence.
        if model_config and "api_params" in model_config.extra_config:
            # Model spec provides API params (e.g., model name)
            model_api_params = model_config.extra_config["api_params"]

            # Only use model spec params if user hasn't provided any params
            # This prevents conflicts when users provide custom params (e.g., model_id for watsonx)
            if not self.options.params:
                self.merged_params = model_api_params.copy()
            else:
                # User provided params - use them as-is (don't merge with model spec)
                self.merged_params = self.options.params.copy()
        else:
            self.merged_params = self.options.params.copy()

    def initialize(self) -> None:
        """Initialize the API engine.

        For API runtimes, initialization is minimal - just validate options.
        """
        if self._initialized:
            return

        _log.info(
            f"Initializing API VLM inference engine (endpoint: {self.options.url})"
        )

        # Validate that we have a URL
        if not self.options.url:
            raise ValueError("API runtime requires a URL")

        self._initialized = True
        _log.info("API runtime initialized")

    def predict_batch(self, input_batch: List[VlmEngineInput]) -> List[VlmEngineOutput]:
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

        def _process_single_input(input_data: VlmEngineInput) -> VlmEngineOutput:
            """Process a single input via API."""
            # Prepare image using shared utility
            images = preprocess_image_batch([input_data.image])
            image = images[0]

            # Prepare API parameters (use merged params which include model spec params)
            api_params = {
                **self.merged_params,
                "temperature": input_data.temperature,
            }

            # Add max_tokens if specified
            if input_data.max_new_tokens:
                api_params["max_tokens"] = input_data.max_new_tokens

            # Add stop strings if specified
            if input_data.stop_strings:
                api_params["stop"] = input_data.stop_strings

            # Extract custom stopping criteria using shared utility
            custom_stoppers = extract_generation_stoppers(
                input_data.extra_generation_config
            )

            request_start_time = time.time()
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

            generation_time = time.time() - request_start_time

            return VlmEngineOutput(
                text=generated_text,
                stop_reason=stop_reason,
                metadata={
                    "generation_time": generation_time,
                    "num_tokens": num_tokens,
                },
            )

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
                executor.submit(_process_single_input, input_data)
                for input_data in input_batch
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
