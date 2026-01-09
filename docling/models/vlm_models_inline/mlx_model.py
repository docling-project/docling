import logging
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL.Image import Image
from transformers import StoppingCriteria

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import (
    Page,
    VlmPrediction,
    VlmPredictionToken,
    VlmStopReason,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.generation_utils import GenerationStopper
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)

# Global lock for MLX model calls - MLX models are not thread-safe
# All MLX models share this lock to prevent concurrent MLX operations
_MLX_GLOBAL_LOCK = threading.Lock()


class HuggingFaceMlxModel(BaseVlmPageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled

        self.vlm_options = vlm_options
        self.max_tokens = vlm_options.max_new_tokens
        self.temperature = vlm_options.temperature

        if self.enabled:
            try:
                from mlx_vlm import generate, load, stream_generate  # type: ignore
                from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
                from mlx_vlm.utils import load_config  # type: ignore
            except ImportError:
                raise ImportError(
                    "mlx-vlm is not installed. Please install it via `pip install mlx-vlm` to use MLX VLM models."
                )

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            self.apply_chat_template = apply_chat_template
            self.stream_generate = stream_generate

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(
                    self.vlm_options.repo_id,
                    revision=self.vlm_options.revision,
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

            # Validate custom stopping criteria - MLX doesn't support HF StoppingCriteria
            if self.vlm_options.custom_stopping_criteria:
                for criteria in self.vlm_options.custom_stopping_criteria:
                    if isinstance(criteria, StoppingCriteria):
                        raise ValueError(
                            f"MLX models do not support HuggingFace StoppingCriteria instances. "
                            f"Found {type(criteria).__name__}. Use GenerationStopper instead."
                        )
                    elif isinstance(criteria, type) and issubclass(
                        criteria, StoppingCriteria
                    ):
                        raise ValueError(
                            f"MLX models do not support HuggingFace StoppingCriteria classes. "
                            f"Found {criteria.__name__}. Use GenerationStopper instead."
                        )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        page_list = list(page_batch)
        if not page_list:
            return

        valid_pages = []
        invalid_pages = []

        for page in page_list:
            assert page._backend is not None
            if not page._backend.is_valid():
                invalid_pages.append(page)
            else:
                valid_pages.append(page)

        # Process valid pages in batch
        if valid_pages:
            with TimeRecorder(conv_res, f"vlm-mlx-{self.vlm_options.repo_id}"):
                # Prepare images and prompts for batch processing
                images = []
                user_prompts = []
                pages_with_images = []

                for page in valid_pages:
                    assert page.size is not None
                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )

                    # Only process pages with valid images
                    if hi_res_image is not None:
                        images.append(hi_res_image)

                        # Define prompt structure
                        user_prompt = self._build_prompt_safe(page)

                        user_prompts.append(user_prompt)
                        pages_with_images.append(page)

                # Use process_images for the actual inference
                if images:  # Only if we have valid images
                    predictions = list(self.process_images(images, user_prompts))

                    # Attach results to pages
                    for page, prediction in zip(pages_with_images, predictions):
                        page.predictions.vlm_response = prediction

        # Yield all pages (valid and invalid)
        for page in invalid_pages:
            yield page
        for page in valid_pages:
            yield page

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Process raw images without page metadata.

        Args:
            image_batch: Iterable of PIL Images or numpy arrays
            prompt: Either:
                - str: Single prompt used for all images
                - list[str]: List of prompts (one per image, must match image count)

        Raises:
            ValueError: If prompt list length doesn't match image count.
        """
        # Convert image batch to list for length validation
        image_list = list(image_batch)

        if len(image_list) == 0:
            return

        # Handle prompt parameter
        if isinstance(prompt, str):
            # Single prompt for all images
            user_prompts = [prompt] * len(image_list)
        elif isinstance(prompt, list):
            # List of prompts (one per image)
            if len(prompt) != len(image_list):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of images ({len(image_list)})"
                )
            user_prompts = prompt
        else:
            raise ValueError(f"prompt must be str or list[str], got {type(prompt)}")

        # MLX models are not thread-safe - use global lock to serialize access
        with _MLX_GLOBAL_LOCK:
            _log.debug("MLX model: Acquired global lock for thread safety")
            for image, user_prompt in zip(image_list, user_prompts):
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    if image.ndim == 3 and image.shape[2] in [3, 4]:
                        # RGB or RGBA array
                        from PIL import Image as PILImage

                        image = PILImage.fromarray(image.astype(np.uint8))
                    elif image.ndim == 2:
                        # Grayscale array
                        from PIL import Image as PILImage

                        image = PILImage.fromarray(image.astype(np.uint8), mode="L")
                    else:
                        raise ValueError(
                            f"Unsupported numpy array shape: {image.shape}"
                        )

                # Ensure image is in RGB mode (handles RGBA, L, etc.)
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Use the MLX chat template approach like in the __call__ method
                formatted_prompt = self.apply_chat_template(
                    self.processor, self.config, user_prompt, num_images=1
                )

                # Stream generate with stop strings and custom stopping criteria support
                start_time = time.time()
                _log.debug("start generating ...")

                tokens: list[VlmPredictionToken] = []
                output = ""

                # Use stream_generate for proper stop string handling
                for token in self.stream_generate(
                    self.vlm_model,
                    self.processor,
                    formatted_prompt,
                    [image],  # MLX stream_generate expects list of images
                    max_tokens=self.max_tokens,
                    verbose=False,
                    temp=self.temperature,
                ):
                    # Collect token information
                    if len(token.logprobs.shape) == 1:
                        tokens.append(
                            VlmPredictionToken(
                                text=token.text,
                                token=token.token,
                                logprob=token.logprobs[token.token],
                            )
                        )
                    elif (
                        len(token.logprobs.shape) == 2 and token.logprobs.shape[0] == 1
                    ):
                        tokens.append(
                            VlmPredictionToken(
                                text=token.text,
                                token=token.token,
                                logprob=token.logprobs[0, token.token],
                            )
                        )
                    else:
                        _log.warning(
                            f"incompatible shape for logprobs: {token.logprobs.shape}"
                        )

                    output += token.text

                    # Check for any configured stop strings
                    if self.vlm_options.stop_strings:
                        if any(
                            stop_str in output
                            for stop_str in self.vlm_options.stop_strings
                        ):
                            _log.debug("Stopping generation due to stop string match")
                            break

                    # Check for custom stopping criteria (GenerationStopper instances)
                    if self.vlm_options.custom_stopping_criteria:
                        for criteria in self.vlm_options.custom_stopping_criteria:
                            # Handle both instances and classes of GenerationStopper
                            if isinstance(criteria, GenerationStopper):
                                stopper = criteria
                            elif isinstance(criteria, type) and issubclass(
                                criteria, GenerationStopper
                            ):
                                stopper = criteria()

                            # Determine the text window to check based on lookback_tokens
                            lookback_tokens = stopper.lookback_tokens()
                            # Check only the last N characters worth of text
                            # This is a simplified approach - in practice, you might want to
                            # decode the last N tokens from the token list for more accuracy
                            text_to_check = (
                                output[-lookback_tokens:]
                                if len(output) > lookback_tokens
                                else output
                            )

                            try:
                                if stopper.should_stop(text_to_check):
                                    _log.info(
                                        f"Stopping generation due to GenerationStopper: {type(stopper).__name__}"
                                    )
                                    break
                            except Exception as e:
                                _log.warning(
                                    f"Error in GenerationStopper.should_stop: {e}"
                                )
                                continue
                        else:  # note: for-else idiom
                            continue  # Only executed if the inner loop didn't break
                        break  # Break the outer loop if any stopper triggered

                generation_time = time.time() - start_time

                _log.debug(
                    f"{generation_time:.2f} seconds for {len(tokens)} tokens ({len(tokens) / generation_time:.1f} tokens/sec)."
                )

                # Apply decode_response to the output before yielding
                decoded_output = self.vlm_options.decode_response(output)
                input_prompt = (
                    formatted_prompt if self.vlm_options.track_input_prompt else None
                )
                yield VlmPrediction(
                    text=decoded_output,
                    generation_time=generation_time,
                    generated_tokens=tokens,
                    num_tokens=len(tokens),
                    stop_reason=VlmStopReason.UNSPECIFIED,
                    input_prompt=input_prompt,
                )
            _log.debug("MLX model: Released global lock")
