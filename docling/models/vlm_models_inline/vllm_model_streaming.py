import asyncio
import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL.Image import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class VllmVlmModelStreaming(BaseVlmPageModel, HuggingFaceModelDownloadMixin):
    """Streaming sibling to VllmVlmModel with periodic partial callbacks.

    - Public API is unchanged (including process_images return values).
    - Internally uses vLLM AsyncLLM streaming.
    - Calls self.handle_partial(index, delta_text, state) every N tokens
    """

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        self.partial_token_interval: int = 64

        if self.enabled:
            from transformers import AutoProcessor
            from vllm.async_llm import AsyncLLM
            from vllm.sampling_params import SamplingParams

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for VLM: {self.device}")

            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            llm_kwargs: Dict[str, Any] = {
                "model": str(artifacts_path),
                "limit_mm_per_prompt": {"image": 1},
                "trust_remote_code": vlm_options.trust_remote_code,
                "model_impl": "transformers",
                "gpu_memory_utilization": 0.3,
            }
            if self.device == "cpu":
                llm_kwargs["device"] = "cpu"
            if vlm_options.quantized and vlm_options.load_in_8bit:
                llm_kwargs["quantization"] = "bitsandbytes"

            # Async engine + background loop
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever, daemon=True
            )
            self._loop_thread.start()

            async def _init():
                return AsyncLLM(**llm_kwargs)

            self._engine = asyncio.run_coroutine_threadsafe(
                _init(), self._loop
            ).result()

            # Processor / tokenizer (used for formatting & token fallback)
            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
            )

            # Sampling params
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop=vlm_options.stop_strings if vlm_options.stop_strings else None,
                **vlm_options.extra_generation_config,
            )

    # -------------------------- Public API (unchanged) --------------------------

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        page_list = list(page_batch)
        if not page_list:
            return

        valid_pages: list[Page] = []
        invalid_pages: list[Page] = []
        for page in page_list:
            assert page._backend is not None
            (invalid_pages if not page._backend.is_valid() else valid_pages).append(
                page
            )

        if valid_pages:
            with TimeRecorder(conv_res, "vlm"):
                images, user_prompts, pages_with_images = [], [], []

                for page in valid_pages:
                    assert page.size is not None
                    hi_res_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )
                    if hi_res_image is None:
                        continue

                    images.append(hi_res_image)
                    user_prompt = (
                        self.vlm_options.prompt(page.parsed_page)
                        if callable(self.vlm_options.prompt)
                        else self.vlm_options.prompt
                    )
                    user_prompts.append(user_prompt)
                    pages_with_images.append(page)

                if images:
                    predictions = list(self.process_images(images, user_prompts))
                    for page, prediction in zip(pages_with_images, predictions):
                        page.predictions.vlm_response = prediction

        for page in invalid_pages:
            yield page
        for page in valid_pages:
            yield page

    def process_images(
        self,
        image_batch: Iterable[Union[Image, np.ndarray]],
        prompt: Union[str, list[str]],
    ) -> Iterable[VlmPrediction]:
        """Same contract as the non-streaming version; internally streams tokens."""
        if not self.enabled:
            return

        pil_images: list[Image] = []
        for img in image_batch:
            if isinstance(img, np.ndarray):
                from PIL import Image as PILImage

                if img.ndim == 3 and img.shape[2] in [3, 4]:
                    pil_img = PILImage.fromarray(img.astype(np.uint8))
                elif img.ndim == 2:
                    pil_img = PILImage.fromarray(img.astype(np.uint8), mode="L")
                else:
                    raise ValueError(f"Unsupported numpy array shape: {img.shape}")
            else:
                pil_img = img

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        if len(pil_images) == 0:
            return

        if isinstance(prompt, str):
            user_prompts = [prompt] * len(pil_images)
        elif isinstance(prompt, list):
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of images ({len(pil_images)})"
                )
            user_prompts = prompt
        else:
            raise ValueError(f"prompt must be str or list[str], got {type(prompt)}")

        # Use base class' formulate_prompt
        prompts: list[str] = [self.formulate_prompt(p) for p in user_prompts]

        llm_inputs = [
            {"prompt": p, "multi_modal_data": {"image": im}}
            for p, im in zip(prompts, pil_images)
        ]

        start_time = time.time()
        futures = [
            asyncio.run_coroutine_threadsafe(
                self._stream_and_accumulate(idx, inp), self._loop
            )
            for idx, inp in enumerate(llm_inputs)
        ]
        results = [f.result() for f in futures]
        generation_time = time.time() - start_time

        if results:
            first_tokens = results[0]["token_count"]
            if first_tokens is not None:
                _log.debug(
                    f"Generated {first_tokens} tokens in time {generation_time:.2f} seconds."
                )

        for res in results:
            decoded_text = self.vlm_options.decode_response(res["text"])
            yield VlmPrediction(text=decoded_text, generation_time=generation_time)

    # -------------------------- Hooks & internals --------------------------

    def handle_partial(self, index: int, delta_text: str, state: dict) -> None:
        """Override to inspect partials. Called roughly every N tokens."""
        # default: no-op

    async def _stream_and_accumulate(self, index: int, inp: dict) -> dict:
        """Submit one streaming request, invoking handle_partial every N tokens.

        Returns:
          {"text": str, "token_count": Optional[int]}
        """
        from vllm.sampling_params import SamplingParams

        req_id = await self._engine.add_request(
            prompt=inp["prompt"],
            multi_modal_data=inp.get("multi_modal_data"),
            use_streaming=True,
            sampling_params=self.sampling_params
            if isinstance(self.sampling_params, SamplingParams)
            else SamplingParams(**dict(self.sampling_params)),
        )

        full_text_parts: list[str] = []
        pending_parts: list[str] = []  # text since last callback
        token_count: Optional[int] = None
        last_callback_tokens: int = 0
        state: dict[str, Any] = {}

        # Helper: best-effort token count using metrics; fallback to tokenizer
        def current_token_count_estimate(cumulative_text: str, metrics_obj) -> int:
            # Prefer vLLM-provided count if present
            try:
                if metrics_obj and hasattr(metrics_obj, "num_generated_tokens"):
                    val = metrics_obj.num_generated_tokens
                    if isinstance(val, int):
                        return val
            except Exception:
                pass
            # Fallback: approximate via tokenizer (no special tokens)
            try:
                tokenizer = getattr(self.processor, "tokenizer", None)
                if tokenizer is not None:
                    return len(
                        tokenizer.encode(cumulative_text, add_special_tokens=False)
                    )
            except Exception:
                pass
            # Last-ditch: approximate by characters (very rough)
            return len(cumulative_text)

        async for out in self._engine.generate(req_id):
            # Get cumulative/current text
            try:
                current_text = out.outputs[0].text or ""
            except Exception:
                current_text = ""

            # Derive delta
            delta = getattr(out.outputs[0], "text_delta", None)
            if delta is None:
                prev = "".join(full_text_parts)
                if len(current_text) > len(prev) and current_text.startswith(prev):
                    delta = current_text[len(prev) :]
                else:
                    delta = current_text

            if delta:
                full_text_parts.append(delta)
                pending_parts.append(delta)

            # Update counts
            try:
                token_count = current_token_count_estimate(
                    "".join(full_text_parts), getattr(out, "metrics", None)
                )
            except Exception:
                token_count = None

            # Fire callback every N tokens
            if token_count is not None:
                while (
                    token_count - last_callback_tokens >= self.partial_token_interval
                    and pending_parts
                ):
                    # deliver text accumulated since last callback
                    delta_text = "".join(pending_parts)
                    self.handle_partial(index=index, delta_text=delta_text, state=state)
                    pending_parts.clear()
                    last_callback_tokens += self.partial_token_interval

        # Flush any remainder to the callback at the end (optional but useful)
        if pending_parts:
            self.handle_partial(
                index=index, delta_text="".join(pending_parts), state=state
            )
            pending_parts.clear()

        # Try to get precise final token count if still unknown
        if token_count is None:
            try:
                final_outputs = await self._engine.get_request_output(req_id)
                if final_outputs and final_outputs[0].outputs:
                    token_ids = getattr(final_outputs[0].outputs[0], "token_ids", None)
                    if token_ids is not None:
                        token_count = len(token_ids)
            except Exception:
                pass

        return {"text": "".join(full_text_parts), "token_count": token_count}
