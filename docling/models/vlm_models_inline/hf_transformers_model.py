import importlib.metadata
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL.Image import Image
from transformers import StoppingCriteria, StoppingCriteriaList, StopStringCriteria

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction, VlmStopReason
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.base_model import BaseVlmPageModel
from docling.models.utils.generation_utils import (
    GenerationStopper,
    HFStoppingCriteriaWrapper,
)
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceTransformersVlmModel(BaseVlmPageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled

        self.vlm_options = vlm_options

        if self.enabled:
            import torch
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoModelForVision2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
                GenerationConfig,
            )

            transformers_version = importlib.metadata.version("transformers")
            if (
                self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct"
                and transformers_version >= "4.52.0"
            ):
                raise NotImplementedError(
                    f"Phi 4 only works with transformers<4.52.0 but you have {transformers_version=}. Please downgrage running pip install -U 'transformers<4.52.0'."
                )

            self.device = decide_device(
                accelerator_options.device,
                supported_devices=vlm_options.supported_devices,
            )
            _log.debug(f"Available device for VLM: {self.device}")

            self.use_cache = vlm_options.use_kv_cache
            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = self.download_models(
                    self.vlm_options.repo_id, revision=self.vlm_options.revision
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_quantization_config: Optional[BitsAndBytesConfig] = None
            if vlm_options.quantized:
                self.param_quantization_config = BitsAndBytesConfig(
                    load_in_8bit=vlm_options.load_in_8bit,
                    llm_int8_threshold=vlm_options.llm_int8_threshold,
                )

            model_cls: Any = AutoModel
            if (
                self.vlm_options.transformers_model_type
                == TransformersModelType.AUTOMODEL_CAUSALLM
            ):
                model_cls = AutoModelForCausalLM
            elif (
                self.vlm_options.transformers_model_type
                == TransformersModelType.AUTOMODEL_VISION2SEQ
            ):
                model_cls = AutoModelForVision2Seq
            elif (
                self.vlm_options.transformers_model_type
                == TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
            ):
                model_cls = AutoModelForImageTextToText

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
                revision=vlm_options.revision,
            )
            self.processor.tokenizer.padding_side = "left"

            self.vlm_model = model_cls.from_pretrained(
                artifacts_path,
                device_map=self.device,
                dtype=self.vlm_options.torch_dtype,
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and accelerator_options.cuda_use_flash_attention2
                    else "sdpa"
                ),
                trust_remote_code=vlm_options.trust_remote_code,
                revision=vlm_options.revision,
            )
            if sys.version_info < (3, 14):
                self.vlm_model = torch.compile(self.vlm_model)  # type: ignore
            else:
                self.vlm_model.eval()

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(
                artifacts_path, revision=vlm_options.revision
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
            with TimeRecorder(conv_res, "vlm"):
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
                    with TimeRecorder(conv_res, "vlm_inference"):
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
        """
        Batched inference for Hugging Face Image-Text-to-Text VLMs (e.g., SmolDocling / SmolVLM).
        - Lets the processor handle all padding & batching for text+images.
        - Trims generated sequences per row using attention_mask (no pad-id fallbacks).
        - Keeps your formulate_prompt() exactly as-is.
        """
        import numpy as np
        import torch
        from PIL import Image as PILImage

        # -- Normalize images to RGB PIL
        pil_images: list[Image] = []
        for img in image_batch:
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] in (3, 4):
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

        if not pil_images:
            return

        # -- Normalize prompts (1 per image)
        if isinstance(prompt, str):
            user_prompts = [prompt] * len(pil_images)
        else:
            if len(prompt) != len(pil_images):
                raise ValueError(
                    f"Number of prompts ({len(prompt)}) must match number of images ({len(pil_images)})"
                )
            user_prompts = prompt

        # Use your prompt formatter verbatim
        if self.vlm_options.transformers_prompt_style == TransformersPromptStyle.NONE:
            inputs = self.processor(
                pil_images,
                return_tensors="pt",
                padding=True,  # pad across batch for both text and vision
                **self.vlm_options.extra_processor_kwargs,
            )
        else:
            prompts: list[str] = [self.formulate_prompt(p) for p in user_prompts]

            # -- Processor performs BOTH text+image preprocessing + batch padding (recommended)
            inputs = self.processor(
                text=prompts,
                images=pil_images,
                return_tensors="pt",
                padding=True,  # pad across batch for both text and vision
                **self.vlm_options.extra_processor_kwargs,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # -- Optional stopping criteria
        stopping_criteria_list: StoppingCriteriaList = StoppingCriteriaList()

        # Add string-based stopping criteria
        if self.vlm_options.stop_strings:
            stopping_criteria_list.append(
                StopStringCriteria(
                    stop_strings=self.vlm_options.stop_strings,
                    tokenizer=self.processor.tokenizer,
                )
            )

        # Add custom stopping criteria
        if self.vlm_options.custom_stopping_criteria:
            for criteria in self.vlm_options.custom_stopping_criteria:
                # If it's a class (not an instance), determine the type and handle accordingly
                if isinstance(criteria, type):
                    # Check if it's a GenerationStopper class
                    if issubclass(criteria, GenerationStopper):
                        # Instantiate GenerationStopper and wrap it
                        stopper_instance = criteria()
                        wrapped_criteria = HFStoppingCriteriaWrapper(
                            self.processor.tokenizer, stopper_instance
                        )
                        stopping_criteria_list.append(wrapped_criteria)
                    elif issubclass(criteria, StoppingCriteria):
                        # It's a StoppingCriteria class, instantiate with tokenizer
                        criteria_instance = criteria(self.processor.tokenizer)
                        stopping_criteria_list.append(criteria_instance)
                elif isinstance(criteria, GenerationStopper):
                    # Wrap GenerationStopper instances in HFStoppingCriteriaWrapper
                    wrapped_criteria = HFStoppingCriteriaWrapper(
                        self.processor.tokenizer, criteria
                    )
                    stopping_criteria_list.append(wrapped_criteria)
                else:
                    # If it's already an instance of StoppingCriteria, use it directly
                    stopping_criteria_list.append(criteria)

        stopping_criteria = (
            StoppingCriteriaList(stopping_criteria_list)
            if stopping_criteria_list
            else None
        )

        # -- Filter out decoder-specific keys from extra_generation_config
        decoder_keys = {
            "skip_special_tokens",
            "clean_up_tokenization_spaces",
            "spaces_between_special_tokens",
        }
        generation_config = {
            k: v
            for k, v in self.vlm_options.extra_generation_config.items()
            if k not in decoder_keys
        }
        decoder_config = {
            k: v
            for k, v in self.vlm_options.extra_generation_config.items()
            if k in decoder_keys
        }

        # -- Generate (Image-Text-to-Text class expects these inputs from processor)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": self.use_cache,
            "generation_config": self.generation_config,
            **generation_config,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.vlm_model.generate(**gen_kwargs)
        generation_time = time.time() - start_time

        input_len = inputs["input_ids"].shape[1]  # common right-aligned prompt length
        trimmed_sequences = generated_ids[:, input_len:]  # only newly generated tokens

        # -- Decode with the processor/tokenizer (skip specials, keep DocTags as text)
        decode_fn = getattr(self.processor, "batch_decode", None)
        if decode_fn is None and getattr(self.processor, "tokenizer", None) is not None:
            decode_fn = self.processor.tokenizer.batch_decode
        if decode_fn is None:
            raise RuntimeError(
                "Neither processor.batch_decode nor tokenizer.batch_decode is available."
            )

        decoded_texts: list[str] = decode_fn(
            trimmed_sequences,
            **decoder_config,
        )

        # -- Clip off pad tokens from decoded texts
        pad_token = self.processor.tokenizer.pad_token
        if pad_token:
            decoded_texts = [text.rstrip(pad_token) for text in decoded_texts]

        # -- Optional logging
        num_tokens = None
        if generated_ids.shape[0] > 0:
            num_tokens = int(generated_ids[0].shape[0])
            _log.debug(
                f"Generated {num_tokens} tokens in {generation_time:.2f}s "
                f"for batch size {generated_ids.shape[0]}."
            )

        for i, text in enumerate(decoded_texts):
            input_prompt = (
                prompts[i] if self.vlm_options.track_input_prompt and prompts else None
            )
            # Apply decode_response to the output text
            decoded_text = self.vlm_options.decode_response(text)
            yield VlmPrediction(
                text=decoded_text,
                generation_time=generation_time,
                num_tokens=num_tokens,
                stop_reason=VlmStopReason.UNSPECIFIED,
                input_prompt=input_prompt,
            )
