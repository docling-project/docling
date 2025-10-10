import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type, Union

from PIL import Image
from transformers import AutoModelForImageTextToText

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import (
    PictureDescriptionBaseOptions,
    PictureDescriptionVlmOptions,
)
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class PictureDescriptionVlmModel(
    PictureDescriptionBaseModel, HuggingFaceModelDownloadMixin
):
    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionVlmOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionVlmOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionVlmOptions

        if self.enabled:
            if artifacts_path is None:
                artifacts_path = self.download_models(repo_id=self.options.repo_id)
            else:
                artifacts_path = Path(artifacts_path) / self.options.repo_cache_folder

            self.device = decide_device(accelerator_options.device)

            try:
                import torch
                from transformers import AutoModelForVision2Seq, AutoProcessor
            except ImportError:
                raise ImportError(
                    "transformers >=4.46 is not installed. Please install Docling with the required extras `pip install docling[vlm]`."
                )

            # Initialize processor and model
            with _model_init_lock:
                self.processor = AutoProcessor.from_pretrained(artifacts_path)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    dtype=torch.bfloat16,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "sdpa"
                    ),
                )
                self.model = torch.compile(self.model)  # type: ignore

            self.provenance = f"{self.options.repo_id}"
            
# Constants for VLM Quality Check 
MAX_ATTEMPTS = 3
MIN_WORD_COUNT = 5
INSUFFICIENT_WORDS = {"in", "this", "the", "a", "an", "the image", "this image"}            


def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        from transformers import GenerationConfig

        # Use the base prompt defined in the model options
        base_prompt = self.options.prompt

        # TODO: Implement actual batch generation instead of single image iteration
        for image in images:
            
            # --- START HACKATHON FIX: VLM RETRY LOOP ---
            current_prompt = base_prompt
            
            for attempt in range(MAX_ATTEMPTS):
                # 1. --- PREPARE INPUTS (Original Logic) ---
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": current_prompt},
                        ],
                    },
                ]
                
                # Apply chat template and tokenize
                prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
                inputs = inputs.to(self.device)

                # 2. --- GENERATE OUTPUTS (Original Logic) ---
                generated_ids = self.model.generate(
                    **inputs,
                    generation_config=GenerationConfig(**self.options.generation_config),
                )
                generated_texts = self.processor.batch_decode(
                    generated_ids[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                response = generated_texts[0].strip()

                # 3. --- CHECK QUALITY AND DECIDE RETRY ---
                response_words = response.lower().split()
                word_count = len(response_words)
                
                # Predicates defined using external constants
                is_generic_fail = response_words and response_words[0] in INSUFFICIENT_WORDS
                is_length_fail = word_count < MIN_WORD_COUNT
                
                if not is_generic_fail and not is_length_fail:
                    # Success! Yield the good description and exit the inner loop
                    yield response
                    break 
                
                # If quality check fails and we have attempts remaining:
                if attempt < MAX_ATTEMPTS - 1:
                    print(f"DEBUG: VLM failed on attempt {attempt+1} ('{response[:20]}...'). Retrying.")
                    
                    # Construct a stronger re-prompt based on failure reason
                    failure_reason = ""
                    if is_generic_fail:
                        failure_reason += "Your previous answer was too generic (like 'This' or 'In'). "
                    if is_length_fail:
                        failure_reason += f"The description was too short (under {MIN_WORD_COUNT} words). "
                        
                    # Update the prompt for the next attempt
                    current_prompt = (
                        f"{base_prompt} {failure_reason} Provide a DETAILED, SPECIFIC, multi-sentence caption."
                    )
                else:
                    # Last attempt failed (attempt == MAX_ATTEMPTS - 1), 
                    # yield the best (last) response found and exit the inner loop
                    yield response
                    break 
            # --- END HACKATHON FIX: VLM RETRY LOOP ---