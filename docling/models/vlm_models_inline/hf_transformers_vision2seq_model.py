import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BasePageModel
from docling.models.hf_vlm_model import HuggingFaceVlmModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceVlmModel_AutoModelForVision2Seq(BasePageModel):
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
            from transformers import (  # type: ignore
                AutoModelForVision2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
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

            # PARAMETERS:
            if artifacts_path is None:
                # artifacts_path = self.download_models(self.vlm_options.repo_id)
                artifacts_path = HuggingFaceVlmModel.download_models(
                    self.vlm_options.repo_id
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_quantization_config: Optional[BitsAndBytesConfig] = None
            if vlm_options.quantized:
                self.param_quantization_config = BitsAndBytesConfig(
                    load_in_8bit=vlm_options.load_in_8bit,
                    llm_int8_threshold=vlm_options.llm_int8_threshold,
                )

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=vlm_options.trust_remote_code,
            )
            self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                artifacts_path,
                device_map=self.device,
                # torch_dtype=torch.bfloat16,
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and accelerator_options.cuda_use_flash_attention2
                    else "eager"
                ),
                trust_remote_code=vlm_options.trust_remote_code,
            )

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "vlm"):
                    assert page.size is not None

                    hi_res_image = page.get_image(scale=self.vlm_options.scale)

                    if hi_res_image is not None:
                        im_width, im_height = hi_res_image.size

                    # populate page_tags with predicted doc tags
                    page_tags = ""

                    """
                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")
                    """

                    # Define prompt structure
                    prompt = self.formulate_prompt()

                    inputs = self.processor(
                        text=prompt, images=[hi_res_image], return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    start_time = time.time()
                    # Call model to generate:
                    generated_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,
                        temperature=self.temperature,
                    )

                    generation_time = time.time() - start_time
                    generated_texts = self.processor.batch_decode(
                        generated_ids[:, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=False,
                    )[0]

                    num_tokens = len(generated_ids[0])
                    page_tags = generated_texts

                    _log.debug(
                        f"Generated {num_tokens} tokens in time {generation_time:.2f} seconds."
                    )
                    page.predictions.vlm_response = VlmPrediction(
                        text=page_tags,
                        generation_time=generation_time,
                    )

                yield page

    def formulate_prompt(self) -> str:
        """Formulate a prompt for the VLM."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is a page from a document.",
                    },
                    {"type": "image"},
                    {"type": "text", "text": self.vlm_options.prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False
        )
        return prompt
