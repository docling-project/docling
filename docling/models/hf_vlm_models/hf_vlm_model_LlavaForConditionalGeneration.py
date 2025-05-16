import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    HuggingFaceVlmOptions,
)
from docling.models.base_model import BasePageModel
from docling.models.hf_vlm_model import HuggingFaceVlmModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceVlmModel_LlavaForConditionalGeneration(BasePageModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: HuggingFaceVlmOptions,
    ):
        self.enabled = enabled

        self.trust_remote_code = True

        self.vlm_options = vlm_options

        if self.enabled:
            from transformers import (  # type: ignore
                AutoProcessor,
                LlavaForConditionalGeneration,
            )

            self.device = decide_device(accelerator_options.device)
            self.device = HuggingFaceVlmMode.map_device_to_cpu_if_mlx(self.device)

            self.use_cache = vlm_options.use_kv_cache
            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature
            
            _log.debug(f"Available device for VLM: {self.device}")
            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = HuggingFaceVlmModel.download_models(
                    self.vlm_options.repo_id
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=self.trust_remote_code,
            )
            self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                artifacts_path,
                device_map=self.device,
                # torch_dtype="auto",
                # quantization_config=self.param_quantization_config,
                _attn_implementation=(
                    "flash_attention_2"
                    if self.device.startswith("cuda")
                    and accelerator_options.cuda_use_flash_attention2
                    else "eager"
                ),
            ).to(self.device)

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

                    hi_res_image = page.get_image(scale=2.0)  # 144dpi
                    # hi_res_image = page.get_image(scale=1.0)  # 72dpi

                    if hi_res_image is not None:
                        im_width, im_height = hi_res_image.size

                    """
                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")
                    """
                    
                    images = [hi_res_image]

                    # Define prompt structure
                    prompt = self.formulate_prompt()

                    inputs = self.processor(
                        text=prompt, images=images, return_tensors="pt"
                    ).to(self.device)

                    # Generate response
                    start_time = time.time()
                    generate_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,  # Enables KV caching which can improve performance
                        temperature=self.temperature,
                    )

                    #num_tokens = len(generate_ids[0])
                    generation_time = time.time() - start_time

                    response = self.processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    page.predictions.vlm_response = VlmPrediction(
                        text=response,
                        #generated_tokens=num_tokens,
                        generation_time=generation_time,
                    )

                yield page

    def formulate_prompt(self) -> str:
        """Formulate a prompt for the VLM."""
        if self.vlm_options.repo_id == "mistral-community/pixtral-12b":
            chat = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": self.vlm_options.prompt},
                        {"type": "image"},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(chat)
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt
        else:
            raise ValueError(f"No prompt template for {self.vlm_options.repo_id}")

        return ""
