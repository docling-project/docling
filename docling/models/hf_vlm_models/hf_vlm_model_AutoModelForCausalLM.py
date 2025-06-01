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


class HuggingFaceVlmModel_AutoModelForCausalLM(BasePageModel):
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
            import torch
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoProcessor,
                BitsAndBytesConfig,
                GenerationConfig,
            )

            self.device = decide_device(accelerator_options.device)
            self.device = HuggingFaceVlmModel.map_device_to_cpu_if_mlx(self.device)
            _log.debug(f"Available device for VLM: {self.device}")

            self.use_cache = vlm_options.use_kv_cache
            self.max_new_tokens = vlm_options.max_new_tokens
            self.temperature = vlm_options.temperature

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            if artifacts_path is None:
                artifacts_path = HuggingFaceVlmModel.download_models(
                    self.vlm_options.repo_id
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_question = vlm_options.prompt  # "Perform Layout Analysis."
            self.param_quantization_config = BitsAndBytesConfig(
                load_in_8bit=vlm_options.load_in_8bit,  # True,
                llm_int8_threshold=vlm_options.llm_int8_threshold,  # 6.0
            )
            self.param_quantized = vlm_options.quantized  # False

            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=self.trust_remote_code,
            )
            if self.param_quantized:
                print("using quantized")
                self.vlm_model = AutoModelForCausalLM.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    torch_dtype="auto",
                    quantization_config=self.param_quantization_config,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "eager"
                    ),
                    trust_remote_code=self.trust_remote_code,
                )  # .to(self.device)
            else:
                print("using original")
                self.vlm_model = AutoModelForCausalLM.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    torch_dtype="auto",  # torch.bfloat16,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "eager"
                    ),
                    trust_remote_code=self.trust_remote_code,
                )  # .to(self.device)

            model_path = artifacts_path

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(model_path)

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

                    hi_res_image = page.get_image(scale=2)  # self.vlm_options.scale)

                    if hi_res_image is not None:
                        im_width, im_height = hi_res_image.size

                    # Define prompt structure
                    prompt = self.formulate_prompt()
                    print(f"prompt: '{prompt}', size: {im_width}, {im_height}")

                    inputs = self.processor(
                        text=prompt, images=hi_res_image, return_tensors="pt"
                    ).to(self.device)

                    # Generate response
                    start_time = time.time()
                    generate_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=self.use_cache,  # Enables KV caching which can improve performance
                        temperature=self.temperature,
                        generation_config=self.generation_config,
                        num_logits_to_keep=1,
                    )
                    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

                    num_tokens = len(generate_ids[0])
                    generation_time = time.time() - start_time

                    response = self.processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    _log.debug(
                        f"Generated {num_tokens} tokens in time {generation_time:.2f} seconds."
                    )
                    page.predictions.vlm_response = VlmPrediction(
                        text=response, generation_time=generation_time
                    )

                yield page

    def formulate_prompt(self) -> str:
        """Formulate a prompt for the VLM."""
        if self.vlm_options.repo_id == "microsoft/Phi-4-multimodal-instruct":
            # more info here: https://huggingface.co/microsoft/Phi-4-multimodal-instruct#loading-the-model-locally

            user_prompt = "<|user|>"
            assistant_prompt = "<|assistant|>"
            prompt_suffix = "<|end|>"

            prompt = f"{user_prompt}<|image_1|>{self.vlm_options.prompt}{prompt_suffix}{assistant_prompt}"
            _log.debug(f"prompt for {self.vlm_options.repo_id}: {prompt}")

            return prompt
        else:
            raise ValueError(f"No prompt template for {self.vlm_options.repo_id}")

        return ""
