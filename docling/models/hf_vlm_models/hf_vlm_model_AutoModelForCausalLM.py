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
            self.device = "cpu"  # FIXME

            _log.debug(f"Available device for VLM: {self.device}")
            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            # PARAMETERS:
            if artifacts_path is None:
                # artifacts_path = self.download_models(self.vlm_options.repo_id)
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
            if not self.param_quantized:
                self.vlm_model = AutoModelForCausalLM.from_pretrained(
                    artifacts_path,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "eager"
                    ),
                    trust_remote_code=self.trust_remote_code,
                ).to(self.device)

            else:
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
                ).to(self.device)

            model_path = artifacts_path
            print(f"model: {model_path}")

            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(model_path)

    """
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        if not progress:
            disable_progress_bars()
        download_path = snapshot_download(
            repo_id=repo_id,
            force_download=force,
            local_dir=local_dir,
            # revision="v0.0.1",
        )

        return Path(download_path)
    """

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

                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    # Define prompt structure
                    user_prompt = "<|user|>"
                    assistant_prompt = "<|assistant|>"
                    prompt_suffix = "<|end|>"

                    # Part 1: Image Processing
                    prompt = f"{user_prompt}<|image_1|>Convert this image into MarkDown and only return the bare MarkDown!{prompt_suffix}{assistant_prompt}"

                    inputs = self.processor(
                        text=prompt, images=hi_res_image, return_tensors="pt"
                    ).to(self.device)

                    # Generate response
                    start_time = time.time()
                    generate_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=128,
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

                    # inference_time = time.time() - start_time
                    # tokens_per_second = num_tokens / generation_time
                    # print("")
                    # print(f"Page Inference Time: {inference_time:.2f} seconds")
                    # print(f"Total tokens on page: {num_tokens:.2f}")
                    # print(f"Tokens/sec: {tokens_per_second:.2f}")
                    # print("")
                    page.predictions.vlm_response = VlmPrediction(text=response)

                yield page
