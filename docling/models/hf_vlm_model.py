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
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceVlmModel(BasePageModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: HuggingFaceVlmOptions,
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

            device = decide_device(accelerator_options.device)
            self.device = device

            _log.debug(f"Available device for HuggingFace VLM: {device}")

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_question = vlm_options.prompt  # "Perform Layout Analysis."
            self.param_quantization_config = BitsAndBytesConfig(
                load_in_8bit=vlm_options.load_in_8bit,  # True,
                llm_int8_threshold=vlm_options.llm_int8_threshold,  # 6.0
            )
            self.param_quantized = vlm_options.quantized  # False

            self.processor = AutoProcessor.from_pretrained(artifacts_path)
            if not self.param_quantized:
                self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                    artifacts_path,
                    device_map=device,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "eager"
                    ),
                )  # .to(self.device)

            else:
                self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                    artifacts_path,
                    device_map=device,
                    torch_dtype="auto",
                    quantization_config=self.param_quantization_config,
                    _attn_implementation=(
                        "flash_attention_2"
                        if self.device.startswith("cuda")
                        and accelerator_options.cuda_use_flash_attention2
                        else "eager"
                    ),
                )  # .to(self.device)

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

                    # populate page_tags with predicted doc tags
                    page_tags = ""

                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "This is a page from a document.",
                                },
                                {"type": "image"},
                                {"type": "text", "text": self.param_question},
                            ],
                        }
                    ]
                    prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=False
                    )
                    inputs = self.processor(
                        text=prompt, images=[hi_res_image], return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    start_time = time.time()
                    # Call model to generate:
                    generated_ids = self.vlm_model.generate(
                        **inputs, max_new_tokens=4096, use_cache=True
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

                    # inference_time = time.time() - start_time
                    # tokens_per_second = num_tokens / generation_time
                    # print("")
                    # print(f"Page Inference Time: {inference_time:.2f} seconds")
                    # print(f"Total tokens on page: {num_tokens:.2f}")
                    # print(f"Tokens/sec: {tokens_per_second:.2f}")
                    # print("")
                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                yield page
