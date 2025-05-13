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

from transformers import AutoProcessor, LlavaForConditionalGeneration

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
        print(self.vlm_options)

        if self.enabled:
            import torch
            from transformers import (  # type: ignore
                LlavaForConditionalGeneration,
                AutoProcessor,
            )

            self.device = decide_device(accelerator_options.device)
            self.device = "cpu"  # FIXME

            torch.set_num_threads(12)  # Adjust the number as needed
            
            _log.debug(f"Available device for VLM: {self.device}")
            repo_cache_folder = vlm_options.repo_id.replace("/", "--")

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            model_path = artifacts_path
            print(f"model: {model_path}")

            self.max_new_tokens = 64 # FIXME
            
            self.processor = AutoProcessor.from_pretrained(
                artifacts_path,
                trust_remote_code=self.trust_remote_code,
            )
            self.vlm_model = LlavaForConditionalGeneration.from_pretrained(artifacts_path).to(self.device)

    
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

                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    images = [
                        hi_res_image
                    ]
                    prompt = "<s>[INST]Describe the images.\n[IMG][/INST]"
                    
                    inputs = self.processor(text=prompt, images=images, return_tensors="pt", use_fast=False).to(self.device) #.to("cuda")
                    generate_ids = self.vlm_model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True  # Enables KV caching which can improve performance
                    )
                    response = self.processor.batch_decode(generate_ids,
                                                           skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)[0]
                    print(f"response: {response}")
                    """
                    _log.debug(
                        f"Generated {num_tokens} tokens in time {generation_time:.2f} seconds."
                    )
                    """
                    # inference_time = time.time() - start_time
                    # tokens_per_second = num_tokens / generation_time
                    # print("")
                    # print(f"Page Inference Time: {inference_time:.2f} seconds")
                    # print(f"Total tokens on page: {num_tokens:.2f}")
                    # print(f"Tokens/sec: {tokens_per_second:.2f}")
                    # print("")
                    page.predictions.vlm_response = VlmPrediction(text=response)

                yield page
