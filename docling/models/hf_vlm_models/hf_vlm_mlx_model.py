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
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceMlxModel(BasePageModel):
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
            try:
                from mlx_vlm import generate, load  # type: ignore
                from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
                from mlx_vlm.utils import load_config, stream_generate  # type: ignore
            except ImportError:
                raise ImportError(
                    "mlx-vlm is not installed. Please install it via `pip install mlx-vlm` to use MLX VLM models."
                )

            repo_cache_folder = vlm_options.repo_id.replace("/", "--")
            self.apply_chat_template = apply_chat_template
            self.stream_generate = stream_generate

            # PARAMETERS:
            if artifacts_path is None:
                artifacts_path = self.download_models(self.vlm_options.repo_id)
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_question = vlm_options.prompt  # "Perform Layout Analysis."

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

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

                    prompt = self.apply_chat_template(
                        self.processor, self.config, self.param_question, num_images=1
                    )

                    start_time = time.time()
                    # Call model to generate:
                    output = ""
                    for token in self.stream_generate(
                        self.vlm_model,
                        self.processor,
                        prompt,
                        [hi_res_image],
                        max_tokens=4096,
                        verbose=False,
                    ):
                        output += token.text
                        if "</doctag>" in token.text:
                            break

                    generation_time = time.time() - start_time
                    page_tags = output

                    _log.debug(f"Generation time {generation_time:.2f} seconds.")

                    # inference_time = time.time() - start_time
                    # tokens_per_second = num_tokens / generation_time
                    # print("")
                    # print(f"Page Inference Time: {inference_time:.2f} seconds")
                    # print(f"Total tokens on page: {num_tokens:.2f}")
                    # print(f"Tokens/sec: {tokens_per_second:.2f}")
                    # print("")
                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                yield page
