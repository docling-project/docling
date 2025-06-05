import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction, VlmPredictionToken
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceMlxModel(BasePageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        vlm_options: InlineVlmOptions,
    ):
        self.enabled = enabled

        self.vlm_options = vlm_options
        self.max_tokens = vlm_options.max_new_tokens
        self.temperature = vlm_options.temperature

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
                artifacts_path = self.download_models(
                    self.vlm_options.repo_id,
                )
            elif (artifacts_path / repo_cache_folder).exists():
                artifacts_path = artifacts_path / repo_cache_folder

            self.param_question = vlm_options.prompt

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, f"vlm-mlx-{self.vlm_options.repo_id}"):
                    assert page.size is not None

                    hi_res_image = page.get_image(scale=self.vlm_options.scale)
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
                    _log.debug("start generating ...")

                    # Call model to generate:
                    tokens: list[VlmPredictionToken] = []

                    output = ""
                    for token in self.stream_generate(
                        self.vlm_model,
                        self.processor,
                        prompt,
                        [hi_res_image],
                        max_tokens=self.max_tokens,
                        verbose=False,
                        temp=self.temperature,
                    ):
                        if len(token.logprobs.shape) == 1:
                            tokens.append(
                                VlmPredictionToken(
                                    text=token.text,
                                    token=token.token,
                                    logprob=token.logprobs[token.token],
                                )
                            )
                        elif (
                            len(token.logprobs.shape) == 2
                            and token.logprobs.shape[0] == 1
                        ):
                            tokens.append(
                                VlmPredictionToken(
                                    text=token.text,
                                    token=token.token,
                                    logprob=token.logprobs[0, token.token],
                                )
                            )
                        else:
                            _log.warning(
                                f"incompatible shape for logprobs: {token.logprobs.shape}"
                            )

                        output += token.text
                        if "</doctag>" in token.text:
                            break

                    generation_time = time.time() - start_time
                    page_tags = output

                    _log.debug(
                        f"{generation_time:.2f} seconds for {len(tokens)} tokens ({len(tokens) / generation_time} tokens/sec)."
                    )
                    page.predictions.vlm_response = VlmPrediction(
                        text=page_tags,
                        generation_time=generation_time,
                        generated_tokens=tokens,
                    )

                yield page
