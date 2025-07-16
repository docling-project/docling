import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

from PIL import Image

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Page, VlmPrediction, VlmPredictionToken
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import InlineVlmOptions
from docling.models.base_model import BasePageModel, BaseVlmModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class HuggingFaceMlxModel(BaseVlmModel, HuggingFaceModelDownloadMixin):
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
        self.scale = self.vlm_options.scale

        self.max_size = 512
        if isinstance(self.vlm_options.max_size, int):
            self.max_size = self.vlm_options.max_size

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

            ## Load the model
            self.vlm_model, self.processor = load(artifacts_path)
            self.config = load_config(artifacts_path)

    def get_user_prompt(self, page: Optional[Page]) -> str:
        if callable(self.vlm_options.prompt) and page is not None:
            return self.vlm_options.prompt(page.parsed_page)
        else:
            user_prompt = self.vlm_options.prompt
            prompt = self.apply_chat_template(
                self.processor, self.config, user_prompt, num_images=1
            )
            return prompt

    def predict_on_page_image(
        self, *, page_image: Image.Image, prompt: str, output_tokens: bool = False
    ) -> tuple[str, Optional[list[VlmPredictionToken]]]:
        tokens = []
        output = ""
        for token in self.stream_generate(
            self.vlm_model,
            self.processor,
            prompt,
            [page_image],
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
            elif len(token.logprobs.shape) == 2 and token.logprobs.shape[0] == 1:
                tokens.append(
                    VlmPredictionToken(
                        text=token.text,
                        token=token.token,
                        logprob=token.logprobs[0, token.token],
                    )
                )
            else:
                _log.warning(f"incompatible shape for logprobs: {token.logprobs.shape}")

            output += token.text
            if "</doctag>" in token.text:
                break

        return output, tokens

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

                    page_image = page.get_image(
                        scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                    )
                    """
                    if page_image is not None:
                        im_width, im_height = page_image.size
                    """
                    assert page_image is not None

                    # populate page_tags with predicted doc tags
                    page_tags = ""

                    if page_image:
                        if page_image.mode != "RGB":
                            page_image = page_image.convert("RGB")

                    """
                    if callable(self.vlm_options.prompt):
                        user_prompt = self.vlm_options.prompt(page.parsed_page)
                    else:
                        user_prompt = self.vlm_options.prompt
                    prompt = self.apply_chat_template(
                        self.processor, self.config, user_prompt, num_images=1
                    )
                    """
                    prompt = self.get_user_prompt(page)

                    # Call model to generate:
                    start_time = time.time()
                    """
                    tokens: list[VlmPredictionToken] = []

                    output = ""
                    for token in self.stream_generate(
                        self.vlm_model,
                        self.processor,
                        prompt,
                        [page_image],
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
                    """
                    output, tokens = self.predict_on_page_image(
                        page_image=page_image, prompt=prompt, output_tokens=True
                    )

                    generation_time = time.time() - start_time
                    page_tags = output

                    """
                    _log.debug(
                        f"{generation_time:.2f} seconds for {len(tokens)} tokens ({len(tokens) / generation_time} tokens/sec)."
                    )
                    """

                    page.predictions.vlm_response = VlmPrediction(
                        text=page_tags,
                        generation_time=generation_time,
                        generated_tokens=tokens,
                    )

                yield page
