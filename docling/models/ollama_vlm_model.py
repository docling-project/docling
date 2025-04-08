import base64
import io
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image
import ollama

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    OllamaVlmOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class OllamaVlmModel(BasePageModel):

    def __init__(
        self,
        enabled: bool,
        vlm_options: OllamaVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            self.client = ollama.Client(self.vlm_options.base_url)
            self.model_id = self.vlm_options.model_id
            self.client.pull(self.model_id)
            self.options = {}
            self.prompt_content = f"This is a page from a document.\n{self.vlm_options.prompt}"
            if self.vlm_options.num_ctx:
                self.options["num_ctx"] = self.vlm_options.num_ctx

    @staticmethod
    def _encode_image(image: Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="png")
        return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

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

                    # populate page_tags with predicted doc tags
                    page_tags = ""

                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    res = self.client.chat(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": self.prompt_content,
                                "images": [self._encode_image(hi_res_image)],
                            },
                        ],
                        options={
                            "temperature": 0,
                        }
                    )
                    page_tags = res.message.content

                    # inference_time = time.time() - start_time
                    # tokens_per_second = num_tokens / generation_time
                    # print("")
                    # print(f"Page Inference Time: {inference_time:.2f} seconds")
                    # print(f"Total tokens on page: {num_tokens:.2f}")
                    # print(f"Tokens/sec: {tokens_per_second:.2f}")
                    # print("")
                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                yield page
