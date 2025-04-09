from typing import Iterable

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OpenAiVlmOptions
from docling.models.base_model import BasePageModel
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import openai_image_request


class OpenAiVlmModel(BasePageModel):

    def __init__(
        self,
        enabled: bool,
        vlm_options: OpenAiVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            self.url = "/".join(
                [self.vlm_options.base_url.rstrip("/"), "chat/completions"]
            )
            self.apikey = self.vlm_options.apikey
            self.model_id = self.vlm_options.model_id
            self.timeout = self.vlm_options.timeout
            self.prompt_content = (
                f"This is a page from a document.\n{self.vlm_options.prompt}"
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
                    assert hi_res_image is not None
                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    page_tags = openai_image_request(
                        image=hi_res_image,
                        prompt=self.prompt_content,
                        url=self.url,
                        apikey=self.apikey,
                        timeout=self.timeout,
                        model=self.model_id,
                        temperature=0,
                    )

                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                yield page
