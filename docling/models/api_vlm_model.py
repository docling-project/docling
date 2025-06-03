from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BasePageModel
from docling.utils.api_image_request import api_image_request
from docling.utils.profiling import TimeRecorder


class ApiVlmModel(BasePageModel):
    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        vlm_options: ApiVlmOptions,
    ):
        self.enabled = enabled
        self.vlm_options = vlm_options
        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True, or using the CLI "
                    "--enable-remote-services."
                )

            self.timeout = self.vlm_options.timeout
            self.concurrency = self.vlm_options.concurrency
            self.prompt_content = (
                f"This is a page from a document.\n{self.vlm_options.prompt}"
            )
            self.params = {
                **self.vlm_options.params,
                "temperature": 0,
            }

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        def _vlm_request(page):
            assert page._backend is not None
            if not page._backend.is_valid():
                return page
            else:
                with TimeRecorder(conv_res, "vlm"):
                    assert page.size is not None

                    hi_res_image = page.get_image(scale=self.vlm_options.scale)
                    assert hi_res_image is not None
                    if hi_res_image:
                        if hi_res_image.mode != "RGB":
                            hi_res_image = hi_res_image.convert("RGB")

                    page_tags = api_image_request(
                        image=hi_res_image,
                        prompt=self.prompt_content,
                        url=self.vlm_options.url,
                        timeout=self.timeout,
                        headers=self.vlm_options.headers,
                        **self.params,
                    )

                    page.predictions.vlm_response = VlmPrediction(text=page_tags)

                return page

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_vlm_request, page_batch)
