from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

from docling.datamodel.base_models import Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions
from docling.exceptions import OperationNotAllowed
from docling.models.base_model import BasePageModel
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.api_image_request import (
    api_image_request,
    api_image_request_streaming,
)
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
            self.params = {
                **self.vlm_options.params,
                "temperature": self.vlm_options.temperature,
            }

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        def _vlm_request(page):
            assert page._backend is not None
            if not page._backend.is_valid():
                return page

            with TimeRecorder(conv_res, "vlm"):
                assert page.size is not None

                hi_res_image = page.get_image(
                    scale=self.vlm_options.scale, max_size=self.vlm_options.max_size
                )
                assert hi_res_image is not None
                if hi_res_image and hi_res_image.mode != "RGB":
                    hi_res_image = hi_res_image.convert("RGB")

                prompt = self.vlm_options.build_prompt(page.parsed_page)

                if self.vlm_options.custom_stopping_criteria:
                    # Instantiate any GenerationStopper classes before passing to streaming
                    instantiated_stoppers = []
                    for criteria in self.vlm_options.custom_stopping_criteria:
                        if isinstance(criteria, GenerationStopper):
                            instantiated_stoppers.append(criteria)
                        elif isinstance(criteria, type) and issubclass(
                            criteria, GenerationStopper
                        ):
                            instantiated_stoppers.append(criteria())
                        # Skip non-GenerationStopper criteria (should have been caught in validation)

                    # Streaming path with early abort support
                    page_tags, num_tokens = api_image_request_streaming(
                        image=hi_res_image,
                        prompt=prompt,
                        url=self.vlm_options.url,
                        timeout=self.timeout,
                        headers=self.vlm_options.headers,
                        generation_stoppers=instantiated_stoppers,
                        **self.params,
                    )
                else:
                    # Non-streaming fallback (existing behavior)
                    page_tags, num_tokens = api_image_request(
                        image=hi_res_image,
                        prompt=prompt,
                        url=self.vlm_options.url,
                        timeout=self.timeout,
                        headers=self.vlm_options.headers,
                        **self.params,
                    )

                page_tags = self.vlm_options.decode_response(page_tags)
                page.predictions.vlm_response = VlmPrediction(
                    text=page_tags, num_tokens=num_tokens
                )
            return page

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            yield from executor.map(_vlm_request, page_batch)
