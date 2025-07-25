import importlib.metadata
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.accelerator_options import (
    AcceleratorOptions,
)
from docling.datamodel.base_models import Cluster, Page, VlmPrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.models.base_model import BaseLayoutModel, BasePageModel, BaseVlmModel
from docling.models.layout_model import LayoutModel
from docling.models.utils.hf_model_download import (
    HuggingFaceModelDownloadMixin,
)
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


class TwoStageVlmModel(BasePageModel, HuggingFaceModelDownloadMixin):
    def __init__(
        self,
        *,
        layout_model: BaseLayoutModel,
        vlm_model: BaseVlmModel,
    ):
        self.layout_model = layout_model
        self.vlm_model = vlm_model

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "two-staged-vlm"):
                    assert page.size is not None

                    page_image = page.get_image(
                        scale=self.vlm_model.scale, max_size=self.vlm_model.max_size
                    )
                    assert page_image is not None

                    pred_clusters = self.layout_model.predict_on_page_image(
                        page_image=page_image
                    )

                    page, processed_clusters, processed_cells = (
                        self.layout_model.postprocess_on_page_image(
                            page=page, clusters=pred_clusters
                        )
                    )

                    user_prompt = self.vlm_model.get_user_prompt(page=page)
                    prompt = self.formulate_prompt(
                        user_prompt=user_prompt,
                        clusters=processed_clusters,
                        image_width=page_image.width,
                        image_height=page_image.height,
                    )

                    start_time = time.time()
                    generated_text, generated_tokens = (
                        self.vlm_model.predict_on_page_image(
                            page_image=page_image, prompt=prompt
                        )
                    )
                    print("generated-text: \n", generated_text, "\n")
                    page.predictions.vlm_response = VlmPrediction(
                        text=generated_text,
                        generation_time=time.time() - start_time,
                        generated_tokens=generated_tokens,
                    )
                    exit(-1)

                yield page

    def formulate_prompt(
        self,
        *,
        user_prompt: str,
        clusters: list[Cluster],
        image_width: int,
        image_height: int,
        vlm_width: int = 512,
        vlm_height: int = 512,
    ) -> str:
        """Formulate a prompt for the VLM."""

        known_clusters = ["here is a list of unsorted text-blocks:", "<doctags>"]
        for cluster in clusters:
            print(" => ", cluster)

            loc_l = f"<loc_{int(vlm_width * cluster.bbox.l / image_width)}>"
            loc_b = f"<loc_{int(vlm_height * cluster.bbox.b / image_height)}>"
            loc_r = f"<loc_{int(vlm_width * cluster.bbox.r / image_width)}>"
            loc_t = f"<loc_{int(vlm_height * cluster.bbox.t / image_height)}>"

            known_clusters.append(
                f"<{cluster.label}>{loc_l}{loc_b}{loc_r}{loc_t}</{cluster.label}>"
            )

        known_clusters.append("</doctags>")

        user_prompt = "\n".join(known_clusters) + f"\n\n{user_prompt}"
        print("user-prompt: ", user_prompt, "\n")

        return user_prompt
