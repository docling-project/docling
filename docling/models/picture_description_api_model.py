import base64
import io
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Type, Union

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict

from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PictureDescriptionApiOptions,
    PictureDescriptionBaseOptions,
)
from docling.exceptions import OperationNotAllowed
from docling.models.picture_description_base_model import PictureDescriptionBaseModel

_log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ApiResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    id: str
    model: Optional[str] = None  # returned by openai
    choices: List[ResponseChoice]
    created: int
    usage: ResponseUsage


class PictureDescriptionApiModel(PictureDescriptionBaseModel):
    # elements_batch_size = 4

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        return PictureDescriptionApiOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionApiOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionApiOptions

        if self.enabled:
            if not enable_remote_services:
                raise OperationNotAllowed(
                    "Connections to remote services is only allowed when set explicitly. "
                    "pipeline_options.enable_remote_services=True."
                )

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        # Note: technically we could make a batch request here,
        # but not all APIs will allow for it. For example, vllm won't allow more than 1.
        for image in images:
            img_io = io.BytesIO()
            image.save(img_io, "PNG")
            image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.options.prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ]

            payload = {
                "messages": messages,
                **self.options.params,
            }

            r = requests.post(
                str(self.options.url),
                headers=self.options.headers,
                json=payload,
                timeout=self.options.timeout,
            )
            if not r.ok:
                _log.error(f"Error calling the API. Reponse was {r.text}")
            r.raise_for_status()

            api_resp = ApiResponse.model_validate_json(r.text)
            generated_text = api_resp.choices[0].message.content.strip()
            yield generated_text
