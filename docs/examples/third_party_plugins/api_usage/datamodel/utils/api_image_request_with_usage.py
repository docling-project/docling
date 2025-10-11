import base64
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import OpenAiApiResponse, OpenAiResponseUsage
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def api_image_request_with_usage(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[Dict[str, str]] = None,
    **params,
) -> Tuple[str, Optional[OpenAiResponseUsage]]:
    """Send an image+prompt to an OpenAI-compatible API and return (text, usage).

    If no usage data is available, the second tuple element will be None.
    """
    img_io = BytesIO()
    image.save(img_io, "PNG")
    image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    payload = {
        "messages": messages,
        **params,
    }

    headers = headers or {}

    r = requests.post(
        str(url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        _log.error(f"Error calling the API. Response was {r.text}")
    r.raise_for_status()

    api_resp = OpenAiApiResponse.model_validate_json(r.text)
    generated_text = api_resp.choices[0].message.content.strip()

    usage = api_resp.usage if hasattr(api_resp, "usage") else None

    return generated_text, usage
