# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Build and send OpenAI-compatible NuExtract requests."""

import base64
from io import BytesIO
from typing import Any

from pydantic import AnyUrl

from docling.datamodel.base_models import ApiImageRequestResult
from docling.datamodel.extraction import (
    ContentItem,
    ImageContentItem,
    TextContentItem,
)
from docling.utils.api_image_request import _post_openai_chat_completion


def _content_item_to_openai(item: ContentItem) -> dict[str, Any]:
    if isinstance(item, TextContentItem):
        return {"type": "text", "text": item.text}
    if isinstance(item, ImageContentItem):
        with item.image.convert("RGBA") as image, BytesIO() as buffer:
            image.save(buffer, "PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    raise ValueError(f"Unsupported content item: {type(item)}")


def api_nuextract_request(
    content_items: list[ContentItem],
    *,
    template: str,
    url: AnyUrl,
    timeout: float = 120,
    headers: dict[str, str] | None = None,
    usage_response_key: str | None = "usage",
    token_extract_key: str | None = None,
    **params: Any,
) -> ApiImageRequestResult:
    """POST one NuExtract request: document item(s) in content, template out-of-band."""
    return _post_openai_chat_completion(
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": [_content_item_to_openai(i) for i in content_items],
                }
            ],
            "chat_template_kwargs": {"template": template},
            **params,
        },
        url=url,
        timeout=timeout,
        headers=headers,
        usage_response_key=usage_response_key,
        token_extract_key=token_extract_key,
    )
