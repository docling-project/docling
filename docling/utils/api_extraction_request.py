# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Encode ordered extraction content using the explicit vLLM chat contract."""

import base64
from copy import deepcopy
from io import BytesIO
from typing import Any

from pydantic import AnyUrl

from docling.datamodel.base_models import ApiImageRequestResult
from docling.datamodel.extraction import (
    ContentItem,
    ImageContentItem,
    TextContentItem,
)
from docling.datamodel.extraction_options import _reject_request_fields
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


def api_extraction_request(
    content_items: list[ContentItem],
    *,
    prompt: str,
    chat_template_kwargs: dict[str, Any],
    constraint_schema: dict[str, Any] | None = None,
    url: AnyUrl,
    timeout: float = 120,
    headers: dict[str, str] | None = None,
    usage_response_key: str | None = "usage",
    token_extract_key: str | None = None,
    **params: Any,
) -> ApiImageRequestResult:
    """POST once through the shared transport; never retry without constraints."""
    _reject_request_fields(params)
    content = [_content_item_to_openai(item) for item in content_items]
    if prompt:
        content.append({"type": "text", "text": prompt})
    payload = {**deepcopy(params), "messages": [{"role": "user", "content": content}]}
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = deepcopy(chat_template_kwargs)
    if constraint_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "schema": deepcopy(constraint_schema),
            },
        }
    return _post_openai_chat_completion(
        payload=payload,
        url=url,
        timeout=timeout,
        headers=headers,
        usage_response_key=usage_response_key,
        token_extract_key=token_extract_key,
    )
