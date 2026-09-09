# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Remote NuExtract request (dim 2).

NuExtract carries the schema/template out-of-band: the message content holds
only the document item(s), and the template rides ``chat_template_kwargs`` at
the top level of the POST body (vLLM merges the OpenAI client's ``extra_body``
into the body, so this is the same field). This differs from the plain
image-request shape, which puts the prompt in the message text.
"""

import base64
import logging
from io import BytesIO
from typing import Any

from pydantic import AnyUrl

from docling.datamodel.base_models import (
    ApiImageRequestResult,
    OpenAiApiResponse,
    VlmStopReason,
)
from docling.datamodel.extraction import (
    ContentItem,
    ImageContentItem,
    TextContentItem,
)
from docling.utils.api_image_request import (
    _extract_generated_text,
    _extract_response_usage,
    _extract_total_tokens,
    _make_retry_session,
    _map_stop_reason,
    _parse_response_json,
    _resolve_usage_response_key,
    _response_preview,
)

_log = logging.getLogger(__name__)


def _content_item_to_openai(item: ContentItem) -> dict[str, Any]:
    if isinstance(item, TextContentItem):
        return {"type": "text", "text": item.text}
    if isinstance(item, ImageContentItem):
        img = item.image.copy().convert("RGBA")
        buf = BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }
    raise ValueError(f"Unsupported content item: {type(item)}")


def api_nuextract_request(
    content_items: list[ContentItem],
    template: str,
    url: AnyUrl,
    timeout: float = 120,
    headers: dict[str, str] | None = None,
    *,
    usage_response_key: str | None = "usage",
    token_extract_key: str | None = None,
    **params: Any,
) -> ApiImageRequestResult:
    """POST one NuExtract request: document item(s) in content, template out-of-band."""
    try:
        content = [_content_item_to_openai(i) for i in content_items]

        payload = {
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"template": template},
            **params,
        }

        with _make_retry_session() as session:
            r = session.post(
                str(url),
                headers=headers or {},
                json=payload,
                timeout=timeout,
            )
        if not r.ok:
            _log.error(
                "Error calling the NuExtract API. status=%s content_type=%s response=%r",
                r.status_code,
                r.headers.get("content-type"),
                _response_preview(r.text),
            )
            return ApiImageRequestResult("", 0, VlmStopReason.UNSPECIFIED)

        response_payload = _parse_response_json(r)
        if response_payload is None:
            return ApiImageRequestResult("", 0, VlmStopReason.UNSPECIFIED)

        usage_key = _resolve_usage_response_key(
            usage_response_key=usage_response_key,
            token_extract_key=token_extract_key,
        )
        usage = _extract_response_usage(response_payload, usage_key)

        api_resp = OpenAiApiResponse.model_validate(response_payload)
        generated_text = _extract_generated_text(api_resp.choices[0].message)
        num_tokens = _extract_total_tokens(usage)
        if num_tokens is None and api_resp.usage is not None:
            num_tokens = api_resp.usage.total_tokens
        stop_reason = _map_stop_reason(api_resp.choices[0].finish_reason)

        return ApiImageRequestResult(
            text=generated_text,
            num_tokens=num_tokens,
            stop_reason=stop_reason,
            usage=usage,
            logprobs=api_resp.choices[0].logprobs,
        )
    except Exception as e:
        _log.error(f"Error, could not process NuExtract request: {e}")
        return ApiImageRequestResult("", 0, VlmStopReason.UNSPECIFIED)
