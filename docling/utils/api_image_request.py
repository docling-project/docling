import base64
import json
import logging
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from pydantic import AnyUrl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ApiImageStreamingRequestResult,
    OpenAiApiResponse,
    OpenAiChatMessage,
    VlmStopReason,
)
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)

_RETRY_TOTAL = 5
_RETRY_BACKOFF_FACTOR = 0.1
_RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def _make_retry_session() -> requests.Session:
    retry_strategy = Retry(
        total=_RETRY_TOTAL,
        connect=_RETRY_TOTAL,
        read=0,
        status=_RETRY_TOTAL,
        allowed_methods={"POST"},
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUS_FORCELIST,
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _extract_text_from_tool_arguments(arguments: str | None) -> str:
    if arguments is None:
        return ""

    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments.strip()

    fragments: list[str] = []

    def _collect_text(obj: Any) -> None:
        if isinstance(obj, list):
            for item in obj:
                _collect_text(item)
        elif isinstance(obj, dict):
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
            for value in obj.values():
                _collect_text(value)

    _collect_text(payload)
    return "\n".join(fragments)


def _extract_generated_text(message: OpenAiChatMessage) -> str:
    if message.content is not None:
        return message.content.strip()

    for tool_call in message.tool_calls or []:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue

        generated_text = _extract_text_from_tool_arguments(function.get("arguments"))
        if generated_text:
            return generated_text

    return ""


def _map_stop_reason(finish_reason: str | None) -> VlmStopReason:
    if finish_reason == "content_filter":
        _log.warning("API response was filtered due to content safety policy.")
        return VlmStopReason.CONTENT_FILTERED
    elif finish_reason == "length":
        return VlmStopReason.LENGTH
    else:
        return VlmStopReason.END_OF_SEQUENCE


def _extract_response_value(response_payload: Any, key: str | None) -> Any | None:
    if key is None or not isinstance(response_payload, dict):
        return None

    current: Any = response_payload
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _extract_response_usage(
    response_payload: Any, usage_response_key: str | None
) -> Any | None:
    return _extract_response_value(response_payload, usage_response_key)


def _extract_total_tokens(usage: Any) -> int | None:
    if isinstance(usage, dict):
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int):
            return total_tokens
    return None


def _response_preview(text: str, *, limit: int = 500) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}..."


def _parse_response_json(response: requests.Response) -> Any | None:
    if not response.text.strip():
        _log.error(
            "API response body was empty. status=%s content_type=%s",
            response.status_code,
            response.headers.get("content-type"),
        )
        return None

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        _log.error(
            "API response body was not JSON: %s. status=%s content_type=%s response=%r",
            e,
            response.status_code,
            response.headers.get("content-type"),
            _response_preview(response.text),
        )
        return None


def _resolve_usage_response_key(
    usage_response_key: str | None,
    token_extract_key: str | None,
) -> str | None:
    if token_extract_key is not None:
        return token_extract_key
    return usage_response_key


def api_image_request(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: dict[str, str] | None = None,
    *,
    usage_response_key: str | None = "usage",
    token_extract_key: str | None = None,
    **params,
) -> ApiImageRequestResult:
    img_io = BytesIO()
    image = (
        image.copy()
    )  # Fix for inconsistent PIL image width/height to actual byte data
    image = image.convert("RGBA")
    good_image = True
    try:
        image.save(img_io, "PNG")
    except Exception as e:
        good_image = False
        _log.error(f"Error, corrupted PNG of size: {image.size}: {e}")

    if good_image:
        try:
            image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
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

            with _make_retry_session() as session:
                r = session.post(
                    str(url),
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            if not r.ok:
                _log.error(
                    "Error calling the API. status=%s content_type=%s response=%r",
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
            )
        except Exception as e:
            _log.error(f"Error, could not process request: {e}")
            return ApiImageRequestResult("", 0, VlmStopReason.UNSPECIFIED)
    else:
        return ApiImageRequestResult("", 0, VlmStopReason.UNSPECIFIED)


def api_image_request_streaming(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    *,
    timeout: float = 20,
    headers: dict[str, str] | None = None,
    generation_stoppers: list[GenerationStopper] = [],
    usage_response_key: str | None = "usage",
    token_extract_key: str | None = None,
    **params,
) -> ApiImageStreamingRequestResult:
    """
    Stream a chat completion from an OpenAI-compatible server (e.g., vLLM).
    Parses SSE lines: 'data: {json}\\n\\n', terminated by 'data: [DONE]'.
    Accumulates text and calls stopper.should_stop(window) as chunks arrive.
    If stopper triggers, the HTTP connection is closed to abort server-side generation.
    """
    img_io = BytesIO()
    image.save(img_io, "PNG")
    image_b64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload = {
        "messages": messages,
        "stream": True,  # <-- critical for SSE streaming
        **params,
    }

    # Debug: Log the payload to verify temperature is included
    _log.debug(f"API streaming request payload: {json.dumps(payload, indent=2)}")

    # Some servers require Accept: text/event-stream for SSE.
    # It's safe to set it; OpenAI-compatible servers tolerate it.
    hdrs = {"Accept": "text/event-stream", **(headers or {})}

    # Try to force temperature via header if server ignores payload parameter
    if "temperature" in params:
        hdrs["X-Temperature"] = str(params["temperature"])

    # Stream the HTTP response
    with _make_retry_session() as session:
        with session.post(
            str(url), headers=hdrs, json=payload, timeout=timeout, stream=True
        ) as r:
            if not r.ok:
                _log.error(
                    f"Error calling the API {url} in streaming mode. "
                    f"Response was {r.text}"
                )
            r.raise_for_status()

            full_text = []
            usage_payload = None
            num_tokens = None
            usage_key = _resolve_usage_response_key(
                usage_response_key=usage_response_key,
                token_extract_key=token_extract_key,
            )
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:  # keep-alives / blank lines
                    continue
                if not raw_line.startswith("data:"):
                    # Some proxies inject comments; ignore anything not starting with 'data:'
                    continue

                data = raw_line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    _log.debug("Skipping non-JSON SSE chunk: %r", data[:200])
                    continue

                # OpenAI-compatible delta format
                # obj["choices"][0]["delta"]["content"] may be None or missing
                # (e.g., tool calls)
                try:
                    delta = obj["choices"][0].get("delta") or {}
                    piece = delta.get("content") or ""
                except (KeyError, IndexError) as e:
                    _log.debug("Unexpected SSE chunk shape: %s", e)
                    piece = ""

                usage = _extract_response_usage(obj, usage_key)
                if usage is not None:
                    usage_payload = usage
                    num_tokens = _extract_total_tokens(usage)

                if piece:
                    full_text.append(piece)
                    for stopper in generation_stoppers:
                        # Respect stopper's lookback window. We use a simple string window
                        # which works with the GenerationStopper interface.
                        lookback = max(1, stopper.lookback_tokens())
                        window = "".join(full_text)[-lookback:]
                        if stopper.should_stop(window):
                            # Break out of the loop cleanly. The context manager will handle
                            # closing the connection when we exit the 'with' block.
                            # vLLM/OpenAI-compatible servers will detect the client
                            # disconnect and abort the request server-side.
                            return ApiImageStreamingRequestResult(
                                text="".join(full_text),
                                num_tokens=num_tokens,
                                usage=usage_payload,
                            )

            return ApiImageStreamingRequestResult(
                text="".join(full_text),
                num_tokens=num_tokens,
                usage=usage_payload,
            )
