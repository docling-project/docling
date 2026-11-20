# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import base64
import json
import logging
from collections.abc import Iterable
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from pydantic import AnyUrl, ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ApiImageStreamingRequestResult,
    OpenAiApiResponse,
    OpenAiFunctionCall,
    OpenAiResponseLogprobs,
    OpenAiTokenLogprob,
    OpenAiToolCall,
    VlmStopReason,
)
from docling.models.utils.generation_utils import GenerationStopper
from docling.utils.stop_reason import normalize_stop_reason

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


def _failed_request(reason: str) -> ApiImageRequestResult:
    """Result for a request that produced no model output.

    The failure is reported through ``VlmStopReason.INFERENCE_ERROR`` and ``error``
    instead of an exception, so one failing page does not abort the document;
    the pipelines turn it into a PARTIAL_SUCCESS with a per-page ErrorItem.
    """
    return ApiImageRequestResult(
        text="", num_tokens=0, stop_reason=VlmStopReason.INFERENCE_ERROR, error=reason
    )


def _failed_streaming_request(reason: str) -> ApiImageStreamingRequestResult:
    """Streaming counterpart of ``_failed_request``.

    Text streamed before the failure is dropped: the output is incomplete, and
    the pipelines treat a failed request as producing no output.
    """
    return ApiImageStreamingRequestResult(
        text="",
        num_tokens=0,
        error=reason,
        stop_reason=VlmStopReason.INFERENCE_ERROR,
    )


def _extract_generated_text(
    content: str | None, tool_calls: Iterable[OpenAiToolCall]
) -> str:
    if content and content.strip():
        return content
    for call in tool_calls:
        if call.function.arguments.strip():
            return call.function.arguments
    return content or ""


def _map_stop_reason(finish_reason: str | None) -> VlmStopReason:
    if finish_reason == "content_filter":
        _log.warning("API response was filtered due to content safety policy.")
    return normalize_stop_reason(finish_reason)


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
                return _failed_request(
                    f"HTTP {r.status_code}: {_response_preview(r.text)}"
                )

            response_payload = _parse_response_json(r)
            if response_payload is None:
                return _failed_request(
                    f"HTTP {r.status_code}: response body was empty or not JSON"
                )

            usage_key = _resolve_usage_response_key(
                usage_response_key=usage_response_key,
                token_extract_key=token_extract_key,
            )
            usage = _extract_response_usage(response_payload, usage_key)

            api_resp = OpenAiApiResponse.model_validate(response_payload)
            choice = api_resp.choices[0]
            message = choice.message
            generated_text = _extract_generated_text(
                message.content, message.tool_calls or []
            ).strip()
            stop_reason = _map_stop_reason(choice.finish_reason)
            error = None
            if not generated_text and message.refusal and message.refusal.strip():
                error = f"The model refused: {_response_preview(message.refusal)}"
                stop_reason = VlmStopReason.INFERENCE_ERROR
            num_tokens = _extract_total_tokens(usage)
            if num_tokens is None and api_resp.usage is not None:
                num_tokens = api_resp.usage.total_tokens

            return ApiImageRequestResult(
                text=generated_text,
                num_tokens=num_tokens,
                stop_reason=stop_reason,
                usage=usage,
                logprobs=choice.logprobs,
                error=error,
                tool_calls=tuple(message.tool_calls or []),
            )
        except Exception as e:
            _log.error(f"Error, could not process request: {e}")
            return _failed_request(f"{type(e).__name__}: {e}")
    else:
        return _failed_request("Could not encode the page image as PNG")


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
    Accumulates content and raw tool-call arguments. Stoppers watch the answer
    as chunks arrive.
    If stopper triggers, the HTTP connection is closed to abort server-side generation.

    A failed request (HTTP error, timeout, dropped connection) does not raise:
    like ``api_image_request``, it returns an empty result with ``error`` set,
    so one failing page does not abort the document.
    """
    img_io = BytesIO()
    good_image = True
    try:
        image.save(img_io, "PNG")
    except Exception as e:
        good_image = False
        _log.error(f"Error, corrupted PNG of size: {image.size}: {e}")

    if good_image:
        try:
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
            _log.debug(
                f"API streaming request payload: {json.dumps(payload, indent=2)}"
            )

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
                            "Error calling the API in streaming mode. "
                            "status=%s content_type=%s response=%r",
                            r.status_code,
                            r.headers.get("content-type"),
                            _response_preview(r.text),
                        )
                        return _failed_streaming_request(
                            f"HTTP {r.status_code}: {_response_preview(r.text)}"
                        )

                    content = ""
                    refusal = ""
                    tool_calls: dict[int, OpenAiToolCall] = {}
                    text = ""
                    usage_payload = None
                    num_tokens = None
                    finish_reason = None
                    # Servers send logprobs per chunk; collect them so the streamed
                    # result exposes the same OpenAiResponseLogprobs as a non-streamed one.
                    logprob_content: list[OpenAiTokenLogprob] = []
                    logprob_refusal: list[OpenAiTokenLogprob] = []
                    usage_key = _resolve_usage_response_key(
                        usage_response_key=usage_response_key,
                        token_extract_key=token_extract_key,
                    )

                    def _result(
                        stop_reason: VlmStopReason,
                    ) -> ApiImageStreamingRequestResult:
                        logprobs = None
                        if logprob_content or logprob_refusal:
                            logprobs = OpenAiResponseLogprobs(
                                content=logprob_content or None,
                                refusal=logprob_refusal or None,
                            )
                        generated_text = text.strip()
                        error = None
                        if not generated_text and refusal.strip():
                            error = f"The model refused: {_response_preview(refusal)}"
                            stop_reason = VlmStopReason.INFERENCE_ERROR
                        return ApiImageStreamingRequestResult(
                            text=generated_text,
                            num_tokens=num_tokens,
                            usage=usage_payload,
                            logprobs=logprobs,
                            error=error,
                            stop_reason=stop_reason,
                            tool_calls=tuple(tool_calls[i] for i in sorted(tool_calls)),
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

                        # OpenAI-compatible delta format; the final usage chunk
                        # has empty choices.
                        try:
                            choice = obj["choices"][0]
                            delta = choice.get("delta") or {}
                            raw_logprobs = choice.get("logprobs")
                            # finish_reason is null until the server stops; keep the
                            # last value it sets.
                            finish_reason = choice.get("finish_reason") or finish_reason
                        except (KeyError, IndexError) as e:
                            _log.debug("Unexpected SSE chunk shape: %s", e)
                            delta = {}
                            raw_logprobs = None

                        usage = _extract_response_usage(obj, usage_key)
                        if usage is not None:
                            usage_payload = usage
                            num_tokens = _extract_total_tokens(usage)

                        if raw_logprobs:
                            try:
                                chunk_logprobs = OpenAiResponseLogprobs.model_validate(
                                    raw_logprobs
                                )
                            except ValidationError as e:
                                _log.debug("Skipping malformed SSE logprobs: %s", e)
                            else:
                                logprob_content.extend(chunk_logprobs.content or [])
                                logprob_refusal.extend(chunk_logprobs.refusal or [])

                        content += delta.get("content") or ""
                        refusal += delta.get("refusal") or ""
                        for fragment in delta.get("tool_calls") or []:
                            index = fragment.get("index", 0)
                            if index not in tool_calls:
                                tool_calls[index] = OpenAiToolCall(
                                    function=OpenAiFunctionCall()
                                )
                            call = tool_calls[index]
                            function = fragment.get("function") or {}
                            if fragment.get("id"):
                                call.id = (call.id or "") + fragment["id"]
                            if fragment.get("type"):
                                call.type = fragment["type"]
                            if function.get("name"):
                                call.function.name = (
                                    call.function.name or ""
                                ) + function["name"]
                            call.function.arguments += function.get("arguments") or ""

                        previous_text = text
                        text = _extract_generated_text(
                            content, (tool_calls[i] for i in sorted(tool_calls))
                        )
                        if text == previous_text:
                            continue
                        for stopper in generation_stoppers:
                            lookback = max(1, stopper.lookback_tokens())
                            if stopper.should_stop(text[-lookback:]):
                                # Closing the connection aborts server-side generation.
                                if stopper.is_valid_end():
                                    return _result(VlmStopReason.END_OF_SEQUENCE)
                                return _result(VlmStopReason.STOP_SEQUENCE)

                    return _result(_map_stop_reason(finish_reason))
        except Exception as e:
            _log.error(f"Error, could not process request: {e}")
            return _failed_streaming_request(f"{type(e).__name__}: {e}")
    else:
        return _failed_streaming_request("Could not encode the page image as PNG")
