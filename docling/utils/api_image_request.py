# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import base64
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import requests
from PIL import Image
from pydantic import AnyUrl, ValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from docling.datamodel.api_answer_fields import ApiAnswerField
from docling.datamodel.base_models import (
    ApiImageRequestResult,
    ApiImageStreamingRequestResult,
    OpenAiApiResponse,
    OpenAiChatMessage,
    OpenAiFunctionCall,
    OpenAiResponseLogprobs,
    OpenAiTokenLogprob,
    OpenAiToolCall,
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


@dataclass
class _ResponseFields:
    """The text fields of one chat response, from a message or from stream deltas."""

    content: str = ""
    refusal: str = ""
    reasoning: str = ""
    # Keyed by the stream `index`, so fragments of one call join together.
    tool_calls: dict[int, OpenAiToolCall] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: OpenAiChatMessage) -> "_ResponseFields":
        return cls(
            content=message.content or "",
            refusal=message.refusal or "",
            reasoning=message.reasoning or "",
            tool_calls=dict(enumerate(message.tool_calls or [])),
        )

    def add_delta(self, delta: dict[str, Any]) -> None:
        self.content += delta.get("content") or ""
        self.refusal += delta.get("refusal") or ""
        self.reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
        for fragment in delta.get("tool_calls") or []:
            function = fragment.get("function") or {}
            arguments = function.get("arguments") or ""
            index = fragment.get("index", 0)
            if index in self.tool_calls:
                self.tool_calls[index].function.arguments += arguments
            else:
                self.tool_calls[index] = OpenAiToolCall(
                    id=fragment.get("id"),
                    type=fragment.get("type") or "function",
                    function=OpenAiFunctionCall(
                        name=function.get("name"), arguments=arguments
                    ),
                )

    def text_of(self, answer_field: ApiAnswerField) -> str:
        if answer_field == ApiAnswerField.CONTENT:
            return self.content
        if answer_field == ApiAnswerField.REASONING:
            return self.reasoning
        # The raw string: its schema belongs to the tool the client sent, and
        # the selected ResponseFormat parser interprets it.
        return next(
            (
                call.function.arguments
                for call in self.tool_calls.values()
                if call.function.arguments.strip()
            ),
            "",
        )

    def select(self, answer_fields: Sequence[ApiAnswerField]) -> str:
        for answer_field in answer_fields:
            text = self.text_of(answer_field).strip()
            if text:
                return text
        return ""


def _resolve_answer(
    fields: _ResponseFields,
    answer_fields: Sequence[ApiAnswerField],
    stop_reason: VlmStopReason,
) -> tuple[str, VlmStopReason, str | None]:
    """Select the answer, and report an empty answer instead of hiding it.

    Returns the text, the stop reason and an error message.
    """
    text = fields.select(answer_fields)
    if text:
        return text, stop_reason, None

    if fields.refusal.strip():
        error = f"The model refused: {_response_preview(fields.refusal)}"
    elif stop_reason == VlmStopReason.LENGTH:
        # The pipeline already reports the output as incomplete.
        return "", stop_reason, None
    else:
        unselected = {
            answer_field.value: len(fields.text_of(answer_field).strip())
            for answer_field in ApiAnswerField
            if answer_field not in answer_fields
        }
        found = [f"{name} has {n} characters" for name, n in unselected.items() if n]
        if not found:
            return "", stop_reason, None
        selected = ", ".join(answer_field.value for answer_field in answer_fields)
        error = (
            f"No answer in {selected}. {'; '.join(found)}. If the server puts "
            "the answer there, set answer_fields."
        )
    _log.warning(error)
    return "", VlmStopReason.INFERENCE_ERROR, error


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
    answer_fields: Sequence[ApiAnswerField] = (ApiAnswerField.CONTENT,),
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
            fields = _ResponseFields.from_message(choice.message)
            generated_text, stop_reason, error = _resolve_answer(
                fields, answer_fields, _map_stop_reason(choice.finish_reason)
            )
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
                reasoning=fields.reasoning or None,
                tool_calls=tuple(fields.tool_calls.values()),
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
    answer_fields: Sequence[ApiAnswerField] = (ApiAnswerField.CONTENT,),
    **params,
) -> ApiImageStreamingRequestResult:
    """
    Stream a chat completion from an OpenAI-compatible server (e.g., vLLM).
    Parses SSE lines: 'data: {json}\\n\\n', terminated by 'data: [DONE]'.
    Accumulates the response fields and calls stopper.should_stop(window) on
    each field in ``answer_fields`` as chunks arrive.
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

                    fields = _ResponseFields()
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
                        text, stop_reason, error = _resolve_answer(
                            fields, answer_fields, stop_reason
                        )
                        return ApiImageStreamingRequestResult(
                            text=text,
                            num_tokens=num_tokens,
                            usage=usage_payload,
                            logprobs=logprobs,
                            error=error,
                            stop_reason=stop_reason,
                            reasoning=fields.reasoning or None,
                            tool_calls=tuple(fields.tool_calls.values()),
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

                        lengths = [len(fields.text_of(f)) for f in answer_fields]
                        fields.add_delta(delta)
                        for answer_field, length in zip(answer_fields, lengths):
                            text = fields.text_of(answer_field)
                            if len(text) == length:
                                continue
                            for stopper in generation_stoppers:
                                # Respect stopper's lookback window, one window per
                                # answer field. A simple string window works with
                                # the GenerationStopper interface.
                                lookback = max(1, stopper.lookback_tokens())
                                window = text[-lookback:]
                                if stopper.should_stop(window):
                                    # Break out of the loop cleanly. The context manager will handle
                                    # closing the connection when we exit the 'with' block.
                                    # vLLM/OpenAI-compatible servers will detect the client
                                    # disconnect and abort the request server-side.
                                    if stopper.is_valid_end():
                                        return _result(VlmStopReason.END_OF_SEQUENCE)
                                    return _result(VlmStopReason.STOP_SEQUENCE)

                    return _result(_map_stop_reason(finish_reason))
        except Exception as e:
            _log.error(f"Error, could not process request: {e}")
            return _failed_streaming_request(f"{type(e).__name__}: {e}")
    else:
        return _failed_streaming_request("Could not encode the page image as PNG")
