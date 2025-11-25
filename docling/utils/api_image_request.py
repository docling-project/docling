import base64
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import OpenAiApiResponse, VlmStopReason
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def api_image_request(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[dict[str, str]] = None,
    **params,
) -> Tuple[str, Optional[int], VlmStopReason]:
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
        _log.error(f"Error, corrupter PNG of size: {image.size}: {e}")

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

            r = requests.post(
                str(url),
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if not r.ok:
                _log.error(f"Error calling the API. Response was {r.text}")
                # image.show()
            # r.raise_for_status()

            api_resp = OpenAiApiResponse.model_validate_json(r.text)
            generated_text = api_resp.choices[0].message.content.strip()
            num_tokens = api_resp.usage.total_tokens
            stop_reason = (
                VlmStopReason.LENGTH
                if api_resp.choices[0].finish_reason == "length"
                else VlmStopReason.END_OF_SEQUENCE
            )

            return generated_text, num_tokens, stop_reason
        except Exception as e:
            _log.error(f"Error, could not process request: {e}")
            return "", 0, VlmStopReason.UNSPECIFIED
    else:
        return "", 0, VlmStopReason.UNSPECIFIED


def api_image_request_streaming(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    *,
    timeout: float = 20,
    headers: Optional[dict[str, str]] = None,
    generation_stoppers: list[GenerationStopper] = [],
    **params,
) -> Tuple[str, Optional[int]]:
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
    with requests.post(
        str(url), headers=hdrs, json=payload, timeout=timeout, stream=True
    ) as r:
        if not r.ok:
            _log.error(
                f"Error calling the API {url} in streaming mode. Response was {r.text}"
            )
        r.raise_for_status()

        full_text = []
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
            # obj["choices"][0]["delta"]["content"] may be None or missing (e.g., tool calls)
            try:
                delta = obj["choices"][0].get("delta") or {}
                piece = delta.get("content") or ""
            except (KeyError, IndexError) as e:
                _log.debug("Unexpected SSE chunk shape: %s", e)
                piece = ""

            # Try to extract token count
            num_tokens = None
            try:
                if "usage" in obj:
                    usage = obj["usage"]
                    num_tokens = usage.get("total_tokens")
            except Exception as e:
                num_tokens = None
                _log.debug("Usage key not included in response: %s", e)

            if piece:
                full_text.append(piece)
                for stopper in generation_stoppers:
                    # Respect stopper's lookback window. We use a simple string window which
                    # works with the GenerationStopper interface.
                    lookback = max(1, stopper.lookback_tokens())
                    window = "".join(full_text)[-lookback:]
                    if stopper.should_stop(window):
                        # Break out of the loop cleanly. The context manager will handle
                        # closing the connection when we exit the 'with' block.
                        # vLLM/OpenAI-compatible servers will detect the client disconnect
                        # and abort the request server-side.
                        return "".join(full_text), num_tokens

        return "".join(full_text), num_tokens
