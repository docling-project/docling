import base64
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image
from pydantic import AnyUrl

from docling.datamodel.base_models import OpenAiApiResponse
from docling.models.utils.generation_utils import GenerationStopper

_log = logging.getLogger(__name__)


def api_image_request(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    timeout: float = 20,
    headers: Optional[Dict[str, str]] = None,
    token_extract_key: Optional[str] = None,
    **params,
) -> Tuple[str, Optional[dict]]:
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
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload = {"messages": messages, **params}
    headers = headers or {}

    r = requests.post(str(url), headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        _log.error(f"Error calling the API. Response was {r.text}")
    r.raise_for_status()

    # Try to parse JSON body
    try:
        resp_json = r.json()
    except Exception:
        api_resp = OpenAiApiResponse.model_validate_json(r.text)
        generated_text = api_resp.choices[0].message.content.strip()
        return generated_text, None

    usage = None
    if isinstance(resp_json, dict):
        usage = resp_json.get("usage")

    # Extract generated text using common OpenAI shapes
    generated_text = ""
    try:
        generated_text = resp_json["choices"][0]["message"]["content"].strip()
    except Exception:
        try:
            generated_text = resp_json["choices"][0].get("text", "")
            if isinstance(generated_text, str):
                generated_text = generated_text.strip()
        except Exception:
            try:
                api_resp = OpenAiApiResponse.model_validate_json(r.text)
                generated_text = api_resp.choices[0].message.content.strip()
            except Exception:
                generated_text = ""

    # If an explicit token_extract_key is provided and found in usage, use it
    if token_extract_key and isinstance(usage, dict) and token_extract_key in usage:
        extracted = usage.get(token_extract_key)
        generated_text = (
            str(extracted).strip() if extracted is not None else generated_text
        )

    return generated_text, usage


def api_image_request_streaming(
    image: Image.Image,
    prompt: str,
    url: AnyUrl,
    *,
    timeout: float = 20,
    headers: Optional[Dict[str, str]] = None,
    generation_stoppers: List[GenerationStopper] = [],
    **params,
) -> str:
    """
    Stream a chat completion from an OpenAI-compatible server (e.g., vLLM).
    Parses SSE lines: 'data: {json}\n\n', terminated by 'data: [DONE]'.
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

    payload = {"messages": messages, "stream": True, **params}
    _log.debug(f"API streaming request payload: {json.dumps(payload, indent=2)}")

    hdrs = {"Accept": "text/event-stream", **(headers or {})}
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

        full_text: List[str] = []
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

            try:
                delta = obj["choices"][0].get("delta") or {}
                piece = delta.get("content") or ""
            except (KeyError, IndexError) as e:
                _log.debug("Unexpected SSE chunk shape: %s", e)
                piece = ""

            if piece:
                full_text.append(piece)
                for stopper in generation_stoppers:
                    lookback = max(1, stopper.lookback_tokens())
                    window = "".join(full_text)[-lookback:]
                    if stopper.should_stop(window):
                        return "".join(full_text)

        return "".join(full_text)
