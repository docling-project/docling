# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""An OpenAI-compatible chat-completions route pack.

Every remote VLM and picture-description path in Docling funnels through
``docling.utils.api_image_request``, which uses ``requests`` -- so this has to
be served over a real socket rather than an httpx-only mock.

Non-streaming responses are built from this repo's own ``OpenAiApiResponse``
models. Streaming uses FastAPI's ``StreamingResponse`` to emit genuine SSE
chunks, so the client's ``iter_lines`` parsing, its accumulation of ``delta``
content and its generation stoppers all run against real chunked transfer
rather than one pre-assembled body.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from docling.datamodel.base_models import (
    OpenAiApiResponse,
    OpenAiChatMessage,
    OpenAiResponseChoice,
    OpenAiResponseUsage,
)

DEFAULT_COMPLETION = "Fake VLM output."


@dataclass
class FakeOpenAiApi:
    """State plus an ``APIRouter`` serving ``/v1/chat/completions``."""

    completion: str = DEFAULT_COMPLETION
    #: Emitted one SSE chunk at a time when the caller asks for a stream.
    stream_chunks: list[str] = field(default_factory=list)
    prompt_tokens: int = 11
    completion_tokens: int = 7
    #: Set to omit the usage block, as some gateways do.
    report_usage: bool = True
    #: Held open before responding, to drive client read timeouts.
    delay_seconds: float = 0.0
    #: Non-2xx status to answer with instead of a completion.
    fail_status: int | None = None
    #: Emitted verbatim before the data chunks; proxies inject comments here.
    stream_preamble: list[str] = field(default_factory=list)
    router: APIRouter = field(init=False)
    #: Set by the fixture once the server is bound, so tests can reach it.
    service: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.router = self._build_router()

    # -- payloads --------------------------------------------------------

    def _usage(self) -> OpenAiResponseUsage | None:
        if not self.report_usage:
            return None
        return OpenAiResponseUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
        )

    def _completion_response(self, model: str | None) -> dict[str, Any]:
        response = OpenAiApiResponse(
            id="chatcmpl-fake-1",
            model=model,
            created=1_700_000_000,
            choices=[
                OpenAiResponseChoice(
                    index=0,
                    message=OpenAiChatMessage(
                        role="assistant", content=self.completion
                    ),
                    finish_reason="stop",
                )
            ],
            usage=self._usage(),
        )
        return json.loads(response.model_dump_json())

    def _chunks(self) -> list[str]:
        return self.stream_chunks or [self.completion]

    async def _sse(self, model: str | None) -> AsyncIterator[bytes]:
        """Emit the OpenAI streaming delta format, terminated by [DONE]."""
        for line in self.stream_preamble:
            yield f"{line}\n\n".encode()
        for piece in self._chunks():
            event = {
                "id": "chatcmpl-fake-1",
                "model": model,
                "created": 1_700_000_000,
                "choices": [{"index": 0, "delta": {"content": piece}}],
            }
            yield f"data: {json.dumps(event)}\n\n".encode()
        usage = self._usage()
        if usage is not None:
            final = {
                "id": "chatcmpl-fake-1",
                "model": model,
                "created": 1_700_000_000,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": json.loads(usage.model_dump_json()),
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    # -- routes ----------------------------------------------------------

    def _build_router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            payload = json.loads(await request.body())
            model = payload.get("model")
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if self.fail_status is not None:
                return JSONResponse(
                    {"error": {"message": "fake failure"}},
                    status_code=self.fail_status,
                )
            if payload.get("stream"):
                return StreamingResponse(
                    self._sse(model), media_type="text/event-stream"
                )
            return JSONResponse(self._completion_response(model))

        return router
