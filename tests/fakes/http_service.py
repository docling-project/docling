"""A FastAPI app served by uvicorn on an ephemeral localhost port.

Docling reaches remote services through two client libraries -- ``httpx`` in
the service client, ``requests`` in the OCR/VLM API helpers -- and through
both sync and async code paths. Only a real socket serves all of them from
one harness, which rules out ASGI-transport and library-specific mocks
(``respx`` sees no ``requests`` traffic, ``responses`` sees no ``httpx``).

FastAPI is used because the services being faked are themselves FastAPI apps:
routes can be written with the same decorators and ``response_model`` as the
originals, so a reviewer can diff them, and ``StreamingResponse`` covers SSE
without hand-rolling chunked encoding.

Route packs build on this: see :mod:`tests.fakes.docling_serve` and
:mod:`tests.fakes.openai_compatible`.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RecordedRequest:
    """A request as the server received it."""

    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    def param(self, name: str) -> str | None:
        values = self.query.get(name)
        return values[0] if values else None


@dataclass
class Response:
    """A canned response used for test overrides and fault injection.

    ``body`` may be ``bytes``, ``str``, or any JSON-serialisable object; the
    last is encoded as JSON. Real behaviour belongs in the route packs -- this
    exists so a test can force a 429, a 503 or a malformed body onto an
    otherwise healthy route.
    """

    status: int = 200
    body: Any = b""
    headers: dict[str, str] = field(default_factory=dict)

    def to_starlette(self) -> JSONResponse | PlainTextResponse:
        if isinstance(self.body, (bytes, str)):
            content = self.body.decode() if isinstance(self.body, bytes) else self.body
            return PlainTextResponse(
                content, status_code=self.status, headers=self.headers
            )
        return JSONResponse(
            json.loads(json.dumps(self.body, default=str)),
            status_code=self.status,
            headers=self.headers,
        )


Override = Callable[[RecordedRequest], "Response | None"]


class FakeService:
    """Hosts one or more route packs and records what it served.

    Overrides registered by tests take precedence over the mounted routes,
    which is how faults are injected without stubbing the client.
    """

    def __init__(self) -> None:
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.requests: list[RecordedRequest] = []
        self.base_url = ""
        self._overrides: list[tuple[str, re.Pattern[str], Override]] = []
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._install_middleware()

    # -- composition -----------------------------------------------------

    def include(self, router: APIRouter) -> None:
        """Mount a route pack."""
        self.app.include_router(router)

    # -- test overrides --------------------------------------------------

    def add_route(
        self, method: str, pattern: str, handler: Callable[..., Response]
    ) -> None:
        """Always serve ``handler`` for requests matching ``method``/``pattern``.

        Takes precedence over any mounted route, so a test can replace a
        healthy endpoint with a failing one for its duration.
        """
        compiled = re.compile(pattern)

        def override(request: RecordedRequest) -> Response:
            return handler(request, compiled.fullmatch(request.path))

        self._overrides.insert(0, (method.upper(), compiled, override))

    def respond_once(self, method: str, pattern: str, response: Response) -> None:
        """Serve ``response`` for the next matching request only."""
        used = threading.Event()

        def override(request: RecordedRequest) -> Response | None:
            if used.is_set():
                return None  # fall through to the real route
            used.set()
            return response

        self._overrides.insert(0, (method.upper(), re.compile(pattern), override))

    def _find_override(self, request: RecordedRequest) -> Response | None:
        for method, pattern, override in self._overrides:
            if method != request.method or not pattern.fullmatch(request.path):
                continue
            response = override(request)
            if response is not None:
                return response
        return None

    # -- recording -------------------------------------------------------

    def _install_middleware(self) -> None:
        service = self

        class _Recorder(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                body = await request.body()
                recorded = RecordedRequest(
                    method=request.method,
                    path=request.url.path,
                    query={
                        key: request.query_params.getlist(key)
                        for key in request.query_params
                    },
                    headers={k.lower(): v for k, v in request.headers.items()},
                    body=body,
                )
                with service._lock:
                    service.requests.append(recorded)
                forced = service._find_override(recorded)
                if forced is not None:
                    return forced.to_starlette()
                return await call_next(request)

        self.app.add_middleware(_Recorder)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> str:
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=0, log_level="warning", access_log=False
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # port=0 means the real port is known only once the socket is bound.
        tick = threading.Event()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("the fake service failed to start")
            tick.wait(0.01)
        host, port = self._server.servers[0].sockets[0].getsockname()[:2]
        self.base_url = f"http://{host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def __enter__(self) -> FakeService:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- assertions ------------------------------------------------------

    def requests_for(self, method: str, pattern: str) -> list[RecordedRequest]:
        compiled = re.compile(pattern)
        return [
            r
            for r in self.requests
            if r.method == method.upper() and compiled.fullmatch(r.path)
        ]
