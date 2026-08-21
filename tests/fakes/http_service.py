"""A real HTTP server for tests, bound to an ephemeral localhost port.

Docling reaches remote services through two different client libraries --
``httpx`` in the service client and ``requests`` in the KServe and image
helpers -- and through both sync and async code paths. Serving a real socket
covers all of them with one fake, and keeps transport behaviour (timeouts,
connection resets, retries, redirects) reachable from tests, which a
library-specific mock transport cannot do.

Route packs build on this: see :mod:`tests.fakes.docling_serve`.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit


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
    """What a route handler returns.

    ``body`` may be ``bytes``, ``str``, or any JSON-serialisable object; the
    last is encoded as JSON and given a JSON content type unless one is set.
    ``delay`` holds the response open, which is how read-timeout handling is
    exercised.
    """

    status: int = 200
    body: Any = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0

    def encoded(self) -> tuple[bytes, dict[str, str]]:
        headers = dict(self.headers)
        if isinstance(self.body, bytes):
            return self.body, headers
        if isinstance(self.body, str):
            headers.setdefault("Content-Type", "text/plain; charset=utf-8")
            return self.body.encode(), headers
        headers.setdefault("Content-Type", "application/json")
        return json.dumps(self.body, default=str).encode(), headers


Handler = Callable[[RecordedRequest, re.Match[str]], Response]


class FakeHttpService:
    """Routes requests to registered handlers and records what it served.

    Handlers are matched in registration order, so a test can shadow a route
    pack's handler by registering a more specific one first.
    """

    def __init__(self) -> None:
        self._routes: list[tuple[str, re.Pattern[str], Handler]] = []
        self.requests: list[RecordedRequest] = []
        self.base_url = ""
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- registration ----------------------------------------------------

    def route(self, method: str, pattern: str) -> Callable[[Handler], Handler]:
        """Register a handler for ``method`` and a full-match path regex."""

        def decorator(handler: Handler) -> Handler:
            self.add_route(method, pattern, handler)
            return handler

        return decorator

    def add_route(self, method: str, pattern: str, handler: Handler) -> None:
        self._routes.insert(0, (method.upper(), re.compile(pattern), handler))

    def respond_once(self, method: str, pattern: str, response: Response) -> None:
        """Serve ``response`` for the next matching request only.

        Used to inject a single fault -- a 429, a 503, a truncated body --
        ahead of an otherwise healthy route, so retry paths can be driven
        without stubbing the client.
        """
        used = threading.Event()

        def handler(request: RecordedRequest, match: re.Match[str]) -> Response:
            if used.is_set():
                return self._dispatch(request, skip_first=True)
            used.set()
            return response

        self.add_route(method, pattern, handler)

    # -- lifecycle -------------------------------------------------------

    def start(self) -> str:
        service = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass  # keep pytest output clean

            def _handle(self) -> None:
                parsed = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                request = RecordedRequest(
                    method=self.command,
                    path=parsed.path,
                    query=parse_qs(parsed.query),
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=self.rfile.read(length) if length else b"",
                )
                with service._lock:
                    service.requests.append(request)

                response = service._dispatch(request)
                if response.delay:
                    time.sleep(response.delay)
                payload, headers = response.encoded()
                self.send_response(response.status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _handle
            do_POST = _handle
            do_PUT = _handle
            do_DELETE = _handle
            do_HEAD = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # -- dispatch --------------------------------------------------------

    def _dispatch(
        self, request: RecordedRequest, *, skip_first: bool = False
    ) -> Response:
        matched = 0
        for method, pattern, handler in self._routes:
            if method != request.method:
                continue
            match = pattern.fullmatch(request.path)
            if match is None:
                continue
            matched += 1
            if skip_first and matched == 1:
                continue
            return handler(request, match)
        return Response(status=404, body={"detail": f"no route for {request.path}"})

    # -- assertions ------------------------------------------------------

    def requests_for(self, method: str, pattern: str) -> list[RecordedRequest]:
        compiled = re.compile(pattern)
        return [
            r
            for r in self.requests
            if r.method == method.upper() and compiled.fullmatch(r.path)
        ]
