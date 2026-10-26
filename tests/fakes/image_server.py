# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""A real local HTTP server and name mapping for remote-fetch tests.

The server listens on 127.0.0.1. :func:`use_test_network` maps test hostnames to
fixed addresses and treats 127.0.0.1 as a public address, so fetches to the
local server exercise the same code path as fetches to a public host, while any
other loopback or private address keeps being refused.
"""

import base64
import ipaddress
import mimetypes
import socket
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from requests.structures import CaseInsensitiveDict

from docling.backend.utils import image_resource_loader

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)

# The address the test server listens on, accepted as "public" by the tests.
SERVER_IP = "127.0.0.1"


@dataclass
class RecordedRequest:
    host: str
    path: str
    headers: CaseInsensitiveDict[str]


@dataclass
class LocalServer:
    port: int
    files: dict[str, bytes] = field(default_factory=lambda: {"/img.png": PNG_1X1})
    requests: list[RecordedRequest] = field(default_factory=list)
    release_slow: threading.Event = field(default_factory=threading.Event)
    slow_started: threading.Event = field(default_factory=threading.Event)

    def url(self, host: str, path: str) -> str:
        return f"http://{host}:{self.port}{path}"

    def paths(self) -> list[str]:
        return [urlparse(r.path).path for r in self.requests]


@contextmanager
def local_server() -> Iterator[LocalServer]:
    """Serve a tiny set of endpoints on 127.0.0.1.

    - Any path in ``files`` (by default ``/img.png``, a 1x1 PNG).
    - ``/redirect?to=URL``: a 302 redirect to ``URL``.
    - ``/slow.png``: a PNG, sent once ``release_slow`` is set.
    """
    state = LocalServer(port=0)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            state.requests.append(
                RecordedRequest(
                    host=self.headers.get("Host", ""),
                    path=self.path,
                    headers=CaseInsensitiveDict(self.headers.items()),
                )
            )
            parsed = urlparse(self.path)
            if parsed.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", parse_qs(parsed.query)["to"][0])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/slow.png":
                state.slow_started.set()
                state.release_slow.wait(timeout=10)
                body = PNG_1X1
            elif parsed.path in state.files:
                body = state.files[parsed.path]
            else:
                self.send_error(404)
                return
            content_type = mimetypes.guess_type(parsed.path)[0]
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer((SERVER_IP, 0), Handler)
    state.port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        state.release_slow.set()
        server.shutdown()
        server.server_close()


def use_test_network(
    monkeypatch: pytest.MonkeyPatch,
    hosts: dict[str, list[str]],
    public_ips: Iterable[str] = (SERVER_IP,),
) -> None:
    """Resolve ``hosts`` to fixed addresses and accept ``public_ips`` as public.

    Other names are resolved normally. Proxy variables are cleared so requests
    connect directly.
    """
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)

    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host in hosts:
            infos = []
            for ip in hosts[host]:
                if ipaddress.ip_address(ip).version == 6:
                    infos.append(
                        (
                            socket.AF_INET6,
                            socket.SOCK_STREAM,
                            6,
                            "",
                            (ip, port or 0, 0, 0),
                        )
                    )
                else:
                    infos.append(
                        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
                    )
            return infos
        return real_getaddrinfo(host, port, *args, **kwargs)

    real_gethostbyname = socket.gethostbyname

    def fake_gethostbyname(host):
        if host in hosts:
            return next(ip for ip in hosts[host] if ":" not in ip)
        return real_gethostbyname(host)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)

    real_is_restricted = image_resource_loader._ip_is_restricted
    allowed = {ipaddress.ip_address(ip) for ip in public_ips}
    monkeypatch.setattr(
        image_resource_loader,
        "_ip_is_restricted",
        lambda ip: ip not in allowed and real_is_restricted(ip),
    )
