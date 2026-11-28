# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
import requests

from docling.utils.utils import download_url_with_progress


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status = int(self.path.removeprefix("/"))
            if status != 200:
                self.send_error(status)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"model bytes")

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_download_url_returns_rewound_stream(server_url: str) -> None:
    stream = download_url_with_progress(f"{server_url}/200")
    assert stream.tell() == 0
    assert stream.read() == b"model bytes"


@pytest.mark.parametrize("status", [404, 500])
def test_download_url_raises_on_http_error(server_url: str, status: int) -> None:
    with pytest.raises(requests.HTTPError) as exc:
        download_url_with_progress(f"{server_url}/{status}")
    assert exc.value.response is not None
    assert exc.value.response.status_code == status
