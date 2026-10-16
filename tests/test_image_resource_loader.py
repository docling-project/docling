# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for the shared image-resource loader and its safety limits."""

import base64
import socket
import threading
from io import BytesIO
from urllib.parse import quote

import pytest
from PIL import Image

from docling.backend.utils.image_resource_loader import (
    ImageResourceLoader,
    validate_url_safety,
)
from docling.exceptions import OperationNotAllowed
from tests.fakes.image_server import SERVER_IP, local_server, use_test_network

_TEST_HOSTS = {
    "images.test": [SERVER_IP],
    "cdn.test": [SERVER_IP],
    "internal.test": ["10.0.0.7"],
    "dual.test": [SERVER_IP, "::1"],
}


@pytest.fixture
def server(monkeypatch):
    use_test_network(monkeypatch, _TEST_HOSTS)
    with local_server() as state:
        yield state


def _redirect(server, host: str, target: str) -> str:
    return server.url(host, f"/redirect?to={quote(target, safe='')}")


def test_validate_url_safety_requires_hostname():
    with pytest.raises(ValueError, match="must contain a valid hostname"):
        validate_url_safety("https:///no-host")


def test_validate_url_safety_rejects_unresolvable_hostname(monkeypatch):
    def failing_getaddrinfo(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", failing_getaddrinfo)
    with pytest.raises(ValueError, match="Cannot resolve hostname"):
        validate_url_safety("http://does-not-exist.invalid/file")


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "169.254.169.254",
        "[::1]",
        "[::ffff:127.0.0.1]",  # IPv4-mapped loopback
        "[64:ff9b::7f00:1]",  # NAT64-embedded 127.0.0.1
        "[64:ff9b:1::5db8:d822]",  # local-use NAT64 prefix
        "[2002:7f00:1::]",  # 6to4-embedded 127.0.0.1
    ],
)
def test_validate_url_safety_rejects_non_public_ip_literals(host):
    with pytest.raises(ValueError, match="restricted IP address"):
        validate_url_safety(f"http://{host}/file")


def test_validate_url_safety_rejects_when_any_record_is_private(monkeypatch):
    """A public IPv4 record does not excuse a loopback IPv6 record."""
    use_test_network(monkeypatch, {"dual.test": ["93.184.216.34", "::1"]})
    with pytest.raises(ValueError, match="restricted IP address"):
        validate_url_safety("http://dual.test/file")


def test_load_image_data_skips_svg():
    loader = ImageResourceLoader(enable_remote_fetch=True)
    assert loader.load_image_data("http://example.com/logo.svg", None) is None


def test_fetch_from_public_host(server):
    loader = ImageResourceLoader(enable_remote_fetch=True)
    data = loader.load_image_data(server.url("images.test", "/img.png"), None)
    assert data is not None and data.startswith(b"\x89PNG")


@pytest.mark.parametrize("host", ["internal.test", "dual.test", "localhost"])
def test_fetch_refuses_non_public_host(server, host):
    loader = ImageResourceLoader(enable_remote_fetch=True)
    with pytest.raises(ValueError, match="restricted IP address"):
        loader.load_image_data(server.url(host, "/img.png"), None)
    assert server.requests == []


def test_fetch_refuses_redirect_to_non_public_host(server):
    loader = ImageResourceLoader(enable_remote_fetch=True)
    start = _redirect(server, "images.test", server.url("localhost", "/img.png"))
    with pytest.raises(ValueError, match="restricted IP address"):
        loader.load_image_data(start, None)
    assert server.paths() == ["/redirect"]


def test_fetch_follows_redirects_up_to_limit(server):
    target = server.url("cdn.test", "/img.png")
    once = _redirect(server, "images.test", target)
    twice = _redirect(server, "images.test", once)

    loader = ImageResourceLoader(enable_remote_fetch=True, max_redirects=1)
    assert loader.load_image_data(once, None) is not None
    with pytest.raises(ValueError, match="maximum number of redirects"):
        loader.load_image_data(twice, None)


def test_fetch_connects_to_the_address_it_validated(server, monkeypatch):
    """Each hop is resolved once and the connection goes to that address.

    Later answers for the name (here a loopback address that is not allowed)
    are never used, and the original hostname is sent in the Host header.
    """
    answers = [SERVER_IP, "::1"]
    lookups: list[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def changing_getaddrinfo(host, port, *args, **kwargs):
        if host == "changing.test":
            ip = answers[min(len(lookups), 1)]
            lookups.append(ip)
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port or 0, 0, 0) if ":" in ip else (ip, port or 0)
            return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", changing_getaddrinfo)
    loader = ImageResourceLoader(enable_remote_fetch=True)
    data = loader.load_image_data(server.url("changing.test", "/img.png"), None)

    assert data is not None and data.startswith(b"\x89PNG")
    assert lookups == [SERVER_IP]
    assert [r.host for r in server.requests] == [f"changing.test:{server.port}"]


def test_fetch_tries_next_validated_address(monkeypatch):
    """An address that refuses the connection is skipped for the next one."""
    use_test_network(
        monkeypatch,
        {"fallback.test": ["::1", SERVER_IP]},
        public_ips=["::1", SERVER_IP],
    )
    with local_server() as server:
        loader = ImageResourceLoader(enable_remote_fetch=True)
        data = loader.load_image_data(server.url("fallback.test", "/img.png"), None)
        assert server.paths() == ["/img.png"]
    assert data is not None and data.startswith(b"\x89PNG")


def test_fetch_goes_through_configured_proxy(server, monkeypatch):
    """With a proxy configured, the proxy connects to the destination."""
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTP_PROXY", f"http://{SERVER_IP}:{server.port}")
    loader = ImageResourceLoader(enable_remote_fetch=True)
    data = loader.load_image_data("http://proxied.test/img.png", None)

    assert data is not None and data.startswith(b"\x89PNG")
    assert [r.path for r in server.requests] == ["http://proxied.test/img.png"]


def test_fetch_does_not_affect_other_name_lookups(server):
    """While a fetch is in flight, lookups in other threads resolve normally."""
    loader = ImageResourceLoader(enable_remote_fetch=True)
    errors: list[BaseException] = []

    def fetch() -> None:
        try:
            loader.load_image_data(server.url("images.test", "/slow.png"), None)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=fetch)
    worker.start()
    try:
        assert server.slow_started.wait(timeout=10)
        infos = socket.getaddrinfo("localhost", 80, proto=socket.IPPROTO_TCP)
        assert infos
    finally:
        server.release_slow.set()
        worker.join(timeout=10)
    assert errors == []


def test_headers_sent_only_to_allowed_origins(server):
    loader = ImageResourceLoader(
        enable_remote_fetch=True,
        headers={"X-Api-Key": "k"},
        header_origins=[server.url("images.test", "")],
    )
    loader.load_image_data(server.url("images.test", "/img.png"), None)
    loader.load_image_data(server.url("cdn.test", "/img.png"), None)
    loader.load_image_data(
        _redirect(server, "images.test", server.url("cdn.test", "/img.png")), None
    )

    received = [
        (r.host.split(":")[0], r.headers.get("X-Api-Key")) for r in server.requests
    ]
    assert received == [
        ("images.test", "k"),
        ("cdn.test", None),
        ("images.test", "k"),
        ("cdn.test", None),
    ]


def test_headers_not_sent_without_allowed_origin(server):
    loader = ImageResourceLoader(enable_remote_fetch=True, headers={"X-Api-Key": "k"})
    loader.load_image_data(server.url("images.test", "/img.png"), None)
    assert "X-Api-Key" not in server.requests[0].headers


def test_fetch_exceeding_size_limit(server):
    loader = ImageResourceLoader(enable_remote_fetch=True, max_remote_image_bytes=10)
    with pytest.raises(ValueError, match="size"):
        loader.load_image_data(server.url("images.test", "/img.png"), None)


def test_remote_fetch_disabled():
    loader = ImageResourceLoader()
    with pytest.raises(OperationNotAllowed):
        loader.load_image_data("http://images.test/img.png", None)


def test_load_image_data_missing_local_file(tmp_path):
    loader = ImageResourceLoader(enable_local_fetch=True)
    base_path = str(tmp_path / "doc.html")
    missing = str(tmp_path / "missing.png")

    with pytest.raises(ValueError, match="File does not exist or it is not readable"):
        loader.load_image_data(missing, base_path)


def test_load_image_data_local_requires_base_path():
    loader = ImageResourceLoader(enable_local_fetch=True)
    with pytest.raises(OperationNotAllowed, match="requires base_path"):
        loader.load_image_data("/some/where/image.png", None)


def _jpeg_data_uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_create_image_ref_skips_undecodable_image():
    # A truncated file only fails when Pillow decodes it, with an OSError.
    data = _jpeg(Image.new("RGB", (200, 100), "red"))
    truncated = _jpeg_data_uri(data[: len(data) // 2])

    with pytest.warns(UserWarning, match="Could not process an image"):
        assert ImageResourceLoader().create_image_ref(truncated, None) is None
