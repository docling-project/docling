# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for the shared image-resource loader and its safety limits."""

import socket
from unittest.mock import Mock, patch

import pytest
import requests

from docling.backend.utils.image_resource_loader import (
    ImageResourceLoader,
    pinned_dns_resolution,
    validate_url_safety,
)
from docling.exceptions import OperationNotAllowed

# A globally routable IP literal (example.com) used so the source URL passes the
# SSRF check without a DNS lookup, keeping these tests offline.
_GLOBAL_IP_URL = "http://93.184.216.34/image.png"
# Two distinct public IP literals used to exercise same-origin vs cross-origin
# header scoping without any DNS resolution.
_ORIGIN_A = "http://93.184.216.34"
_ORIGIN_B = "http://93.184.216.35"


def _gai(ip: str, family):
    """Build a single socket.getaddrinfo-style result tuple for ``ip``."""
    if family == socket.AF_INET6:
        sockaddr = (ip, 0, 0, 0)
    else:
        sockaddr = (ip, 0)
    return (family, socket.SOCK_STREAM, 6, "", sockaddr)


def _remote_response(data=b"x", *, redirect_to=None):
    """A stand-in requests.Response for the manual redirect loop."""
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.iter_content = Mock(return_value=[data])
    resp.close = Mock()
    if redirect_to is not None:
        resp.is_redirect = True
        resp.is_permanent_redirect = False
        resp.headers = {"location": redirect_to}
    else:
        resp.is_redirect = False
        resp.is_permanent_redirect = False
        resp.headers = {}
    return resp


def test_validate_url_safety_requires_hostname():
    with pytest.raises(ValueError, match="must contain a valid hostname"):
        validate_url_safety("https:///no-host")


def test_validate_url_safety_rejects_unresolvable_hostname():
    with patch.object(socket, "gethostbyname", side_effect=socket.gaierror):
        with pytest.raises(ValueError, match="Cannot resolve hostname"):
            validate_url_safety("http://does-not-exist.invalid/file")


def test_load_image_data_skips_svg():
    loader = ImageResourceLoader(enable_remote_fetch=True)
    assert loader.load_image_data("http://example.com/logo.svg", None) is None


def test_load_image_data_revalidates_redirect_target():
    """A redirect to a restricted IP must be blocked, not silently followed."""
    loader = ImageResourceLoader(enable_remote_fetch=True)

    def fake_get(session, url, **kwargs):
        # Emulate requests dispatching the response hook on a redirect so the
        # registered safety check runs against the redirect target. A
        # protocol-relative location also exercises the relative-redirect join.
        redirect = Mock()
        redirect.is_redirect = True
        redirect.is_permanent_redirect = False
        redirect.headers = {"location": "//169.254.169.254/latest/meta-data"}
        redirect.url = url
        for hook in session.hooks["response"]:
            hook(redirect)
        return Mock()

    with patch.object(requests.Session, "get", fake_get):
        with pytest.raises(ValueError, match="restricted IP address"):
            loader.load_image_data(_GLOBAL_IP_URL, None)


def test_load_image_data_streaming_exceeds_size_limit():
    """Downloads without a content-length header are still capped while streaming."""
    loader = ImageResourceLoader(enable_remote_fetch=True, max_remote_image_bytes=10)

    def fake_get(session, url, **kwargs):
        resp = Mock()
        resp.headers = {}  # no content-length, so the cap is enforced per chunk
        resp.raise_for_status = Mock()
        resp.iter_content = Mock(return_value=[b"x" * 8, b"x" * 8])
        return resp

    with patch.object(requests.Session, "get", fake_get):
        with pytest.raises(ValueError, match="Downloaded data exceeds size limit"):
            loader.load_image_data(_GLOBAL_IP_URL, None)


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


def test_validate_url_safety_rejects_ipv6_only_private_host():
    """A host whose only address is a private IPv6 must be rejected."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [_gai("::1", socket.AF_INET6)]

    with patch.object(socket, "getaddrinfo", fake_getaddrinfo):
        with pytest.raises(ValueError, match="restricted IP address"):
            validate_url_safety("http://ipv6-only.example/file")


def test_validate_url_safety_rejects_when_any_record_is_private():
    """A public IPv4 A record does not excuse a private IPv6 AAAA record."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [
            _gai("93.184.216.34", socket.AF_INET),  # public
            _gai("::1", socket.AF_INET6),  # private/loopback
        ]

    with patch.object(socket, "getaddrinfo", fake_getaddrinfo):
        with pytest.raises(ValueError, match="restricted IP address"):
            validate_url_safety("http://dual-stack.example/file")


@pytest.mark.parametrize(
    "host",
    [
        "[::ffff:127.0.0.1]",  # IPv4-mapped loopback
        "[64:ff9b::7f00:1]",  # NAT64-embedded 127.0.0.1
        "[2002:7f00:1::]",  # 6to4-embedded 127.0.0.1
    ],
)
def test_validate_url_safety_rejects_ipv4_embedded_in_ipv6(host):
    with pytest.raises(ValueError, match="restricted IP address"):
        validate_url_safety(f"http://{host}/file")


def test_pinned_dns_resolution_blocks_private_address():
    """The pinned resolver validates every address before a connection is made."""

    def fake_getaddrinfo(host, *args, **kwargs):
        return [_gai("::1", socket.AF_INET6)]

    with patch.object(socket, "getaddrinfo", fake_getaddrinfo):
        with pinned_dns_resolution():
            assert socket.getaddrinfo is not fake_getaddrinfo  # guard installed
            with pytest.raises(ValueError, match="restricted IP address"):
                socket.getaddrinfo("evil.example", 443)
        # The guard is removed once the context exits.
        assert socket.getaddrinfo is fake_getaddrinfo


def test_configured_headers_scoped_to_source_origin():
    """Custom headers go only to the allowlisted origin, never to another host."""
    loader = ImageResourceLoader(
        enable_remote_fetch=True,
        headers={"Authorization": "Bearer secret", "X-API-Key": "k"},
        header_origin=_ORIGIN_A,
    )
    captured: dict[str, dict] = {}

    def fake_get(session, url, **kwargs):
        captured[url] = kwargs.get("headers", {})
        return _remote_response()

    with patch.object(requests.Session, "get", fake_get):
        loader.load_image_data(f"{_ORIGIN_A}/same-origin.png", None)
        loader.load_image_data(f"{_ORIGIN_B}/other-origin.png", None)

    same = captured[f"{_ORIGIN_A}/same-origin.png"]
    other = captured[f"{_ORIGIN_B}/other-origin.png"]
    assert same["Authorization"] == "Bearer secret"
    assert same["X-API-Key"] == "k"
    assert "Authorization" not in other
    assert "X-API-Key" not in other
    assert "Range" in other  # size cap is always sent


def test_configured_headers_dropped_on_cross_origin_redirect():
    """A redirect to a different origin must not carry the configured headers."""
    loader = ImageResourceLoader(
        enable_remote_fetch=True,
        headers={"Authorization": "Bearer secret"},
        header_origin=_ORIGIN_A,
    )
    calls: list[tuple[str, dict]] = []

    def fake_get(session, url, **kwargs):
        calls.append((url, kwargs.get("headers", {})))
        if len(calls) == 1:
            return _remote_response(redirect_to=f"{_ORIGIN_B}/final.png")
        return _remote_response()

    with patch.object(requests.Session, "get", fake_get):
        loader.load_image_data(f"{_ORIGIN_A}/start.png", None)

    assert calls[0][0] == f"{_ORIGIN_A}/start.png"
    assert calls[0][1].get("Authorization") == "Bearer secret"
    assert calls[1][0] == f"{_ORIGIN_B}/final.png"
    assert "Authorization" not in calls[1][1]


def test_headers_not_sent_when_no_source_origin():
    """With no allowlisted origin (e.g. local source), headers are withheld."""
    loader = ImageResourceLoader(
        enable_remote_fetch=True,
        headers={"Authorization": "Bearer secret"},
        header_origin=None,
    )
    captured: dict[str, dict] = {}

    def fake_get(session, url, **kwargs):
        captured[url] = kwargs.get("headers", {})
        return _remote_response()

    with patch.object(requests.Session, "get", fake_get):
        loader.load_image_data(f"{_ORIGIN_A}/img.png", None)

    assert "Authorization" not in captured[f"{_ORIGIN_A}/img.png"]
