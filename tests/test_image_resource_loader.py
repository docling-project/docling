"""Unit tests for the shared image-resource loader and its safety limits."""

import socket
from unittest.mock import Mock, patch

import pytest
import requests

from docling.backend.utils.image_resource_loader import (
    ImageResourceLoader,
    validate_url_safety,
)
from docling.exceptions import OperationNotAllowed

# A globally routable IP literal (example.com) used so the source URL passes the
# SSRF check without a DNS lookup, keeping these tests offline.
_GLOBAL_IP_URL = "http://93.184.216.34/image.png"


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
