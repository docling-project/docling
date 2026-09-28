# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Shared image-resource loading for declarative backends.

Turns an image source (a URI, a local file, or a remote URL) into a DoclingDocument
`ImageRef`, enforcing the relevant safety limits: a path-traversal guard when resolving
relative paths, the `enable_local_fetch` / `enable_remote_fetch` toggles, and the
base64 and remote download size caps.
"""

import base64
import contextlib
import ipaddress
import logging
import os
import re
import socket
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urljoin, urlparse, urlsplit

import certifi
import requests
import urllib3
from docling_core.types.doc.document import ImageRef
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from requests.utils import get_environ_proxies, select_proxy

from docling.exceptions import OperationNotAllowed

_log = logging.getLogger(__name__)


_IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

# NAT64 well-known prefix (RFC 6052): 64:ff9b::/96 embeds an IPv4 address in its
# low 32 bits.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
# Ranges refused explicitly, independently of the ``ipaddress`` tables shipped
# with the running Python version. 64:ff9b:1::/48 is the local-use NAT64 prefix
# (RFC 8215), which translates to addresses of the local network.
_DENIED_NETWORKS = (ipaddress.ip_network("64:ff9b:1::/48"),)


def _ip_is_restricted(ip: _IPAddress) -> bool:
    """Return True if ``ip`` is not a globally routable, public address."""
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or any(ip in network for network in _DENIED_NETWORKS)
    )


def _embedded_ipv4(ip: _IPAddress) -> Optional[ipaddress.IPv4Address]:
    """Extract an IPv4 address embedded in an IPv6 address, if any.

    Covers IPv4-mapped (``::ffff:a.b.c.d``), 6to4 (``2002::/16``) and NAT64
    (``64:ff9b::/96``) forms.
    """
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip in _NAT64_PREFIX:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def _validate_ip(ip: _IPAddress) -> None:
    """Reject an address that is (or embeds) a non-public IP.

    Raises:
        ValueError: If ``ip`` -- or an IPv4 address embedded within it -- is a
            private, loopback, link-local, reserved, multicast or unspecified
            address, or is otherwise not globally routable.
    """
    embedded = _embedded_ipv4(ip)
    if _ip_is_restricted(ip) or (embedded is not None and _ip_is_restricted(embedded)):
        raise ValueError(f"Access to restricted IP address not allowed: {ip}")


def resolve_public_addresses(host: str) -> list[_IPAddress]:
    """Resolve ``host`` and return its addresses if all of them are public.

    ``host`` may be an IP literal (IPv6 with or without brackets) or a hostname,
    which is resolved to every IPv4 and IPv6 address it maps to. All addresses
    are validated, since a connection may use any of them.

    Raises:
        ValueError: If the host cannot be resolved, or any of its addresses is
            not a public, globally routable IP.
    """
    host = host.strip("[]")
    try:
        ips: list[_IPAddress] = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except (socket.gaierror, socket.herror, UnicodeError) as e:
            raise ValueError(f"Cannot resolve hostname: {host}") from e
        ips = []
        for info in infos:
            with contextlib.suppress(ValueError):
                ip = ipaddress.ip_address(info[4][0])
                if ip not in ips:
                    ips.append(ip)
        if not ips:
            raise ValueError(f"Cannot resolve hostname: {host}")

    for ip in ips:
        _validate_ip(ip)
    return ips


def validate_url_safety(url: str) -> None:
    """Reject URLs whose host is not reachable on public addresses only.

    Every address the URL's host resolves to (IPv4 and IPv6) must be globally
    routable. Private, loopback, link-local, reserved, multicast, and unspecified
    addresses are refused, as are IPv4 addresses embedded in IPv6 addresses
    (IPv4-mapped, 6to4 and NAT64 forms) and the local-use NAT64 prefix.

    Args:
        url: The URL whose host is validated.

    Raises:
        ValueError: If the URL has no hostname, the hostname cannot be
            resolved, or any resolved address is a restricted (non-global) IP.
    """
    hostname = urlparse(url).hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname")
    resolve_public_addresses(hostname)


_DEFAULT_PORTS = {"http": 80, "https": 443}
_TIMEOUT = urllib3.Timeout(connect=5, read=30)


@contextlib.contextmanager
def _open_direct(url: str, headers: dict[str, str]) -> Iterator[urllib3.HTTPResponse]:
    """Send a GET for ``url`` to one of its host's validated addresses.

    The host is resolved once and the connection pool is bound to a validated
    address, trying the next one if the connection cannot be established. The
    original hostname is sent in the ``Host`` header and, for https, used for
    SNI and certificate verification.
    """
    parts = urlsplit(url)
    hostname = parts.hostname
    if not hostname:
        raise ValueError("URL must contain a valid hostname")
    default_port = _DEFAULT_PORTS[parts.scheme]
    port = parts.port or default_port
    host_header = f"[{hostname}]" if ":" in hostname else hostname
    if port != default_port:
        host_header += f":{port}"
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")

    errors: list[urllib3.exceptions.HTTPError] = []
    for ip in resolve_public_addresses(hostname):
        pool: urllib3.HTTPConnectionPool
        if parts.scheme == "https":
            pool = urllib3.HTTPSConnectionPool(
                str(ip),
                port,
                timeout=_TIMEOUT,
                retries=False,
                cert_reqs="CERT_REQUIRED",
                # Same CA bundle lookup as requests.
                ca_certs=os.environ.get("REQUESTS_CA_BUNDLE")
                or os.environ.get("CURL_CA_BUNDLE")
                or certifi.where(),
                server_hostname=hostname,
                assert_hostname=hostname,
            )
        else:
            pool = urllib3.HTTPConnectionPool(
                str(ip), port, timeout=_TIMEOUT, retries=False
            )
        with pool:
            try:
                response = pool.urlopen(
                    "GET",
                    target,
                    headers={"Host": host_header, **headers},
                    redirect=False,
                    preload_content=False,
                )
            except (
                urllib3.exceptions.NewConnectionError,
                urllib3.exceptions.ConnectTimeoutError,
            ) as e:
                errors.append(e)
                continue
            with response:
                yield response
            return
    raise errors[-1]


@contextlib.contextmanager
def _open_with_proxy(
    url: str, headers: dict[str, str]
) -> Iterator[urllib3.HTTPResponse]:
    """Send a GET for ``url`` through the proxy configured in the environment."""
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(5, 30),
        allow_redirects=False,
    ) as response:
        yield response.raw


def url_origin(url: str) -> Optional[tuple[str, str, Optional[int]]]:
    """Return the ``(scheme, host, port)`` origin of a URL, or None.

    Ports default to the well-known port for the scheme so that, e.g.,
    ``http://h/`` and ``http://h:80/`` share an origin. Returns None when the
    value has no scheme/host (e.g. a local path), which callers treat as "no
    allowlisted origin".
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        return None
    default_ports = {"http": 80, "https": 443, "ftp": 21}
    port = parsed.port or default_ports.get(parsed.scheme.lower())
    return (parsed.scheme.lower(), parsed.hostname.lower(), port)


@dataclass(frozen=True)
class RemoteResource:
    """A remote resource downloaded by :meth:`ImageResourceLoader.fetch_remote`."""

    status_code: int
    headers: dict[str, str]
    content: bytes


class ImageResourceLoader:
    """Resolve and load image resources for declarative document backends.

    The `base_path` against which relative locations are resolved is supplied
    per call rather than stored, so a backend that mutates its base path between
    calls always uses the current value.
    """

    def __init__(
        self,
        *,
        enable_local_fetch: bool = False,
        enable_remote_fetch: bool = False,
        max_image_data_base64_bytes: int = 20 * 1024 * 1024,
        max_remote_image_bytes: int = 20 * 1024 * 1024,
        max_redirects: int = 5,
        headers: Optional[dict[str, str]] = None,
        header_origins: Iterable[str] = (),
    ) -> None:
        self.enable_local_fetch = enable_local_fetch
        self.enable_remote_fetch = enable_remote_fetch
        self.max_image_data_base64_bytes = max_image_data_base64_bytes
        self.max_remote_image_bytes = max_remote_image_bytes
        self.max_redirects = max_redirects
        self.headers = headers
        # Only requests to these origins carry the configured `headers`.
        self.header_origins = {
            origin
            for origin in (url_origin(value) for value in header_origins)
            if origin is not None
        }

    @staticmethod
    def is_remote_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https", "ftp", "s3", "gs"}

    @staticmethod
    def is_local_path(value: str) -> bool:
        """Check if value is a local filesystem path (not a URI)."""
        parsed = urlparse(value)
        return not parsed.netloc and (
            not parsed.scheme
            or (len(parsed.scheme) == 1 and parsed.scheme.isalpha())  # Windows case
        )

    @staticmethod
    def is_absolute_path(loc: str) -> bool:
        return Path(loc).is_absolute() or (  # Windows-specific absolute paths:
            len((parsed_loc := urlparse(loc)).scheme) == 1
            and parsed_loc.scheme.isalpha()
            and not parsed_loc.netloc
        )

    def resolve_relative_path(self, loc: str, base_path: Optional[str]) -> str:
        loc = loc.strip()

        # Strip file:// prefix for validation as local path
        if loc.startswith(file_prefix := "file://"):
            loc = loc[len(file_prefix) :]

        abs_loc = loc

        if base_path:
            if loc.startswith("//"):
                abs_loc = "https:" + loc
            elif not loc.startswith(("http://", "https://", "data:", "#")):
                if ImageResourceLoader.is_remote_url(base_path):
                    abs_loc = urljoin(base_path, loc)
                elif ImageResourceLoader.is_local_path(base_path):
                    if ImageResourceLoader.is_absolute_path(loc):
                        raise ValueError(
                            f"Absolute paths are not allowed with local base_path: '{loc}'"
                        )

                    base_dir = Path(base_path).parent.resolve()
                    resolved_path = (base_dir / loc).resolve()

                    if not resolved_path.is_relative_to(base_dir):
                        raise ValueError(
                            f"Path traversal blocked: '{loc}' resolves outside base directory"
                        )
                    abs_loc = str(resolved_path)
                else:
                    raise ValueError(f"Invalid base_path format: '{base_path}'")

        _log.debug(f"Resolved location {loc} to {abs_loc}")
        return abs_loc

    def create_image_ref(
        self, src_url: str, base_path: Optional[str]
    ) -> Optional[ImageRef]:
        try:
            img_data = self.load_image_data(src_url, base_path)
            if img_data:
                img = Image.open(BytesIO(img_data))
                return ImageRef.from_pil(img, dpi=int(img.info.get("dpi", (72,))[0]))
        except (
            requests.RequestException,
            urllib3.exceptions.HTTPError,
            ValidationError,
            UnidentifiedImageError,
            OperationNotAllowed,
            TypeError,
            ValueError,
        ) as e:
            warnings.warn(f"Could not process an image from {src_url}: {e}")

        return None

    def load_image_ref(self, src: str, base_path: Optional[str]) -> Optional[ImageRef]:
        """Resolve `src` against `base_path` and decode it into an ImageRef."""
        return self.create_image_ref(
            self.resolve_relative_path(src, base_path), base_path
        )

    def fetch_remote(self, url: str) -> RemoteResource:
        """Download a remote http(s) resource, following redirects.

        Every hop is resolved once, all its addresses are validated with
        :func:`resolve_public_addresses`, and the connection is opened to one of
        the validated addresses (see :func:`_open_direct`). Configured headers
        are sent only to the allowed origins, hop by hop. The download is capped
        at ``max_remote_image_bytes``.

        When a proxy is configured for the URL (e.g. with the ``HTTP_PROXY`` /
        ``HTTPS_PROXY`` environment variables), the request goes through the
        proxy, which then connects to the destination and is responsible for
        filtering it.

        Raises:
            OperationNotAllowed: If remote fetch is disabled.
            ValueError: If a hop is not an http(s) URL on a public address, there
                are too many redirects, the server returns an error status, or
                the resource exceeds the size limit.
            requests.RequestException, urllib3.exceptions.HTTPError: On
                connection or transfer errors.
        """
        if not self.enable_remote_fetch:
            raise OperationNotAllowed(
                "Fetching remote resources is only allowed when set explicitly. "
                "Set options.enable_remote_fetch=True."
            )

        for _ in range(self.max_redirects + 1):
            if urlsplit(url).scheme not in _DEFAULT_PORTS:
                raise ValueError(f"Only http(s) URLs can be fetched: {url}")
            headers = self._request_headers(url)
            if select_proxy(url, get_environ_proxies(url)):
                opened = _open_with_proxy(url, headers)
            else:
                opened = _open_direct(url, headers)
            with opened as response:
                location = response.get_redirect_location()
                if location:
                    url = urljoin(url, location)
                    continue
                if response.status >= 400:
                    raise ValueError(f"HTTP status {response.status} for {url}")
                return RemoteResource(
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=self._read_limited(response),
                )
        raise ValueError("Exceeded maximum number of redirects")

    def _request_headers(self, url: str) -> dict[str, str]:
        """Return the configured headers if ``url`` is on an allowed origin."""
        if self.headers and url_origin(url) in self.header_origins:
            return dict(self.headers)
        return {}

    def _read_limited(self, response: urllib3.HTTPResponse) -> bytes:
        max_size = self.max_remote_image_bytes
        content_length = response.headers.get("content-length")
        if (
            content_length
            and content_length.isdigit()
            and int(content_length) > max_size
        ):
            raise ValueError(f"Resource size exceeds limit: {content_length} bytes")

        chunks = []
        total_size = 0
        for chunk in response.stream(8192, decode_content=True):
            total_size += len(chunk)
            if total_size > max_size:
                raise ValueError("Downloaded data exceeds size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def load_image_data(
        self, src_loc: str, base_path: Optional[str]
    ) -> Optional[bytes]:
        if src_loc.lower().endswith(".svg"):
            _log.debug(f"Skipping SVG file: {src_loc}")
            return None

        if ImageResourceLoader.is_remote_url(src_loc):
            return self.fetch_remote(src_loc).content
        elif src_loc.startswith("data:"):
            encoded_data = re.sub(r"^data:image/.+;base64,", "", src_loc)
            decoded_data = base64.b64decode(encoded_data)

            if len(decoded_data) > self.max_image_data_base64_bytes:
                raise ValueError(
                    f"Decoded image exceeds size limit of {self.max_image_data_base64_bytes} bytes."
                )

            return decoded_data

        if not self.enable_local_fetch:
            raise OperationNotAllowed(
                "Fetching local resources is only allowed when set explicitly. "
                "Set options.enable_local_fetch=True."
            )

        # Require base_path for directory confinement (validation done in resolve_relative_path)
        if not base_path:
            raise OperationNotAllowed(
                f"Local file access requires base_path for directory confinement: '{src_loc}'"
            )

        if os.path.isfile(src_loc) and os.access(src_loc, os.R_OK):
            with open(src_loc, "rb") as f:
                return f.read()
        else:
            raise ValueError("File does not exist or it is not readable.")
