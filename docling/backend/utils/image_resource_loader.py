"""Shared image-resource loading for declarative backends.

Turns an image source (a ``data:`` URI, a local file, or a remote URL) into a
Docling :class:`~docling_core.types.doc.document.ImageRef`, enforcing the
relevant safety limits: a path-traversal guard when resolving relative paths,
the ``enable_local_fetch`` / ``enable_remote_fetch`` toggles, and the base64 and
remote download size caps.

This logic originated in the HTML backend. It is factored out here so the HTML
and Markdown backends can share a single implementation instead of duplicating
it (or constructing one backend from inside the other).
"""

import base64
import ipaddress
import logging
import os
import re
import socket
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from docling_core.types.doc.common.reference import ImageRef
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from docling.exceptions import OperationNotAllowed

_log = logging.getLogger(__name__)


def validate_url_safety(url: str) -> None:
    """Reject URLs that resolve to a non-public IP address.

    Guards against SSRF by requiring the URL's host to resolve to a globally
    routable address. Private, loopback, link-local, reserved, multicast, and
    unspecified addresses are refused.

    Args:
        url: The URL whose host is validated.

    Raises:
        ValueError: If the URL has no hostname, the hostname cannot be
            resolved, or it resolves to a restricted (non-global) IP address.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise ValueError("URL must contain a valid hostname")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
        except (socket.gaierror, socket.herror) as e:
            raise ValueError(f"Cannot resolve hostname: {hostname}") from e

    if not (
        ip.is_global
        and not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    ):
        raise ValueError(f"Access to restricted IP address not allowed: {ip}")


class ImageResourceLoader:
    """Resolve and load image resources for declarative document backends.

    The ``base_path`` against which relative locations are resolved is supplied
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
    ) -> None:
        self.enable_local_fetch = enable_local_fetch
        self.enable_remote_fetch = enable_remote_fetch
        self.max_image_data_base64_bytes = max_image_data_base64_bytes
        self.max_remote_image_bytes = max_remote_image_bytes
        self.max_redirects = max_redirects
        self.headers = headers

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
            requests.HTTPError,
            ValidationError,
            UnidentifiedImageError,
            OperationNotAllowed,
            TypeError,
            ValueError,
        ) as e:
            warnings.warn(f"Could not process an image from {src_url}: {e}")

        return None

    def load_image_ref(self, src: str, base_path: Optional[str]) -> Optional[ImageRef]:
        """Resolve ``src`` against ``base_path`` and decode it into an ImageRef."""
        return self.create_image_ref(
            self.resolve_relative_path(src, base_path), base_path
        )

    def load_image_data(
        self, src_loc: str, base_path: Optional[str]
    ) -> Optional[bytes]:
        if src_loc.lower().endswith(".svg"):
            _log.debug(f"Skipping SVG file: {src_loc}")
            return None

        if ImageResourceLoader.is_remote_url(src_loc):
            if not self.enable_remote_fetch:
                raise OperationNotAllowed(
                    "Fetching remote resources is only allowed when set explicitly. "
                    "Set options.enable_remote_fetch=True."
                )

            validate_url_safety(src_loc)

            max_size = self.max_remote_image_bytes
            headers = {"Range": f"bytes=0-{max_size - 1}"}

            # Merge custom headers from options if provided
            if self.headers:
                headers.update(self.headers)

            # Create session with redirect limit
            session = requests.Session()
            session.max_redirects = self.max_redirects

            # Hook to validate each redirect target
            def _check_redirect_safety(response, *args, **kwargs):
                """Validate each redirect target before following it."""
                if response.is_redirect or response.is_permanent_redirect:
                    redirect_url = response.headers.get("location")
                    if redirect_url:
                        # Handle relative redirects
                        if not redirect_url.startswith(("http://", "https://")):
                            redirect_url = urljoin(response.url, redirect_url)

                        # Validate the redirect target
                        validate_url_safety(redirect_url)

            session.hooks["response"].append(_check_redirect_safety)

            response = session.get(
                src_loc, stream=True, headers=headers, timeout=(5, 30)
            )
            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_size:
                raise ValueError(f"Resource size exceeds limit: {content_length} bytes")

            chunks = []
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > max_size:
                        raise ValueError("Downloaded data exceeds size limit")
                    chunks.append(chunk)

            return b"".join(chunks)
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
