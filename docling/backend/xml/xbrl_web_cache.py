# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Arelle web cache that applies the fetch options of the XBRL backend.

Arelle resolves every document referenced by an XBRL instance (schemas,
linkbases, imported schemas) through `WebCache.getfilename`, which returns a
local path for local references and downloads http(s) references with the
`urllib` opener of the web cache. This module checks both:

- local paths and `file:` URLs are allowed only when local fetching is enabled,
  or when they are inside the directories prepared by the backend (the copy of
  the instance and of the taxonomy folder);
- other schemes than http(s) are refused;
- http(s) requests, including every redirect, connect only to public, globally
  routable addresses, with the same checks as the image resource loader.
"""

from __future__ import annotations

import http.client
import os
import socket
import ssl
import urllib.request
from collections.abc import Callable, Sequence
from email.message import Message
from pathlib import Path
from typing import IO, Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit

import certifi
from arelle.Cntlr import Cntlr  # type: ignore
from arelle.WebCache import ProxyTuple, WebCache  # type: ignore

from docling.backend.utils.image_resource_loader import (
    resolve_public_addresses,
    validate_url_safety,
)

_REMOTE_SCHEMES: tuple[str, ...] = ("http", "https")
# The default urllib handlers have the order 500. A lower order makes the
# handlers below open every http(s) request and follow every redirect of the
# Arelle opener.
_HANDLER_ORDER = 400

_RefusalCallback = Callable[[str, str], None]


class _AddressNotAllowed(OSError):
    """The host of a request does not resolve to public addresses only.

    It is not a `ConnectionError`, so that Arelle does not retry the request.
    """


def _connect_public(
    address: tuple[str, int],
    timeout: Any,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Open a socket to one of the validated public addresses of a host."""
    host, port = address
    try:
        ips = resolve_public_addresses(host)
    except ValueError as exc:
        raise _AddressNotAllowed(str(exc)) from exc
    error: OSError | None = None
    for ip in ips:
        try:
            return socket.create_connection((str(ip), port), timeout, source_address)
        except OSError as exc:
            error = exc
    assert error is not None
    raise error


# `http.client` opens the socket of a connection through its
# `_create_connection` attribute. Replacing it keeps the request, the `Host`
# header and, for https, the SNI and the certificate check on the original
# hostname, while the socket goes to an address that was checked.
class _PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_public


def _is_proxied(req: urllib.request.Request) -> bool:
    # `urllib.request.ProxyHandler` replaces the host of the request with the
    # host of the proxy.
    return req.host != unquote(urlsplit(req.full_url).netloc)


def _open_checked(
    handler: urllib.request.AbstractHTTPHandler,
    req: urllib.request.Request,
    public_class: type[http.client.HTTPConnection],
    proxied_class: type[http.client.HTTPConnection],
    on_refused: _RefusalCallback,
    **conn_args: Any,
) -> http.client.HTTPResponse:
    try:
        if _is_proxied(req):
            # The proxy connects to the target, so only the target is checked.
            try:
                validate_url_safety(req.full_url)
            except ValueError as exc:
                raise URLError(_AddressNotAllowed(str(exc))) from exc
            return handler.do_open(proxied_class, req, **conn_args)
        return handler.do_open(public_class, req, **conn_args)
    except URLError as exc:
        if isinstance(exc.reason, _AddressNotAllowed):
            on_refused(req.full_url, str(exc.reason))
        raise


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    handler_order = _HANDLER_ORDER

    def __init__(self, on_refused: _RefusalCallback) -> None:
        super().__init__()
        self._on_refused = on_refused

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return _open_checked(
            self,
            req,
            _PublicHTTPConnection,
            http.client.HTTPConnection,
            self._on_refused,
        )


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    handler_order = _HANDLER_ORDER

    def __init__(self, on_refused: _RefusalCallback) -> None:
        # Same CA bundle lookup as the image resource loader.
        self._ssl_context = ssl.create_default_context(
            cafile=os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
            or certifi.where()
        )
        super().__init__(context=self._ssl_context)
        self._on_refused = on_refused

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return _open_checked(
            self,
            req,
            _PublicHTTPSConnection,
            http.client.HTTPSConnection,
            self._on_refused,
            context=self._ssl_context,
        )


class _HTTPOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects to http(s) URLs only.

    The new request is opened by the handlers above, which check its address.
    """

    handler_order = _HANDLER_ORDER

    def __init__(self, on_refused: _RefusalCallback) -> None:
        super().__init__()
        self._on_refused = on_refused

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urlsplit(newurl).scheme.lower() not in _REMOTE_SCHEMES:
            reason = f"Redirect to a URL that is not http(s): {newurl}"
            self._on_refused(req.full_url, reason)
            raise HTTPError(req.full_url, code, reason, headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class FetchPolicyWebCache(WebCache):  # type: ignore[misc]
    """Arelle web cache restricted by the fetch options of the XBRL backend.

    Args:
        cntlr: The Arelle controller that uses this web cache.
        enable_local_fetch: Allow local paths and `file:` URLs anywhere.
        enable_remote_fetch: Allow http(s) downloads. When disabled, the cache
            works offline and http(s) references resolve from the cache only.
        local_roots: Directories whose files are always allowed, for example
            the instance and the taxonomy packages given to the backend.
    """

    def __init__(
        self,
        cntlr: Cntlr,
        *,
        enable_local_fetch: bool,
        enable_remote_fetch: bool,
        local_roots: Sequence[Path],
    ) -> None:
        super().__init__(cntlr, ProxyTuple.coerce(cntlr.config.get("proxySettings")))
        self.enable_local_fetch = enable_local_fetch
        self.local_roots: tuple[Path, ...] = tuple(
            root.resolve() for root in local_roots
        )
        self.workOffline = not enable_remote_fetch
        self.refused: list[str] = []
        self._checked_opener: urllib.request.OpenerDirector | None = None

    def _on_refused(self, url: str, reason: str) -> None:
        self.refused.append(f"{url} ({reason})")

    def _install_checked_handlers(self) -> None:
        # Arelle may build the opener again, e.g. when the proxies change, so
        # the handlers are added to every new opener.
        opener: urllib.request.OpenerDirector = self.opener
        if opener is self._checked_opener:
            return
        opener.add_handler(_PublicHTTPHandler(self._on_refused))
        opener.add_handler(_PublicHTTPSHandler(self._on_refused))
        opener.add_handler(_HTTPOnlyRedirectHandler(self._on_refused))
        self._checked_opener = opener

    def _refusal_reason(self, url: str, normalized: str) -> str | None:
        for target in (url, normalized):
            scheme, sep, _ = target.partition("://")
            if sep and scheme in _REMOTE_SCHEMES:
                # Downloads are checked by the handlers of the opener.
                return None
            if sep and scheme != "file":
                return f"URL scheme not allowed: {scheme}"
        if self.enable_local_fetch:
            return None
        local = normalized.removeprefix("file://").removeprefix("file:\\")
        try:
            path = Path(local).resolve()
        except (OSError, ValueError):
            return "Invalid local path"
        if any(path.is_relative_to(root) for root in self.local_roots):
            return None
        return "Local file access requires enable_local_fetch=True"

    def getfilename(
        self, url: str | None, base: str | None = None, *args: Any, **kwargs: Any
    ) -> str | None:
        if url is not None:
            reason = self._refusal_reason(url, self.normalizeUrl(url, base) or url)
            if reason is not None:
                self._on_refused(url, reason)
                return None
        self._install_checked_handlers()
        return super().getfilename(url, base, *args, **kwargs)
