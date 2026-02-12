from __future__ import annotations

from typing import Collection, Iterable

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Status codes worth retrying because they are transient or throttling related.
_DEFAULT_STATUS_FORCELIST = (408, 425, 429, 500, 502, 503, 504)

# Methods that are safe or idempotent enough for retries in our usage.
_DEFAULT_ALLOWED_METHODS = frozenset(
    ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
)


def _build_retry(
    *,
    total: int = 5,
    backoff_factor: float = 0.2,
    status_forcelist: Collection[int] = _DEFAULT_STATUS_FORCELIST,
    allowed_methods: Iterable[str] | None = _DEFAULT_ALLOWED_METHODS,
) -> Retry:
    return Retry(
        total=total,
        read=total,
        connect=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(allowed_methods) if allowed_methods else None,
        raise_on_status=False,
    )


def create_retry_session(
    *,
    total: int = 5,
    backoff_factor: float = 0.2,
    status_forcelist: Collection[int] = _DEFAULT_STATUS_FORCELIST,
    allowed_methods: Iterable[str] | None = _DEFAULT_ALLOWED_METHODS,
) -> Session:
    """Return a requests Session configured with retry/backoff handling."""
    session = requests.Session()
    retry = _build_retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_DEFAULT_SESSION: Session | None = None


def get_retry_session() -> Session:
    """Return the lazily-created default retry-enabled Session."""
    global _DEFAULT_SESSION
    if _DEFAULT_SESSION is None:
        _DEFAULT_SESSION = create_retry_session()
    return _DEFAULT_SESSION


def request_with_retry(
    method: str,
    url: str,
    *,
    session: Session | None = None,
    **kwargs,
) -> Response:
    """Perform an HTTP request using a retry-enabled Session."""
    sess = session or get_retry_session()
    return sess.request(method=method, url=url, **kwargs)
