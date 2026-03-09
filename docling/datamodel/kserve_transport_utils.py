"""Helpers for KServe transport URL handling."""

from __future__ import annotations


def resolve_kserve_transport_base_url(*, url: str, transport: str) -> str:
    """Resolve runtime base URL for KServe transport clients.

    HTTP accepts either full http(s) URLs or plain host:port.
    gRPC expects plain host:port only and is passed through as-is.
    """
    if transport == "http" and "://" not in url:
        return f"http://{url}"
    return url
