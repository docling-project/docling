"""Helpers for KServe transport URL handling."""

from __future__ import annotations


def resolve_kserve_transport_base_url(*, url: str, transport: str) -> str:
    """Resolve runtime base URL for KServe transport clients.

    HTTP accepts either full http(s) URLs or plain host:port.
    gRPC accepts host:port, [ipv6]:port, dns:///host:port, or dns:///[ipv6]:port.
    """
    if transport == "http" and "://" not in url:
        return f"http://{url}"
    return url
