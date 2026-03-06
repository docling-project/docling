"""Helpers for KServe transport URL normalization."""

from __future__ import annotations

from typing import Any


def normalize_kserve_transport_url_data(data: Any) -> Any:
    """Normalize plain host:port URL inputs based on selected transport.

    Keeps full URLs unchanged. For plain host:port:
    - transport=http -> prefixes http://
    - transport=grpc (default) -> prefixes dns://
    """
    if not isinstance(data, dict):
        return data

    raw_url = data.get("url")
    if not isinstance(raw_url, str):
        return data

    if "://" in raw_url:
        return data

    normalized = dict(data)
    transport = normalized.get("transport", "grpc")
    if transport == "http":
        normalized["url"] = f"http://{raw_url}"
    else:
        normalized["url"] = f"dns://{raw_url}"
    return normalized
