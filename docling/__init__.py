# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling: parse documents and convert them to a unified representation."""

import importlib.metadata
import os

# Databricks sets this to "0" to disable FIPS mode. OpenCV wheels bundle
# OpenSSL builds that treat any value as enabling FIPS and abort on import.
# Remove only the explicit opt-out, before any native dependencies load.
# Preserve explicit FIPS requests and all other values.
if os.environ.get("OPENSSL_FORCE_FIPS_MODE") == "0":
    os.environ.pop("OPENSSL_FORCE_FIPS_MODE")

__all__ = ["__version__"]


def _resolve_version() -> str:
    """Return the installed Docling version, or ``"unknown"``.

    ``docling`` is a meta-package that is absent on slim installs, where only
    ``docling-slim`` is present. This mirrors the fallback that ``docling
    --version`` applies to :class:`docling.datamodel.document.DoclingVersion`.
    """
    for name in ("docling", "docling-slim"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


__version__: str = _resolve_version()
