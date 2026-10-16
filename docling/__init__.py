# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling: parse documents and convert them to a unified representation."""

import importlib.metadata

__all__ = ["__version__"]


def _resolve_version() -> str:
    """Return the installed Docling version, or ``"unknown"``.

    ``docling`` is a meta-package that is absent on slim installs, where only
    ``docling-slim`` is present. ``docling --version`` prints this value.
    """
    for name in ("docling", "docling-slim"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


__version__: str = _resolve_version()
