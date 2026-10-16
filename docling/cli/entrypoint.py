# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Console entry point of the ``docling`` command.

Importing :mod:`docling.cli.main` loads the full conversion stack, which takes
several seconds. This module answers ``docling --version`` without that import
and hands every other invocation to the Typer app.
"""

import platform
import sys

import docling
from docling.utils.utils import safe_version


def print_version(*, full: bool = False) -> None:
    """Print the Docling version, and with ``full`` also its components."""
    print(f"Docling version: {docling.__version__}")
    if not full:
        return
    print(f"Docling Core version: {safe_version('docling-core')}")
    print(f"Docling IBM Models version: {safe_version('docling-ibm-models')}")
    print(f"Docling Parse version: {safe_version('docling-parse')}")
    print(f"Python: {sys.implementation.cache_tag} ({platform.python_version()})")
    print(f"Platform: {platform.platform()}")


def main() -> None:
    """Run the ``docling`` command."""
    args = sys.argv[1:]
    if args[:1] == ["convert"]:
        args = args[1:]
    if "--version" in args and set(args) <= {"--version", "--full"}:
        print_version(full="--full" in args)
        return

    from docling.cli.main import app

    app()
