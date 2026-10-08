# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression test for the scipy-free import of DocumentConverter (#4447).

``docling/models/base_ocr_model.py`` imported ``scipy.ndimage`` at module
level, so a ``docling-slim[format-*]`` install without the ``convert-core``
extra failed with ``ModuleNotFoundError: No module named 'scipy'`` the
moment ``DocumentConverter`` was imported — even for formats (like HTML)
that never run OCR. The fix defers the import into the method that uses it,
matching the reviewer-approved pattern of #4285/#4286.

The test blocks ``scipy`` in a fresh subprocess (``sys.modules`` poisoning
works whether or not scipy is installed in the test env) and asserts the
import chain completes.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_document_converter_imports_without_scipy() -> None:
    code = (
        "import sys; sys.modules['scipy'] = None; "
        "import docling.document_converter as dc; "
        "print('IMPORT-OK')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        "DocumentConverter import failed without scipy:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "IMPORT-OK" in result.stdout
