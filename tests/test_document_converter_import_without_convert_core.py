# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Regression test for importing DocumentConverter without convert-core (#4447).

``scipy`` and ``rtree`` ship only with the ``convert-core`` extra. Modules on
the ``DocumentConverter`` import chain (``base_ocr_model``,
``datamodel/spatial``, ``reading_order_rb``) imported them at module level, so
a ``docling-slim[format-*]`` install without ``convert-core`` failed with
``ModuleNotFoundError`` the moment ``DocumentConverter`` was imported — even
for formats (like HTML) that never run OCR or layout postprocessing.

The test blocks the modules in a fresh subprocess (``sys.modules`` poisoning
works whether or not they are installed in the test env) and asserts the
import chain completes.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "blocked_modules",
    [("scipy",), ("rtree",), ("scipy", "rtree")],
    ids=["scipy", "rtree", "scipy+rtree"],
)
def test_document_converter_imports_without_convert_core(
    blocked_modules: tuple[str, ...],
) -> None:
    code = (
        "import sys; "
        f"sys.modules.update(dict.fromkeys({blocked_modules!r})); "
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
        f"DocumentConverter import failed without {', '.join(blocked_modules)}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "IMPORT-OK" in result.stdout
