# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for docling-slim installs without the ``convert-core`` extra.

``scipy`` and ``rtree`` ship only with ``convert-core``. ``DocumentConverter``
must still import without them, and the components that need them must raise
an ImportError that names the extra.

Each test blocks the modules in a fresh subprocess (``sys.modules`` poisoning
works whether or not they are installed in the test env).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_without(blocked_modules: tuple[str, ...], code: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.modules.update(dict.fromkeys({blocked_modules!r}))\n"
            + code,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Subprocess failed without {', '.join(blocked_modules)}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result.stdout


@pytest.mark.parametrize(
    "blocked_modules",
    [("scipy",), ("rtree",), ("scipy", "rtree")],
    ids=["scipy", "rtree", "scipy+rtree"],
)
def test_document_converter_imports_without_convert_core(
    blocked_modules: tuple[str, ...],
) -> None:
    stdout = _run_without(
        blocked_modules,
        "import docling.document_converter\nprint('IMPORT-OK')",
    )
    assert "IMPORT-OK" in stdout


def test_spatial_index_without_rtree_names_the_extra() -> None:
    stdout = _run_without(
        ("rtree",),
        "from docling.datamodel.spatial import BoundingBoxSpatialIndex\n"
        "try:\n"
        "    BoundingBoxSpatialIndex()\n"
        "except ImportError as e:\n"
        "    print(e)\n",
    )
    assert "docling-slim[convert-core]" in stdout
