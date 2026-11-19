# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Docling must neutralize Databricks' explicit FIPS opt-out before OpenCV loads."""

import importlib
import os
import subprocess
import sys

import pytest

import docling


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", None), ("1", "1"), ("true", "true"), ("", "")],
)
def test_package_import_normalizes_only_fips_opt_out(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str | None
) -> None:
    monkeypatch.setenv("OPENSSL_FORCE_FIPS_MODE", value)
    importlib.reload(docling)
    assert os.environ.get("OPENSSL_FORCE_FIPS_MODE") == expected


def test_opencv_loads_after_docling_with_fips_opt_out() -> None:
    pytest.importorskip("cv2")
    # A fresh process is required: OpenCV is already loaded in this process, and
    # the affected wheels abort with SIGABRT, which Python cannot catch.
    result = subprocess.run(
        [sys.executable, "-c", "import docling, cv2"],
        env={**os.environ, "OPENSSL_FORCE_FIPS_MODE": "0"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
