# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""OCR registration must survive Databricks' explicit FIPS opt-out."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fips_mode", [None, "0"])
def test_ocr_registration_with_fips_environment(fips_mode: str | None) -> None:
    # Use a fresh process: a native SIGABRT cannot be caught in Python, and an
    # OpenCV import cached by another test would hide the initialization failure.
    env = os.environ.copy()
    env.pop("OPENSSL_FORCE_FIPS_MODE", None)
    if fips_mode is not None:
        env["OPENSSL_FORCE_FIPS_MODE"] = fips_mode
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
from docling.models.plugins.defaults import ocr_engines
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import OcrAutoOptions

engines = ocr_engines()["ocr_engines"]
auto = next(engine for engine in engines if engine.__name__ == "OcrAutoModel")
model = auto(
    enabled=False,
    artifacts_path=None,
    options=OcrAutoOptions(),
    accelerator_options=AcceleratorOptions(),
)
assert model.enabled is False
assert os.environ.get("OPENSSL_FORCE_FIPS_MODE") is None
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fips_mode", ["1", "true", ""])
def test_package_import_preserves_fips_requests(fips_mode: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import docling; "
            "assert os.environ['OPENSSL_FORCE_FIPS_MODE'] == os.environ['EXPECTED_FIPS_MODE']",
        ],
        env={
            **os.environ,
            "OPENSSL_FORCE_FIPS_MODE": fips_mode,
            "EXPECTED_FIPS_MODE": fips_mode,
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
