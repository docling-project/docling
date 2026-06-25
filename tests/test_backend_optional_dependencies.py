"""Regression tests for optional format-backend dependencies.

``DocumentConverter`` imports every format backend eagerly (see
``docling/document_converter.py``), so a backend that imports a third-party
package at module load breaks ``import docling`` for any install that omits the
matching extra, such as the slim packages. These tests pin the contract that a
missing optional dependency stays dormant until its backend is actually used.

See https://github.com/docling-project/docling/issues/3613.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_EML_SAMPLE = Path(__file__).parent / "data" / "email" / "sources" / "eml_simple.eml"


def _run_with_blocked_module(
    blocked_module: str, body: str
) -> subprocess.CompletedProcess:
    # The optional packages are installed in the dev/CI environment, so to
    # reproduce a slim install we block one inside a fresh interpreter. Mapping
    # the name to None is what the import system sees for an uninstalled package:
    # `import <name>` then raises ImportError. Doing it before docling is
    # imported forces the backend modules to be loaded cold without the package.
    script = f"import sys\nsys.modules[{blocked_module!r}] = None\n{body}"
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "blocked_module",
    ["mailparser", "marko"],
)
def test_converter_constructs_without_optional_backend_dependency(
    blocked_module: str,
) -> None:
    # Constructing a default DocumentConverter resolves a FormatOption for every
    # input format, so it fails on the eager-import regression and succeeds once
    # the backend defers its optional import.
    result = _run_with_blocked_module(
        blocked_module,
        "from docling.document_converter import DocumentConverter\nDocumentConverter()\n",
    )
    assert result.returncode == 0, result.stderr


def test_email_backend_reports_missing_dependency_with_install_hint() -> None:
    # Using a backend whose optional dependency is absent must fail with an
    # actionable ImportError that names the extra to install, not a NameError or
    # an error mislabeled as a corrupt-document failure.
    body = (
        "from pathlib import Path\n"
        "from docling.backend.email_backend import EmailDocumentBackend\n"
        "from docling.datamodel.base_models import InputFormat\n"
        "from docling.datamodel.document import InputDocument\n"
        f"path = Path({str(_EML_SAMPLE)!r})\n"
        "try:\n"
        "    InputDocument(\n"
        "        path_or_stream=path, format=InputFormat.EMAIL, backend=EmailDocumentBackend\n"
        "    )\n"
        "except ImportError as exc:\n"
        "    assert 'format-email' in str(exc), str(exc)\n"
        "    print('actionable-error-raised')\n"
        "else:\n"
        "    raise AssertionError('expected ImportError when mail-parser is missing')\n"
    )
    result = _run_with_blocked_module("mailparser", body)
    assert result.returncode == 0, result.stderr
    assert "actionable-error-raised" in result.stdout
