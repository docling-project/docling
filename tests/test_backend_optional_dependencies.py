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
_MD_SAMPLE = Path(__file__).parent / "data" / "md" / "sources" / "mixed.md"
_DOCX_SAMPLE = Path(__file__).parent / "data" / "docx" / "sources" / "word_tables.docx"
_PPTX_SAMPLE = (
    Path(__file__).parent / "data" / "pptx" / "sources" / "powerpoint_sample.pptx"
)
_XLSX_SAMPLE = Path(__file__).parent / "data" / "xlsx" / "sources" / "xlsx_01.xlsx"
_TEX_SAMPLE = (
    Path(__file__).parent / "data" / "latex" / "sources" / "1706.03762" / "main.tex"
)


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
    ["mailparser", "marko", "docx", "pptx", "openpyxl", "pylatexenc"],
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


@pytest.mark.parametrize(
    ("blocked_module", "backend_import", "backend_class", "sample", "fmt", "extra"),
    [
        (
            "mailparser",
            "docling.backend.email_backend",
            "EmailDocumentBackend",
            _EML_SAMPLE,
            "EMAIL",
            "format-email",
        ),
        (
            "marko",
            "docling.backend.md_backend",
            "MarkdownDocumentBackend",
            _MD_SAMPLE,
            "MD",
            "format-markdown",
        ),
        (
            "docx",
            "docling.backend.msword_backend",
            "MsWordDocumentBackend",
            _DOCX_SAMPLE,
            "DOCX",
            "format-docx",
        ),
        (
            "pptx",
            "docling.backend.mspowerpoint_backend",
            "MsPowerpointDocumentBackend",
            _PPTX_SAMPLE,
            "PPTX",
            "format-pptx",
        ),
        (
            "openpyxl",
            "docling.backend.msexcel_backend",
            "MsExcelDocumentBackend",
            _XLSX_SAMPLE,
            "XLSX",
            "format-xlsx",
        ),
        (
            "pylatexenc",
            "docling.backend.latex_backend",
            "LatexDocumentBackend",
            _TEX_SAMPLE,
            "LATEX",
            "format-latex",
        ),
    ],
)
def test_backend_reports_missing_dependency_with_install_hint(
    blocked_module: str,
    backend_import: str,
    backend_class: str,
    sample: Path,
    fmt: str,
    extra: str,
) -> None:
    body = (
        "from pathlib import Path\n"
        f"from {backend_import} import {backend_class}\n"
        "from docling.datamodel.base_models import InputFormat\n"
        "from docling.datamodel.document import InputDocument\n"
        f"path = Path({str(sample)!r})\n"
        "try:\n"
        f"    InputDocument(path_or_stream=path, format=InputFormat.{fmt}, backend={backend_class})\n"
        "except ImportError as exc:\n"
        f"    assert {extra!r} in str(exc), str(exc)\n"
        "    print('actionable-error-raised')\n"
        "else:\n"
        f"    raise AssertionError('expected ImportError when {blocked_module} is missing')\n"
    )
    result = _run_with_blocked_module(blocked_module, body)
    assert result.returncode == 0, result.stderr
    assert "actionable-error-raised" in result.stdout
