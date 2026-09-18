# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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
_MSG_SAMPLE = Path(__file__).parent / "data" / "email" / "sources" / "msg_simple.msg"
_HTML_SAMPLE = Path(__file__).parent / "data" / "html" / "sources" / "hyperlink_01.html"
_MD_SAMPLE = Path(__file__).parent / "data" / "md" / "sources" / "mixed.md"
_DOCX_SAMPLE = Path(__file__).parent / "data" / "docx" / "sources" / "word_tables.docx"
_PPTX_SAMPLE = (
    Path(__file__).parent / "data" / "pptx" / "sources" / "powerpoint_sample.pptx"
)
_XLSX_SAMPLE = Path(__file__).parent / "data" / "xlsx" / "sources" / "xlsx_01.xlsx"
_TEX_SAMPLE = (
    Path(__file__).parent / "data" / "latex" / "sources" / "1706.03762" / "main.tex"
)
_JATS_SAMPLE = Path(__file__).parent / "data" / "jats" / "sources" / "pone.0234687.nxml"
_USPTO_SAMPLE = Path(__file__).parent / "data" / "uspto" / "sources" / "ipg08672134.xml"


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
    [
        "mailparser",
        "oxmsg",
        "marko",
        "docx",
        "pptx",
        "openpyxl",
        "pylatexenc",
        "bs4",
        "pypdfium2",
    ],
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
            "oxmsg",
            "docling.backend.email_backend",
            "EmailDocumentBackend",
            _MSG_SAMPLE,
            "EMAIL",
            "format-email",
        ),
        (
            "bs4",
            "docling.backend.html_backend",
            "HTMLDocumentBackend",
            _HTML_SAMPLE,
            "HTML",
            "format-html",
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
        (
            "bs4",
            "docling.backend.xml.jats_backend",
            "JatsDocumentBackend",
            _JATS_SAMPLE,
            "XML_JATS",
            "format-xml-jats",
        ),
        (
            "bs4",
            "docling.backend.xml.uspto_backend",
            "PatentUsptoDocumentBackend",
            _USPTO_SAMPLE,
            "XML_USPTO",
            "format-xml-uspto",
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


def test_service_client_imports_without_pdf_pipeline_dependency() -> None:
    # The service client reaches docling.utils.pdf_outline (via noop_backend ->
    # datamodel.document, which needs only the _PdfOutlineItem model), so a
    # module-level pypdfium2 import there breaks `import docling.service_client`
    # on any slim install that omits the PDF pipeline. Guard the deferred import.
    result = _run_with_blocked_module(
        "pypdfium2",
        "import docling.service_client\n"
        "from docling.datamodel.service.options import ConvertDocumentsOptions\n",
    )
    assert result.returncode == 0, result.stderr


def test_docling_parse_backend_operates_without_pypdfium2() -> None:
    sample = Path(__file__).parent / "data" / "pdf" / "bookmark_sample.pdf"
    body = (
        "from pathlib import Path\n"
        "from docling.backend.docling_parse_backend import "
        "ThreadedDoclingParseDocumentBackend\n"
        "from docling.datamodel.base_models import InputFormat\n"
        "from docling.datamodel.document import InputDocument\n"
        f"path = Path({str(sample)!r})\n"
        "in_doc = InputDocument(path_or_stream=path, format=InputFormat.PDF, "
        "backend=ThreadedDoclingParseDocumentBackend)\n"
        "backend = in_doc._backend\n"
        "page = next(backend.iter_pages())\n"
        "assert page.get_page_image().width > 0\n"
        "backend.unload()\n"
    )

    result = _run_with_blocked_module("pypdfium2", body)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("backend_import", "backend_class", "sample", "fmt"),
    [
        (
            "docling.backend.msexcel_backend",
            "MsExcelDocumentBackend",
            _XLSX_SAMPLE,
            "XLSX",
        ),
        (
            "docling.backend.msword_backend",
            "MsWordDocumentBackend",
            _DOCX_SAMPLE,
            "DOCX",
        ),
        (
            "docling.backend.mspowerpoint_backend",
            "MsPowerpointDocumentBackend",
            _PPTX_SAMPLE,
            "PPTX",
        ),
        (
            "docling.backend.latex_backend",
            "LatexDocumentBackend",
            _TEX_SAMPLE,
            "LATEX",
        ),
    ],
)
def test_office_and_latex_backends_convert_without_pypdfium2(
    backend_import: str,
    backend_class: str,
    sample: Path,
    fmt: str,
) -> None:
    # pypdfium2 ships with the PDF extras, and these backends need it only to
    # rasterize embedded charts and EMF/WMF pictures. Text, tables, and chart
    # data must still convert without it, the way they do when LibreOffice is
    # absent.
    body = (
        "from pathlib import Path\n"
        f"from {backend_import} import {backend_class}\n"
        "from docling.datamodel.base_models import InputFormat\n"
        "from docling.datamodel.document import InputDocument\n"
        f"path = Path({str(sample)!r})\n"
        "in_doc = InputDocument(path_or_stream=path, "
        f"format=InputFormat.{fmt}, backend={backend_class})\n"
        "doc = in_doc._backend.convert()\n"
        "assert doc.texts or doc.tables\n"
    )

    result = _run_with_blocked_module("pypdfium2", body)

    assert result.returncode == 0, result.stderr


def test_powerpoint_uses_shared_pypdfium2_guard() -> None:
    # PowerPoint used to gate the shared converter factory behind its own
    # pypdfium2 import. On slim installs that bypassed the factory entirely, so
    # users never saw its actionable install hint. Multiple backend instances
    # must now reach the shared guard while the warning is still emitted once.
    body = (
        "import logging\n"
        "from docling.backend.mspowerpoint_backend import "
        "MsPowerpointDocumentBackend\n"
        "logging.basicConfig(level=logging.WARNING, format='%(message)s')\n"
        "for _ in range(3):\n"
        "    backend = object.__new__(MsPowerpointDocumentBackend)\n"
        "    backend.pptx_to_pdf_converter = None\n"
        "    backend.pptx_to_pdf_converter_init = False\n"
        "    assert backend._get_libreoffice_converter() is None\n"
    )

    result = _run_with_blocked_module("pypdfium2", body)

    assert result.returncode == 0, result.stderr
    assert result.stderr.count("format-pdf-pypdfium2") == 1, result.stderr


def test_converter_constructs_without_chart_extraction_dependency() -> None:
    """Importing DocumentConverter must not require transformers.

    docling-slim[models-onnxruntime] ships without torch or transformers.
    This regression test verifies the import is fully deferred behind the
    `do_chart_extraction` flag.

    Note: torch cannot be blocked via sys.modules here because scipy (an
    unrelated transitive dependency) also inspects sys.modules["torch"] and
    crashes on None.
    """
    result = _run_with_blocked_module(
        "transformers",
        "from docling.document_converter import DocumentConverter\nDocumentConverter()\n",
    )
    assert result.returncode == 0, result.stderr
