# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, GroupLabel

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
LATEX_DATA_DIR = Path("./tests/data/latex/sources/")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--large-scale",
        action="store_true",
        help="Run the opt-in arXiv LaTeX corpus test.",
    )
    parser.addoption(
        "--max-items",
        type=int,
        default=None,
        help="Stop the arXiv LaTeX corpus test after this many source files.",
    )
    parser.addoption(
        "--write-success",
        action="store_true",
        help="Save successful arXiv LaTeX sources and DocLang output.",
    )
    parser.addoption(
        "--recheck-failures",
        action="store_true",
        help="Convert again the arXiv LaTeX files that failed in an earlier run.",
    )
    parser.addoption(
        "--corpus-shard",
        action="append",
        type=int,
        default=None,
        help="Read only this shard (0-79) of the arXiv LaTeX corpus. Repeat it to "
        "read more shards.",
    )
    parser.addoption(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Write the arXiv LaTeX corpus manifest, report, and saved files to "
        "this folder instead of the repository root.",
    )


@pytest.fixture(scope="module")
def latex_paths() -> list[Path]:
    """Find all LaTeX files in the test data directory."""
    directory = Path("./tests/data/latex/sources/")
    if not directory.exists():
        return []

    paths = list(directory.glob("*.tex"))

    for subdir in directory.iterdir():
        if subdir.is_dir():
            if (subdir / "main.tex").exists():
                paths.append(subdir / "main.tex")
            elif (subdir / f"arxiv_{subdir.name}.tex").exists():
                paths.append(subdir / f"arxiv_{subdir.name}.tex")

    return sorted(paths)
