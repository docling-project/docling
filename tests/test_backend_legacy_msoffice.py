"""End-to-end tests for legacy binary Office format backends.

.doc, .xls, and .ppt files are converted via LibreOffice to their modern Open
XML equivalents before being parsed by the respective existing backend.
"""

from pathlib import Path

import pytest

from docling.backend.docx.drawingml.utils import get_libreoffice_cmd
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

pytestmark = pytest.mark.skipif(
    get_libreoffice_cmd() is None,
    reason="LibreOffice not available",
)

_CASES: list[tuple[InputFormat, str]] = [
    (InputFormat.DOC, "tests/data/doc/sources"),
    (InputFormat.XLS, "tests/data/xls/sources"),
    (InputFormat.PPT, "tests/data/ppt/sources"),
]


@pytest.mark.parametrize("fmt,sources_dir", _CASES)
def test_e2e_legacy_conversions(fmt: InputFormat, sources_dir: str):
    """Convert every legacy file in the sources directory and compare against ground truth.

    LibreOffice conversion produces slightly different image sizes across platforms, so
    bbox comparisons use fuzzy tolerances.
    """
    ext = fmt.value  # "doc", "xls", or "ppt"
    sources = Path(sources_dir)
    paths = sorted(sources.rglob(f"*.{ext}"))
    assert paths, f"No .{ext} test files found in {sources}"

    converter = DocumentConverter(allowed_formats=[fmt])

    for src_path in paths:
        gt_path = src_path.parent.parent / "groundtruth" / src_path.name

        doc = converter.convert(src_path).document

        pred_md = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), (
            f"Markdown mismatch for {src_path}"
        )

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            f"Indented-text mismatch for {src_path}"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE, fuzzy=True), (
            f"Document JSON mismatch for {src_path}"
        )
