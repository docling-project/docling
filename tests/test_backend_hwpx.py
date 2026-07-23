"""Tests for the native HWPX (Hangul Word Processor XML / OWPML) backend."""

import io
import zipfile
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from docling.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA
pytestmark = pytest.mark.cross_platform

DATA_DIR = Path("./tests/data/hwpx")


def get_converter() -> DocumentConverter:
    return DocumentConverter(allowed_formats=[InputFormat.HWPX])


def _convert(name: str) -> DoclingDocument:
    return get_converter().convert(DATA_DIR / name).document


# --- End-to-end groundtruth regression --------------------------------------


def test_e2e_hwpx_conversions():
    hwpx_paths = sorted(DATA_DIR.glob("*.hwpx"))
    assert hwpx_paths, "no HWPX fixtures found"
    converter = get_converter()

    for hwpx_path in hwpx_paths:
        gt_path = DATA_DIR / "groundtruth" / hwpx_path.name

        doc = converter.convert(hwpx_path).document

        pred_md = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


# --- Format detection --------------------------------------------------------


def test_hwpx_format_detection_by_extension():
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    assert dci._guess_format(DATA_DIR / "para-001.hwpx") == InputFormat.HWPX


def test_hwpx_format_detection_by_content_without_extension():
    """An HWPX renamed without an extension is still recognised by its
    ``application/hwp+zip`` mimetype entry."""
    data = (DATA_DIR / "para-001.hwpx").read_bytes()
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="mysteryfile", stream=io.BytesIO(data))
    assert dci._guess_format(stream) == InputFormat.HWPX


def test_generic_zip_is_not_detected_as_hwpx():
    """A plain ZIP without the HWPX marker must not be mistaken for HWPX."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="archive", stream=io.BytesIO(buf.getvalue()))
    assert dci._guess_format(stream) != InputFormat.HWPX


# --- Structural mapping ------------------------------------------------------


def test_paragraphs_preserve_run_text():
    doc = _convert("para-001.hwpx")
    body = " ".join(t.text for t in doc.texts)
    # Merged run text keeps both Hangul and the inline Hanja of the source run.
    assert "오호라" in body
    assert "乾坤" in body


def test_outline_paragraphs_map_to_headings_with_levels():
    doc = _convert("headings-synth.hwpx")
    headings = [
        (t.level, t.text) for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER
    ]
    assert headings == [
        (1, "제1장 서론"),
        (2, "1.1 연구 배경"),
        (3, "1.1.1 세부 사항"),
    ]
    # The trailing plain paragraph must stay body text, not a heading.
    assert any(t.label == DocItemLabel.TEXT for t in doc.texts)


def test_table_grid_dimensions_and_merged_cells():
    doc = _convert("table-text.hwpx")
    assert len(doc.tables) == 1
    data = doc.tables[0].data
    assert (data.num_rows, data.num_cols) == (3, 8)

    merged = [c for c in data.table_cells if c.col_span > 1 or c.row_span > 1]
    assert len(merged) == 2
    spans = {
        (c.start_row_offset_idx, c.start_col_offset_idx, c.col_span) for c in merged
    }
    # Two side-by-side header groups, each spanning four columns of the top row.
    assert spans == {(0, 0, 4), (0, 4, 4)}
    assert "기부 금액" in merged[0].text


def test_bulleted_outline_paragraphs_map_to_list_items():
    doc = _convert("footnote-01.hwpx")
    list_items = [t for t in doc.texts if t.label == DocItemLabel.LIST_ITEM]
    assert len(list_items) > 5
    assert any("개념" in t.text for t in list_items)


def test_footnotes_are_captured():
    doc = _convert("footnote-01.hwpx")
    footnotes = [t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert len(footnotes) >= 1
    assert all(t.text.strip() for t in footnotes)


def test_equation_script_becomes_formula():
    doc = _convert("eq-002.hwpx")
    formulas = [t for t in doc.texts if t.label == DocItemLabel.FORMULA]
    assert len(formulas) >= 1
    # The raw Hancom equation script is preserved verbatim (no LaTeX conversion).
    assert any("sqrt" in t.text for t in formulas)


def test_embedded_picture_carries_image_bytes():
    doc = _convert("test-image.hwpx")
    assert len(doc.pictures) >= 1
    assert any(p.image is not None for p in doc.pictures)


# --- Failure handling --------------------------------------------------------


def test_generic_zip_with_hwpx_extension_fails_cleanly():
    """Detection trusts the extension; the backend rejects a non-HWPX ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    result = get_converter().convert(
        DocumentStream(name="fake.hwpx", stream=io.BytesIO(buf.getvalue())),
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE


def test_binary_hwp_reports_unsupported():
    """A legacy binary ``.hwp`` (OLE/CFB compound file) is not a ZIP."""
    cfb = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512
    result = get_converter().convert(
        DocumentStream(name="legacy.hwpx", stream=io.BytesIO(cfb)),
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE


def test_empty_hwpx_fails_cleanly():
    result = get_converter().convert(
        DocumentStream(name="empty.hwpx", stream=io.BytesIO(b"")),
        raises_on_error=False,
    )
    assert result.status == ConversionStatus.FAILURE
