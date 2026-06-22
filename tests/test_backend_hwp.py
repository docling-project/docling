"""Tests for the experimental HWP/HWPX backend.

The format-detection tests run without the optional ``hangulang`` dependency.
The end-to-end conversion tests are skipped automatically when ``hangulang`` is
not installed (``pip install docling[format-hwp]``).
"""

from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter, HwpFormatOption

DATA_DIR = Path(__file__).parent / "data" / "hwp"


# --- Format detection (no hangulang required) --------------------------------


def test_hwp_guess_format_by_extension(tmp_path: Path):
    dci = _DocumentConversionInput(path_or_stream_iterator=[])

    hwp_path = tmp_path / "sample.hwp"
    hwp_path.write_bytes(b"fake-hwp")
    assert dci._guess_format(hwp_path) == InputFormat.HWP

    hwpx_path = tmp_path / "sample.hwpx"
    hwpx_path.write_bytes(b"PK\x03\x04fake-hwpx")
    assert dci._guess_format(hwpx_path) == InputFormat.HWP


def test_hwp_guess_format_stream_by_extension():
    stream = DocumentStream(name="sample.hwpx", stream=BytesIO(b"PK\x03\x04fake"))
    dci = _DocumentConversionInput(path_or_stream_iterator=[])
    assert dci._guess_format(stream) == InputFormat.HWP


def test_hwp_registered_in_format_machinery():
    assert "hwp" in InputFormat.HWP.value
    assert (
        "hwp"
        in __import__(
            "docling.datamodel.base_models", fromlist=["FormatToExtensions"]
        ).FormatToExtensions[InputFormat.HWP]
    )
    assert (
        "hwpx"
        in __import__(
            "docling.datamodel.base_models", fromlist=["FormatToExtensions"]
        ).FormatToExtensions[InputFormat.HWP]
    )


# --- End-to-end conversion (requires hangulang) ------------------------------

hangulang = pytest.importorskip(
    "hangulang", reason="install the optional 'format-hwp' extra to run these tests"
)


def _converter() -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.HWP],
        format_options={InputFormat.HWP: HwpFormatOption()},
    )


# para-001.hwp and para-001.hwpx are the same source saved in both formats,
# so they must yield the same leading paragraph text.
_PARA_001_PREFIX = "오호라, 건곤(乾坤)이 혼합하고"


@pytest.mark.parametrize("name", ["para-001.hwp", "para-001.hwpx"])
def test_hwp_paragraphs_convert(name: str):
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"missing fixture: {name}")

    result = _converter().convert(path)

    assert result.input.format == InputFormat.HWP
    assert result.document.texts[0].text.startswith(_PARA_001_PREFIX)
    md = result.document.export_to_markdown()
    assert isinstance(md, str) and md.strip()


def test_hwp_table_structure():
    path = DATA_DIR / "table-001.hwp"
    if not path.exists():
        pytest.skip("missing fixture: table-001.hwp")

    result = _converter().convert(path)
    assert result.input.format == InputFormat.HWP
    assert len(result.document.tables) == 1

    table = result.document.tables[0]
    assert table.data.num_rows == 19
    assert table.data.num_cols == 9
    grid = [[cell.text for cell in row] for row in table.data.grid]
    assert grid[0][0] == "구 분"
    assert "5월" in grid[0]


def test_hwp_convert_from_stream():
    path = DATA_DIR / "para-001.hwpx"
    if not path.exists():
        pytest.skip("missing fixture: para-001.hwpx")

    stream = DocumentStream(name="para-001.hwpx", stream=BytesIO(path.read_bytes()))
    result = _converter().convert(stream)
    assert result.input.format == InputFormat.HWP
    assert result.document.texts[0].text.startswith(_PARA_001_PREFIX)
