"""
Smoke tests for facade/parser_processor.py DocumentProcessor.

Calls DocumentProcessor.__call__ with real sample files and validates
the output schema. Each parametrized case is skipped when no matching
sample files are found in sample_files/.
"""
import os
from pathlib import Path

import pytest

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_files"
REQUIRED_KEYS = {"elements", "usage"}
ELEMENT_KEYS = {"id", "page", "category", "content", "coordinates"}


def _samples(ext: str) -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.rglob(f"*{ext}"))


def _validate_result(result: dict) -> None:
    assert isinstance(result, dict)
    for key in REQUIRED_KEYS:
        assert key in result, f"result missing key: {key!r}"
    assert isinstance(result["elements"], list), "elements must be a list"
    assert isinstance(result["usage"]["pages"], int), "usage.pages must be int"
    assert result["usage"]["pages"] >= 1
    for element in result["elements"]:
        for key in ELEMENT_KEYS:
            assert key in element, f"element missing key: {key!r}"


@pytest.fixture(scope="module")
def dp(parser_processor):
    return parser_processor()


# ─── DOCX ─────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".docx"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_docx_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)
    for element in result["elements"]:
        assert element["coordinates"] == [], "docx elements must have empty coordinates"


# ─── HWP ──────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".hwp"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_hwp_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)


# ─── HWPX ─────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".hwpx"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_hwpx_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)


# ─── CSV ──────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".csv"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_csv_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)
    assert all(e["category"] == "table" for e in result["elements"])


# ─── XLSX ─────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".xlsx"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_xlsx_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)
    assert all(e["category"] == "table" for e in result["elements"])


# ─── PDF ──────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.skipif(
    not os.environ.get("GENOS_LAYOUT_AVAILABLE"),
    reason="GENOS_LAYOUT_AVAILABLE not set; skipping PDF smoke test requiring internal layout endpoint",
)
@pytest.mark.parametrize("sample", _samples(".pdf"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_pdf_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)


# ─── Markdown ─────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".md"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_md_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)


# ─── PPTX ─────────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("sample", _samples(".pptx"), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_pptx_smoke(dp, sample):
    result = await dp(None, str(sample))
    _validate_result(result)
