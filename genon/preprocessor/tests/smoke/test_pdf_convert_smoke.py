"""
PDF 변환 smoke — attachment_processor 가 .ppt/.pptx 입력을 받아
PDF SDK (use_pdf_sdk=True) 와 LibreOffice (use_pdf_sdk=False) 양쪽 엔진에서
PDF 변환 후 vectors 까지 정상 반환되는지 확인.

`get_loader` → `convert_to_pdf(use_pdf_sdk=...)` → 후속 처리 → vectors 의
통합 흐름이 두 엔진 모두에서 깨지지 않는지(=fallback 회귀 방지) 검증.
"""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "sample_files"
PPT = SAMPLES / "ppt_sample.ppt"
PPTX = SAMPLES / "pptx_sample.pptx"


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample",
    [
        pytest.param(
            PPT, id="ppt",
            marks=pytest.mark.skipif(not PPT.exists(), reason="ppt_sample.ppt not found"),
        ),
        pytest.param(
            PPTX, id="pptx",
            marks=pytest.mark.skipif(not PPTX.exists(), reason="pptx_sample.pptx not found"),
        ),
    ],
)
@pytest.mark.parametrize("use_pdf_sdk", [True, False], ids=["sdk", "libreoffice"])
async def test_pdf_convert_smoke(basic_processor, sample: Path, use_pdf_sdk: bool):
    """입력 ext × {PDF SDK, LibreOffice} 모두 vectors 반환 확인."""
    dp = basic_processor()
    vectors = await dp(None, str(sample), use_pdf_sdk=use_pdf_sdk)

    assert isinstance(vectors, list) and len(vectors) >= 1
    v = vectors[0]
    if hasattr(v, "model_dump"):
        v = v.model_dump()
    assert "text" in v and isinstance(v["text"], str)
