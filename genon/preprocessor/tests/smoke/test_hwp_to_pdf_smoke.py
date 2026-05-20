"""
HWP → PDF 변환 backend 별 smoke 테스트.

`convert_hwp_to_pdf(primary=...)` 진입점을 backend 3종(pdf_sdk / rhwp / libreoffice)에
대해 직접 호출하고, 가용한 backend 만 실행해 PDF 산출물을 검증한다.
가용 환경(컨테이너 / CI)에서만 의미 있는 검사이며, 로컬 macOS 처럼 자산이 없는
환경에서는 자동으로 `pytest.skip` 처리되어 false negative 가 발생하지 않는다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf
from genon.preprocessor.converters.hwp_to_pdf.config import _AVAILABILITY

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "sample_files"
HWP = SAMPLES / "hwp_sample.hwp"
HWPX = SAMPLES / "hwpx_sample.hwpx"


def _skip_if_unavailable(backend: str) -> None:
    if not _AVAILABILITY[backend]():
        pytest.skip(f"backend {backend!r} not available in this environment")


@pytest.fixture
def isolated_input(tmp_path: Path):
    """샘플 파일을 tmp_path 로 복사해 in-place 산출물 충돌 방지."""
    def _copy(src: Path) -> Path:
        dst = tmp_path / src.name
        shutil.copy2(src, dst)
        return dst
    return _copy


@pytest.mark.smoke
@pytest.mark.parametrize("backend", ["pdf_sdk", "rhwp", "libreoffice"])
@pytest.mark.parametrize(
    "sample",
    [
        pytest.param(HWP, id="hwp", marks=pytest.mark.skipif(not HWP.exists(), reason="hwp_sample.hwp not found")),
        pytest.param(HWPX, id="hwpx", marks=pytest.mark.skipif(not HWPX.exists(), reason="hwpx_sample.hwpx not found")),
    ],
)
def test_hwp_to_pdf_per_backend(backend: str, sample: Path, isolated_input):
    _skip_if_unavailable(backend)
    src = isolated_input(sample)

    out = convert_hwp_to_pdf(str(src), primary=backend, disable_fallback=True)

    assert out is not None, f"{backend} returned None for {sample.name}"
    out_path = Path(out)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    # PDF 매직 바이트 (%PDF-) 확인
    assert out_path.read_bytes()[:5] == b"%PDF-", f"{backend} produced non-PDF bytes for {sample.name}"


@pytest.mark.smoke
@pytest.mark.skipif(not HWP.exists(), reason="hwp_sample.hwp not found")
def test_default_chain_returns_pdf(isolated_input):
    """primary 미지정 시 auto-default chain 으로 한 backend 라도 성공해야 함."""
    if not any(check() for check in _AVAILABILITY.values()):
        pytest.skip("no hwp_to_pdf backend available in this environment")

    src = isolated_input(HWP)
    out = convert_hwp_to_pdf(str(src))
    assert out is not None
    assert Path(out).read_bytes()[:5] == b"%PDF-"
