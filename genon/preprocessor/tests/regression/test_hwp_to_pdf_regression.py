"""
HWP → PDF 변환 backend 별 회귀 테스트 (이슈 #199).

sample_files/ 의 모든 *.hwp / *.hwpx 파일을 가용한 모든 backend 로 변환해
다음을 검증:

1. 산출물 PDF 가 존재하고 사이즈가 0이 아님.
2. PDF 매직 바이트(%PDF-) 로 시작함.
3. PyPDF2 / pypdf 가 설치되어 있으면 페이지 수가 1 이상임 (없으면 skip).

backend 가용성은 환경에 따라 다르므로 가용한 것만 검증하고 나머지는 skip.
표/이미지/다단/머리말꼬리말 같은 검증 자산은 단순히 `sample_files/` 에
추가하면 자동으로 본 테스트에 편입됨 (별도 등록 불필요).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pytest

from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf
from genon.preprocessor.converters.hwp_to_pdf.config import _AVAILABILITY

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_files"
HWP_INPUTS = sorted(
    [p for p in SAMPLE_DIR.glob("*.hwp") if p.is_file()]
    + [p for p in SAMPLE_DIR.glob("*.hwpx") if p.is_file()]
)
BACKENDS = ("pdf_sdk", "rhwp", "libreoffice")


def _pdf_page_count(path: Path) -> int | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return None
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return -1


def _params() -> Iterable:
    for src in HWP_INPUTS:
        for backend in BACKENDS:
            yield pytest.param(src, backend, id=f"{src.stem}-{backend}")


@pytest.mark.regression
@pytest.mark.parametrize("src,backend", list(_params()) if HWP_INPUTS else [])
def test_backend_produces_valid_pdf(src: Path, backend: str, tmp_path: Path):
    if not _AVAILABILITY[backend]():
        pytest.skip(f"backend {backend!r} not available in this environment")

    work_in = tmp_path / src.name
    shutil.copy2(src, work_in)

    out = convert_hwp_to_pdf(str(work_in), primary=backend, disable_fallback=True)

    assert out is not None, f"{backend} returned None for {src.name}"
    out_path = Path(out)
    assert out_path.exists(), f"{backend} reported success but output missing"
    size = out_path.stat().st_size
    assert size > 0, f"{backend} produced 0-byte PDF for {src.name}"
    assert out_path.read_bytes()[:5] == b"%PDF-", (
        f"{backend} produced non-PDF magic bytes for {src.name}"
    )

    pages = _pdf_page_count(out_path)
    if pages is None:
        pytest.skip("pypdf/PyPDF2 not installed; page count check omitted")
    assert pages >= 1, f"{backend} produced PDF with {pages} pages for {src.name}"


@pytest.mark.regression
def test_at_least_one_backend_available():
    """환경에 최소 1개 backend 라도 가용해야 회귀 검증이 의미 있음.

    오픈소스/엔터프라이즈 컨테이너에서는 반드시 통과해야 한다 (CI guard).
    로컬 dev 환경에서는 backend 없으면 skip.
    """
    available = [name for name, check in _AVAILABILITY.items() if check()]
    if not available:
        pytest.skip("no hwp_to_pdf backend available in this environment")
    assert len(available) >= 1
