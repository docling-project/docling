"""
HWPX smoke test — 단일 샘플로 파이프라인이 죽지 않는지만 확인.
벡터 스키마 검증이나 전체 샘플 parametrize는 하지 않는다.
"""
from pathlib import Path
import pytest

SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "hwpx_sample.hwpx"


@pytest.mark.smoke
@pytest.mark.skipif(not SAMPLE.exists(), reason="hwpx_sample.hwpx not found")
def test_hwpx_load_and_chunk(basic_processor):
    """HWPX 파일 로드 → 청크 분할이 에러 없이 완료되는지 확인."""
    dp = basic_processor()

    # HWPX는 DocumentProcessor.load_documents가 아닌
    # hwp_processor.load_documents로 라우팅된다.
    doc = dp.hwp_processor.load_documents(str(SAMPLE))
    assert doc is not None

    chunks, page_chunk_counts = dp.hwp_processor.split_documents(doc)
    assert isinstance(chunks, list) and len(chunks) >= 1
