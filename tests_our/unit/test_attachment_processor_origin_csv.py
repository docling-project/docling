from __future__ import annotations

from pathlib import Path
import asyncio
import sys
import pytest


CSV_SAMPLE = Path(__file__).resolve().parents[2] / "sample_files" / "sample.csv"


class _DummyRequest:
    async def is_disconnected(self) -> bool:  # pragma: no cover
        return False


def _import_origin():
    try:
        from doc_parser.doc_preprocessors.basic_processor import DocumentProcessor
        return DocumentProcessor
    except ModuleNotFoundError:
        # 보정: 테스트 실행 루트가 다를 수 있어 상위 경로 추가
        sys.path.append(str(Path(__file__).resolve().parents[3]))
        from doc_parser.doc_preprocessors.basic_processor import DocumentProcessor
        return DocumentProcessor


@pytest.mark.unit
def test_basic_processor_origin_handles_csv_sample():
    if not CSV_SAMPLE.exists():
        pytest.skip(f"sample not found: {CSV_SAMPLE}")

    DocumentProcessor = _import_origin()
    dp = DocumentProcessor()

    async def _run():
        return await dp(_DummyRequest(), str(CSV_SAMPLE))

    vectors = asyncio.run(_run())

    # 벡터/청크 기본 검증
    assert isinstance(vectors, list)
    assert len(vectors) >= 1

    v0 = vectors[0]
    text = getattr(v0, "text", None) if hasattr(v0, "text") else v0.get("text")
    assert isinstance(text, str)
    assert text.startswith("[DA] ") or len(text) > 0


