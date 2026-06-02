"""enrichment 배선 smoke 테스트.

배포 기본 YAML 로 parser / intelligent 프로세서를 실제 인스턴스화하여 enrichment 관련
배선이 startup 단계에서 깨지지 않는지 빠르게 확인한다. 인스턴스화는 (모델 다운로드를 포함해)
로컬 환경에 따라 실패할 수 있으므로 실패 시 skip 한다. **내부 서버로의 실제 추론 요청은
하지 않는다** — 생성된 enricher 의 배선(존재/타입)과 비활성 no-op 경로만 점검한다.

enrichment 배선/enrich_* 메서드는:
- intelligent_processor.DocumentProcessor 에 직접
- parser_processor.DocumentProcessor 의 내부 엔진 `._intel`(IntelligentDocumentProcessor) 에
존재한다.
"""

import asyncio

import pytest
from unittest.mock import MagicMock


@pytest.fixture(scope="module")
def intel_engine(intelligent_processor):
    """intelligent 파사드 인스턴스 (enrichment 배선 직접 보유)."""
    try:
        return intelligent_processor()
    except Exception as e:  # noqa: BLE001 - 환경 의존(모델/경로) 실패는 skip
        pytest.skip(f"intelligent_processor 인스턴스화 실패(환경 의존): {e}")


@pytest.fixture(scope="module")
def parser_engine(parser_processor):
    """parser 파사드 내부 엔진(._intel): 실제 enrichment 배선을 들고 있다."""
    try:
        proc = parser_processor()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"parser_processor 인스턴스화 실패(환경 의존): {e}")
    return proc._intel


# ── 배선 sanity ────────────────────────────────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("engine_fixture", ["intel_engine", "parser_engine"])
def test_enrichment_wiring(engine_fixture, request):
    engine = request.getfixturevalue(engine_fixture)
    assert hasattr(engine, "enrichment_options")
    # metadata_enricher 는 활성 설정에 따라 enricher 또는 None
    assert hasattr(engine, "metadata_enricher")
    assert isinstance(engine.custom_fields_enrichers, list)


# ── no-op enrich 경로 (네트워크 없이 동작) ────────────────────────────────────

@pytest.mark.smoke
@pytest.mark.parametrize("engine_fixture", ["intel_engine", "parser_engine"])
def test_enrich_custom_fields_noop_without_enrichers(engine_fixture, request):
    """custom_fields_enrichers 가 비어있으면 문서를 그대로 반환(네트워크 호출 없음)."""
    engine = request.getfixturevalue(engine_fixture)
    original = engine.custom_fields_enrichers
    engine.custom_fields_enrichers = []  # 실제 추론 호출이 일어나지 않도록 비움
    try:
        doc = MagicMock(name="doc")
        result = asyncio.run(engine.enrich_custom_fields(doc))
        assert result is doc
    finally:
        engine.custom_fields_enrichers = original


@pytest.mark.smoke
@pytest.mark.parametrize("engine_fixture", ["intel_engine", "parser_engine"])
def test_enrich_metadata_noop_when_disabled(engine_fixture, request):
    """metadata_enricher 가 None 이면 문서를 그대로 반환(네트워크 호출 없음)."""
    engine = request.getfixturevalue(engine_fixture)
    original = getattr(engine, "metadata_enricher", None)
    engine.metadata_enricher = None
    try:
        doc = MagicMock(name="doc")
        result = asyncio.run(engine.enrich_metadata(doc))
        assert result is doc
    finally:
        engine.metadata_enricher = original
