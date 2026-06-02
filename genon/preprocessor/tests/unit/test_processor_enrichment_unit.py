"""프로세서 레벨 enrichment 배선 + 로깅 단위 테스트.

비동기 enrich_metadata / enrich_custom_fields delegation·no-op 경로와, 프로세서 공통
setup_logging(yaml `log_level`) 레벨 매핑을 검증한다.

enrich_metadata/enrich_custom_fields 는 enrichment 배선을 들고 있는 클래스에 정의돼 있다:
- `intelligent_processor.DocumentProcessor` (단독 인텔리전트 파사드)
- `parser_processor.IntelligentDocumentProcessor` (parser 파사드가 내부에서 위임하는 엔진)
setup_logging 은 세 facade DocumentProcessor 가 동일하게 갖는다.

모두 네트워크 비의존이다(enricher 는 AsyncMock 또는 미주입; setup_logging 은 logging 상태만).
facade import 는 무거운 네이티브 의존(weasyprint→libgobject 등)을 끌어오며, 미설치 환경에서
ImportError 가 아닌 OSError 가 날 수 있어 importorskip 으로 못 거른다. 모듈별로 독립 import 하고
실패 시 해당 부분만 skip 한다(CI 는 시스템 deps 설치돼 정상 import).
"""

import asyncio
import importlib
import logging

import pytest
from unittest.mock import AsyncMock, MagicMock


def _try_import(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001 - 환경 의존 import 실패
        return None


_intel_mod = _try_import("facade.intelligent_processor")
_parser_mod = _try_import("facade.parser_processor")
_attach_mod = _try_import("facade.attachment_processor")

if _intel_mod is None or _parser_mod is None:
    pytest.skip(
        "facade intelligent/parser 프로세서 import 불가(환경 의존)",
        allow_module_level=True,
    )


# enrich_* 를 정의한 클래스들(배선 보유 주체).
_ENRICH_PROCESSORS = [
    pytest.param(_intel_mod.DocumentProcessor, id="intelligent"),
    pytest.param(_parser_mod.IntelligentDocumentProcessor, id="parser_intel"),
]

# setup_logging 을 가진 facade DocumentProcessor 들.
_LOGGING_PROCESSORS = [
    pytest.param(_intel_mod.DocumentProcessor, id="intelligent"),
    pytest.param(_parser_mod.DocumentProcessor, id="parser"),
]
if _attach_mod is not None:
    _LOGGING_PROCESSORS.append(pytest.param(_attach_mod.DocumentProcessor, id="attachment"))


def _bare(cls):
    """__init__ 우회 인스턴스 (네트워크/모델 로딩 없이 메서드만 테스트)."""
    return object.__new__(cls)


# ── enrich_metadata ────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("cls", _ENRICH_PROCESSORS)
class TestEnrichMetadata:
    def test_noop_when_enricher_absent(self, cls):
        proc = _bare(cls)  # metadata_enricher 속성 없음 → getattr(None)
        doc = MagicMock()
        result = asyncio.run(proc.enrich_metadata(doc))
        assert result is doc

    def test_noop_when_enricher_none(self, cls):
        proc = _bare(cls)
        proc.metadata_enricher = None
        doc = MagicMock()
        result = asyncio.run(proc.enrich_metadata(doc))
        assert result is doc

    def test_delegates_to_enricher_and_passes_context(self, cls):
        proc = _bare(cls)
        sentinel = MagicMock(name="enriched_doc")
        enricher = MagicMock()
        enricher.enrich = AsyncMock(return_value=sentinel)
        proc.metadata_enricher = enricher

        doc = MagicMock()
        ctx = {}
        result = asyncio.run(proc.enrich_metadata(doc, _enrichment_context=ctx))

        assert result is sentinel
        enricher.enrich.assert_awaited_once()
        called_args, called_kwargs = enricher.enrich.call_args
        assert called_args[0] is doc
        assert called_kwargs["_enrichment_context"] is ctx


# ── enrich_custom_fields ───────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("cls", _ENRICH_PROCESSORS)
class TestEnrichCustomFields:
    def test_noop_when_empty_list(self, cls):
        proc = _bare(cls)
        proc.custom_fields_enrichers = []
        doc = MagicMock()
        result = asyncio.run(proc.enrich_custom_fields(doc))
        assert result is doc

    def test_runs_each_enricher_in_sequence(self, cls):
        proc = _bare(cls)
        doc0 = MagicMock(name="doc0")
        doc1 = MagicMock(name="doc1")
        doc2 = MagicMock(name="doc2")
        e1 = MagicMock()
        e1.enrich = AsyncMock(return_value=doc1)
        e2 = MagicMock()
        e2.enrich = AsyncMock(return_value=doc2)
        proc.custom_fields_enrichers = [e1, e2]

        result = asyncio.run(proc.enrich_custom_fields(doc0))

        assert result is doc2
        e1.enrich.assert_awaited_once()
        e2.enrich.assert_awaited_once()
        # 첫 enricher 결과가 다음 enricher 입력으로 전달된다.
        assert e2.enrich.call_args.args[0] is doc1


# ── setup_logging 레벨 매핑 (yaml defaults.log_level) ──────────────────────────

@pytest.fixture
def _restore_logging():
    """setup_logging 이 전역 logging 상태를 바꾸므로 테스트 후 원복한다."""
    root = logging.getLogger()
    prev_level = root.level
    prev_disable = logging.root.manager.disable
    yield
    logging.disable(logging.NOTSET)
    if prev_disable:
        logging.disable(prev_disable)
    root.setLevel(prev_level)


@pytest.mark.unit
@pytest.mark.parametrize("cls", _LOGGING_PROCESSORS)
@pytest.mark.usefixtures("_restore_logging")
class TestSetupLogging:
    @pytest.mark.parametrize("level_num,expected", [
        (5, logging.DEBUG),
        (4, logging.INFO),
        (3, logging.WARNING),
        (2, logging.ERROR),
        (1, logging.CRITICAL),
    ])
    def test_known_levels(self, cls, level_num, expected):
        proc = _bare(cls)
        proc.setup_logging(level_num)
        assert logging.getLogger().level == expected

    def test_nolog_disables_logging(self, cls):
        proc = _bare(cls)
        proc.setup_logging(0)
        # 0(NOLOG) → logging.disable(CRITICAL)
        assert logging.root.manager.disable == logging.CRITICAL

    def test_unknown_level_falls_back_to_info(self, cls):
        proc = _bare(cls)
        proc.setup_logging(99)
        assert logging.getLogger().level == logging.INFO
