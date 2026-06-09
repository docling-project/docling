"""
Unit tests for facade/convert_processor.py.
Covers enrichment error wrapping and precheck config defaults.
"""
import pytest
from unittest.mock import MagicMock, patch


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def convert_processor_class():
    mod = pytest.importorskip("facade.convert_processor")
    return mod.DocumentProcessor


@pytest.fixture
def processor(convert_processor_class):
    return convert_processor_class()


# ─── enrichment() 에러 래핑 ──────────────────────────────────────────────────

@pytest.mark.unit
def test_convert_enrichment_llm_error_is_rethrown_as_genos_exception():
    """convert_processor.enrichment()에서 LLMApiError가 GenosServiceException으로 래핑되는지 확인"""
    from facade.convert_processor import DocumentProcessor, GenosServiceException
    from docling.prompts.prompt_manager import LLMApiError

    proc = object.__new__(DocumentProcessor)
    proc.enrichment_options = MagicMock()
    raw_error = '{"object":"error","message":"context exceeded","type":"BadRequestError","param":"prompt","code":400}'

    with patch(
        "facade.convert_processor.enrich_document",
        side_effect=LLMApiError(raw_error, status_code=400),
    ):
        with pytest.raises(GenosServiceException) as exc_info:
            proc.enrichment(MagicMock())

    assert exc_info.value.error_msg == raw_error


# ─── DataEnrichmentOptions precheck 기본값 ──────────────────────────────────

@pytest.mark.unit
def test_convert_enrichment_options_precheck_defaults(processor):
    """DataEnrichmentOptions에 precheck 필드가 False 기본값으로 설정되어 있는지 확인"""
    opts = processor.enrichment_options
    assert opts.toc_precheck_enabled is False
    assert opts.toc_max_context_tokens == 128000
    assert opts.toc_completion_reserved_tokens == 12000
    assert opts.metadata_precheck_enabled is False
    assert opts.metadata_max_context_tokens == 128000
    assert opts.metadata_completion_reserved_tokens == 12000


@pytest.mark.unit
def test_convert_metadata_enricher_is_wired_from_yaml(processor):
    """metadata YAML 상세 설정을 facade enricher에 연결하고 내장 추출을 끄는지 확인."""
    enricher = processor.metadata_enricher

    assert enricher is not None
    assert processor.enrichment_options.extract_metadata is False
    assert enricher._output_fields == ["created_date", "authors"]
    assert enricher._parser_cfg == {"type": "json"}
    assert enricher._pages == [1, 4]
    assert "created_date" in enricher._user_prompt
    assert "authors" in enricher._user_prompt
