import json
from unittest.mock import patch

import pytest

from docling.prompts.prompt_manager import LLMApiError, PromptManager


@pytest.mark.unit
def test_prompt_precheck_blocks_before_api_call():
    pm = PromptManager(
        custom_api_configs={
            "toc_extraction": {
                "provider": "custom",
                "api_base_url": "http://dummy.local/v1/chat/completions",
                "model": "dummy-model",
                "precheck_enabled": True,
                "max_context_tokens": 128000,
                "completion_reserved_tokens": 12000,
            }
        }
    )

    with patch.object(
        PromptManager,
        "_rough_token_count",
        return_value=169000,
    ), patch("docling.prompts.prompt_manager.requests.post") as post_mock:
        with pytest.raises(LLMApiError) as exc_info:
            pm.call_ai_model(
                category="toc_extraction",
                prompt_type="korean_document",
                raise_on_error=True,
                raw_text="테스트 본문",
            )

    post_mock.assert_not_called()
    payload = json.loads(exc_info.value.raw_error_message)
    assert payload["object"] == "error"
    assert payload["type"] == "BadRequestError"
    assert payload["param"] == "prompt"
    assert payload["code"] == 400
    assert "프롬프트 입력 토큰 (169000) 초과" in payload["message"]
    assert "(128000 - reserved 12000)." in payload["message"]


@pytest.mark.unit
def test_repetition_penalty_added_to_payload():
    """toc 설정의 repetition_penalty 가 API payload 로 전달되는지 확인."""
    pm = PromptManager(
        custom_api_configs={
            "toc_extraction": {
                "provider": "custom",
                "api_base_url": "http://dummy.local/v1/chat/completions",
                "model": "dummy-model",
                "repetition_penalty": 1.1,
            }
        }
    )

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    with patch("docling.prompts.prompt_manager.requests.post", return_value=_Resp()) as post_mock:
        pm.call_ai_model(
            category="toc_extraction",
            prompt_type="korean_document",
            raw_text="본문",
        )

    assert post_mock.call_count == 1
    payload = post_mock.call_args.kwargs["json"]
    assert payload["repetition_penalty"] == 1.1


@pytest.mark.unit
def test_repetition_penalty_absent_when_not_configured():
    pm = PromptManager(
        custom_api_configs={
            "toc_extraction": {
                "provider": "custom",
                "api_base_url": "http://dummy.local/v1/chat/completions",
                "model": "dummy-model",
            }
        }
    )

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    with patch("docling.prompts.prompt_manager.requests.post", return_value=_Resp()) as post_mock:
        pm.call_ai_model(
            category="toc_extraction",
            prompt_type="korean_document",
            raw_text="본문",
        )

    payload = post_mock.call_args.kwargs["json"]
    assert "repetition_penalty" not in payload


@pytest.mark.unit
def test_format_user_prompt_supports_double_brace_raw_text_alias():
    pm = PromptManager()

    rendered = pm.format_user_prompt(
        category="toc_extraction",
        prompt_type="korean_document",
        custom_user="문서:\n{{raw_text}}",
        raw_text="샘플 본문",
    )

    assert rendered == "문서:\n샘플 본문"


@pytest.mark.unit
def test_format_user_prompt_renders_prior_toc_and_raw_text():
    """통합 TOC 프롬프트 메커니즘: {{prior_toc}} + {{raw_text}}(둘 다 이중 중괄호) 동시 렌더."""
    pm = PromptManager()
    tmpl = "<previous_outline>\n{{prior_toc}}\n</previous_outline>\n문서:\n{{raw_text}}"

    # 첫 추출: prior_toc="" → 빈 outline, 본문 주입
    r_first = pm.format_user_prompt(
        category="toc_extraction", prompt_type="korean_document",
        custom_user=tmpl, raw_text="본문A", prior_toc="",
    )
    assert r_first is not None
    assert "본문A" in r_first
    assert "{raw_text}" not in r_first and "{prior_toc}" not in r_first  # 미치환 잔존 없음

    # 이어쓰기: prior 값에 중괄호가 있어도 안전(치환값은 재파싱되지 않음)
    r_cont = pm.format_user_prompt(
        category="toc_extraction", prompt_type="korean_document",
        custom_user=tmpl, raw_text="본문B", prior_toc="TITLE:DOC\n1. 개요 {x}",
    )
    assert r_cont is not None
    assert "본문B" in r_cont
    assert "1. 개요 {x}" in r_cont


@pytest.mark.unit
def test_format_user_prompt_keeps_other_escaped_braces():
    pm = PromptManager()

    rendered = pm.format_user_prompt(
        category="toc_extraction",
        prompt_type="korean_document",
        custom_user='JSON 예시: {{"key":"value"}}\n문서: {{raw_text}}',
        raw_text="본문",
    )

    assert rendered == 'JSON 예시: {"key":"value"}\n문서: 본문'
